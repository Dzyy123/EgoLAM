#!/usr/bin/env python
"""Build the LARY-Causal EgoVerse image-pair dataset from RLDS tfrecords.

Output layout (under --out-root, which should be $DATA_DIR/regression/egoverse):

  images/<subset>/<episode_id>/frame_<t:06d>.jpg      raw JPEG bytes copied from RLDS
  frames_<split>.csv                                   one row per sampled frame
  pairs_<split>_stride<k>.csv                          one row per (t, t+k) pair
  actions/<subset>/<episode_id>/a_<t>_to_<t+k>.npy     12-D relative hand action (head frame at t)

Every pair row carries the four causal variables used by causal/metrics.py:

  a_*   hand action     : per hand dxyz + rotvec of the t+k pose re-expressed in the head frame at t   [12]
  n_*   camera nuisance : relative head motion between t and t+k (translation in head_t frame + rotvec)    [6]
  s_*   current state   : ee pose at t in head frame                                                       [12]
  embodiment            : human_bimanual / aria_bimanual / MECKA_BIMANUAL / eva_bimanual / scale

Splits: train  <- RLDS 'train' shards (subset)
        val    <- RLDS 'seen_test'
        unseen <- RLDS 'unseen_test'
"""
import argparse
import glob
import json
import os
import random
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SUBSETS = {
    "human": "human_bimanual_front_1",
    "aria": "aria_bimanual_front_1",
    "mecka": "mecka_bimanual_front_1",
    "eva": "eva_bimanual_front_1_left_wrist_right_wrist",
    "scale": "scale_front_1",
}
SPLIT_TO_RLDS = {"train": "train", "val": "seen_test", "unseen": "unseen_test"}
ACTION_STRIDE = 3          # actions_cartesian index j <-> frame t + 3*j
CHUNK_IDX = {15: 5, 30: 10}  # frame gap -> chunk index
EE_COLS = [f"{h}_{c}" for h in ("l", "r") for c in ("x", "y", "z", "yaw", "pitch", "roll")]
A_COLS = [f"{h}_{c}" for h in ("l", "r") for c in ("x", "y", "z", "rx", "ry", "rz")]   # action = dxyz + rotvec


def _quat_xyzw(q7):
    """EgoVerse stores head pose as [x y z | qw qx qy qz]; scipy wants xyzw.

    Verified on RLDS data: R(head)^T (p_ee_world - p_head) reproduces observation.ee_pose exactly
    only with the wxyz interpretation (ZYX yaw-pitch-roll for the rotation part).
    """
    return q7[:, [4, 5, 6, 3]]


def relative_head_motion(head_src, head_tgt):
    """6-D camera egomotion: translation expressed in the source head frame + rotation vector."""
    from scipy.spatial.transform import Rotation as R
    r_src = R.from_quat(_quat_xyzw(head_src))
    r_tgt = R.from_quat(_quat_xyzw(head_tgt))
    dp = r_src.inv().apply(head_tgt[:, :3] - head_src[:, :3])
    drot = (r_src.inv() * r_tgt).as_rotvec()
    return np.concatenate([dp, drot], axis=1)


def relative_hand_action(ee_src, ee_tgt, head_motion):
    """12-D hand action: per hand [dxyz, rotvec], both in the head frame at the source time.

    ee_src / ee_tgt: [N,12] head-frame poses (l_xyz, l_ypr, r_xyz, r_ypr) at t and t+k.
    head_motion:     [N,6] relative head motion (translation in head_t frame, rotvec) from
                     relative_head_motion(); zeros when the camera does not move (eva robot).
    The future pose is first mapped into the source head frame, then differenced; rotations are
    composed properly instead of differencing yaw/pitch/roll, which wraps at +-pi.
    Layout matches A_COLS: l_x l_y l_z l_rx l_ry l_rz r_x r_y r_z r_rx r_ry r_rz.
    """
    from scipy.spatial.transform import Rotation as R
    r_rel = R.from_rotvec(head_motion[:, 3:6])
    p_rel = head_motion[:, :3]
    out = np.zeros_like(ee_src)
    for h in (0, 6):
        p_fut = r_rel.apply(ee_tgt[:, h:h + 3]) + p_rel
        r_fut = r_rel * R.from_euler("ZYX", ee_tgt[:, h + 3:h + 6])
        r_src = R.from_euler("ZYX", ee_src[:, h + 3:h + 6])
        out[:, h:h + 3] = p_fut - ee_src[:, h:h + 3]
        out[:, h + 3:h + 6] = (r_src.inv() * r_fut).as_rotvec()
    return out


def project_keypoints(kp_world, head, frames, ext):
    """[len(frames), 2 hands, 21, 2] NORMALISED image coordinates (u/w, v/h; NaN when behind the camera).

    ext["K"] is expressed in the pixel frame of the stored image and ext["frame_wh"] gives that frame size, so
    the heat-maps downstream do not need to know the per-device resolution (Aria 640x480, mecka 640x360, ...).
    """
    from scipy.spatial.transform import Rotation as R
    r_ext, t_ext, k = np.array(ext["R_cam_from_head"]), np.array(ext["t_cam_from_head"]), np.array(ext["K"])
    wh = np.array(ext.get("frame_wh", [640, 480]), np.float64)
    out = np.full((len(frames), 2, 21, 2), np.nan, np.float32)
    for i, t in enumerate(frames):
        r_head = R.from_quat(head[t][[4, 5, 6, 3]].astype(np.float64)).inv()
        for h, side in enumerate(("left", "right")):
            p_head = r_head.apply(kp_world[side][t].astype(np.float64) - head[t, :3])
            p_cam = p_head @ r_ext.T + t_ext
            z = p_cam[:, 2]
            uv = (k @ p_cam.T).T
            ok = z > 0.05
            out[i, h, ok] = (uv[ok, :2] / uv[ok, 2:3] / wh).astype(np.float32)
    return out


def process_shard(args):
    """Parse every episode in one tfrecord shard and dump sampled frames. Runs in a worker process."""
    shard_path, subset_key, split, out_root, frame_step, frames_per_episode, seed, extr, fixed_frames = args
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    rng = random.Random(seed)
    rows = []
    max_gap = max(CHUNK_IDX)  # need frame t+30 available
    for raw in tf.data.TFRecordDataset([shard_path]):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        f = ex.features.feature

        def fl(k, shape):
            return np.asarray(f[k].float_list.value, dtype=np.float32).reshape(shape)

        def txt(k):
            return f[k].bytes_list.value[0].decode()

        n = len(f["steps/frame_index"].int64_list.value)
        if n < frame_step * 4 + max_gap:
            continue
        ep_id = txt("episode_metadata/source_episode_id")
        ep_index = int(f["episode_metadata/episode_index"].int64_list.value[0])
        emb = txt("episode_metadata/embodiment")
        task = txt("episode_metadata/task_name")
        has_head = "steps/source_float_vectors/obs_head_pose" in f
        ee = fl("steps/observation/ee_pose", (n, 12))
        chunk = fl("steps/actions_cartesian", (n, 100, 12))
        head = fl("steps/source_float_vectors/obs_head_pose", (n, 7)) if has_head else np.full((n, 7), np.nan, np.float32)
        kp_valid, kp_world = {}, {}
        for side in ("left", "right"):
            k = f"steps/source_float_vectors/{side}_obs_keypoints"
            if k in f:
                kp = fl(k, (n, -1))
                kp_valid[side] = np.isfinite(kp).all(1) & (np.abs(kp).sum(1) > 0)
                kp_world[side] = kp.reshape(n, 21, 3)
            else:
                kp_valid[side] = np.zeros(n, bool)
        imgs = f["steps/observation/images/front_1"].bytes_list.value

        # random contiguous window on the frame grid (or the exact frames of an earlier build, see --frames-from)
        if fixed_frames is not None:
            if ep_id not in fixed_frames:
                continue
            grid = np.array(sorted(fixed_frames[ep_id]))
        else:
            grid = np.arange(0, n - max_gap, frame_step)
            if len(grid) > frames_per_episode:
                start = rng.randrange(0, len(grid) - frames_per_episode + 1)
                grid = grid[start:start + frames_per_episode]
        img_dir = os.path.join(out_root, "images", subset_key, ep_id)
        os.makedirs(img_dir, exist_ok=True)
        needed = sorted(set(grid.tolist()) | {int(t + g) for t in grid for g in CHUNK_IDX})
        for t in needed:
            p = os.path.join(img_dir, f"frame_{t:06d}.jpg")
            if not os.path.exists(p):
                with open(p, "wb") as fh:
                    fh.write(imgs[t])
        # 2-D hand keypoints in the RGB image (for hand heat-maps): world -> head frame -> camera (fitted extrinsics)
        ext = extr.get(subset_key) if extr else None
        if ext is not None and has_head and len(kp_world) == 2:
            kp2d = project_keypoints(kp_world, head, needed, ext)
            kdir = os.path.join(out_root, "kp2d", subset_key)
            os.makedirs(kdir, exist_ok=True)
            np.savez_compressed(os.path.join(kdir, f"{ep_id}.npz"), frames=np.array(needed), kp2d=kp2d,
                                valid=np.stack([kp_valid["left"][needed], kp_valid["right"][needed]], 1))
        for t in needed:
            row = {
                "subset": subset_key, "split": split, "embodiment": emb, "episode_id": ep_id,
                "episode_index": ep_index, "task_name": task, "frame": int(t), "n_frames": n,
                "on_grid": bool(t in set(grid.tolist())), "has_head": has_head,
                "kp_valid_left": bool(kp_valid["left"][t]), "kp_valid_right": bool(kp_valid["right"][t]),
                "img": os.path.relpath(os.path.join(img_dir, f"frame_{t:06d}.jpg"), out_root),
            }
            for i, c in enumerate(EE_COLS):
                row[f"ee_{c}"] = float(ee[t, i])
            for g, j in CHUNK_IDX.items():
                for i, c in enumerate(EE_COLS):
                    row[f"fut{g}_{c}"] = float(chunk[t, j, i])
            for i, c in enumerate(["hx", "hy", "hz", "hqx", "hqy", "hqz", "hqw"]):
                row[f"head_{c}"] = float(head[t, i])
            rows.append(row)
    return rows


def list_shards(rlds_root, subset_dir, rlds_split):
    d = os.path.join(rlds_root, subset_dir, "ego_verse_infidata", "1.0.0")
    return sorted(glob.glob(os.path.join(d, f"*-{rlds_split}.tfrecord-*")))


def build_pairs(frames, out_root, split, gap):
    """Turn the frame table into (t, t+gap) pairs with causal labels."""
    frames = frames.sort_values(["subset", "episode_id", "frame"]).reset_index(drop=True)
    key = frames.set_index(["subset", "episode_id", "frame"])
    src = frames[frames.on_grid].copy()
    tgt_idx = pd.MultiIndex.from_arrays([src.subset, src.episode_id, src.frame + gap])
    mask = tgt_idx.isin(key.index)
    src = src[mask]
    tgt = key.loc[tgt_idx[mask]].reset_index()
    src = src.reset_index(drop=True)

    ee_src = src[[f"ee_{c}" for c in EE_COLS]].to_numpy(np.float64)
    ee_tgt = tgt[[f"ee_{c}" for c in EE_COLS]].to_numpy(np.float64)

    head_cols = [f"head_{c}" for c in ["hx", "hy", "hz", "hqx", "hqy", "hqz", "hqw"]]
    hs, ht = src[head_cols].to_numpy(np.float64), tgt[head_cols].to_numpy(np.float64)
    n_head = np.full((len(src), 6), np.nan, np.float32)
    ok = np.isfinite(hs).all(1) & np.isfinite(ht).all(1)
    if ok.any():
        n_head[ok] = relative_head_motion(hs[ok], ht[ok])
    # Hand action = future hand pose re-expressed in the head frame at t, minus current pose.
    # ee_pose[t+k] lives in the head frame at t+k; T_head[t]^-1 T_head[t+k] moves it into frame t.
    # Verified against world-frame left_obs_ee_pose on RLDS data (max error 1e-6 m / 1e-4 deg).
    # For the eva robot (fixed camera, fixed base frame, no head pose) the frames coincide.
    a_hand = relative_hand_action(ee_src, ee_tgt, np.where(ok[:, None], n_head, 0.0)).astype(np.float32)

    # human vs robot embodiment. Visually verified on samples: human_bimanual / aria_bimanual / scale are
    # head-mounted human recordings, MECKA_BIMANUAL shows gloved human arms (motion-capture rig),
    # eva_bimanual is a real bimanual robot with a fixed camera. Confirm with the data team if in doubt.
    robot_embodiments = {"eva_bimanual"}
    out = pd.DataFrame({
        "dataset": "EgoVerse", "type": np.where(src.embodiment.isin(robot_embodiments), "robot", "human"),
        "perspective": "1st",
        "src_img": src.img.values, "tgt_img": tgt.img.values,
        "subset": src.subset.values, "embodiment": src.embodiment.values, "episode_id": src.episode_id.values,
        "task_name": src.task_name.values, "src_frame": src.frame.values, "tgt_frame": tgt.frame.values,
        "kp_valid_left": src.kp_valid_left.values & tgt.kp_valid_left.values,
        "kp_valid_right": src.kp_valid_right.values & tgt.kp_valid_right.values,
    })
    for i, c in enumerate(A_COLS):
        out[f"a_{c}"] = a_hand[:, i]
    for i, c in enumerate(["tx", "ty", "tz", "rx", "ry", "rz"]):
        out[f"n_{c}"] = n_head[:, i]
    for i, c in enumerate(EE_COLS):
        out[f"s_{c}"] = ee_src[:, i]

    # LARY-compatible action .npy (12-D relative hand action)
    act_paths = []
    for r, a in zip(out.itertuples(index=False), a_hand):
        d = os.path.join(out_root, "actions", r.subset, r.episode_id)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"a_{r.src_frame:06d}_to_{r.tgt_frame:06d}.npy")
        np.save(p, a.astype(np.float32))
        act_paths.append(os.path.relpath(p, out_root))
    out["action"] = act_paths
    out.to_csv(os.path.join(out_root, f"pairs_{split}_stride{gap}.csv"), index=False)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rlds-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--subsets", default="human,aria,mecka,eva,scale")
    ap.add_argument("--splits", default="train,val,unseen")
    ap.add_argument("--shards-per-subset", default="train:120,val:40,unseen:40",
                    help="how many tfrecord shards to read per subset and split (1-2 episodes per shard)")
    ap.add_argument("--frame-step", type=int, default=15)
    ap.add_argument("--frames-per-episode", type=int, default=40)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pairs-only", action="store_true", help="rebuild pair CSVs from an existing frames_all.csv")
    ap.add_argument("--only-kp2d", action="store_true", help="re-parse the selected subsets only to (re)write kp2d files; no CSV/pair output")
    ap.add_argument("--frames-from", help="frames_all.csv of an earlier build: reuse its exact per-episode frame windows (keeps pairs identical)")
    ap.add_argument("--extrinsics", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "slotlam", "extrinsics.json"),
                    help="per-subset camera<-head extrinsics from slotlam/fit_camera_extrinsics.py (enables kp2d output)")
    a = ap.parse_args()

    if a.pairs_only:
        frames = pd.read_csv(os.path.join(a.out_root, "frames_all.csv"))
        summary = {}
        for split in a.splits.split(","):
            fs = frames[frames.split == split]
            for gap in CHUNK_IDX:
                pairs = build_pairs(fs, a.out_root, split, gap)
                summary[f"{split}_stride{gap}"] = pairs.groupby("subset").size().to_dict()
                summary[f"{split}_stride{gap}"]["episodes"] = int(pairs.episode_id.nunique())
        print(json.dumps(summary, indent=2))
        return

    shards_per = {kv.split(":")[0]: int(kv.split(":")[1]) for kv in a.shards_per_subset.split(",")}
    extr = json.load(open(a.extrinsics)) if a.extrinsics and os.path.exists(a.extrinsics) else {}
    extr = {sk: extr.get(sd) for sk, sd in SUBSETS.items() if extr.get(sd)}   # key by short subset name
    print(f"extrinsics for: {sorted(extr)}")
    fixed = None
    if a.frames_from:
        fa = pd.read_csv(a.frames_from)
        fa = fa[fa.on_grid]
        fixed = {sk: {ep: g.frame.tolist() for ep, g in fa[fa.subset == sk].groupby("episode_id")} for sk in fa.subset.unique()}
        print("reusing frame windows of", sum(len(v) for v in fixed.values()), "episodes")
    rng = random.Random(a.seed)
    os.makedirs(a.out_root, exist_ok=True)
    jobs = []
    for split in a.splits.split(","):
        for sk in a.subsets.split(","):
            shards = list_shards(a.rlds_root, SUBSETS[sk], SPLIT_TO_RLDS[split])
            if not shards:
                print(f"[warn] no shards for {sk}/{split}", file=sys.stderr)
                continue
            rng.shuffle(shards)
            for i, s in enumerate(shards[:shards_per[split]]):
                # deterministic per-job seed (Python's hash() is salted per process, which desynchronised re-runs)
                job_seed = a.seed * 100003 + zlib.crc32(f"{sk}/{split}/{i}".encode()) % 100000
                jobs.append((s, sk, split, a.out_root, a.frame_step, a.frames_per_episode, job_seed, extr, fixed.get(sk, {}) if fixed else None))
    print(f"{len(jobs)} shard jobs", flush=True)
    if a.dry_run:
        for j in jobs[:10]:
            print(j[:3])
        return

    t0 = time.time()
    all_rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(process_shard, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                all_rows.extend(fu.result())
            except Exception as e:  # keep going, report at the end
                print(f"[error] {e!r}", file=sys.stderr)
            if i % 20 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} shards, {len(all_rows)} frames, {time.time() - t0:.0f}s", flush=True)
    if a.only_kp2d:
        print("kp2d written; CSVs untouched")
        return
    frames = pd.DataFrame(all_rows)
    frames.to_csv(os.path.join(a.out_root, "frames_all.csv"), index=False)
    summary = {}
    for split in a.splits.split(","):
        fs = frames[frames.split == split]
        fs.to_csv(os.path.join(a.out_root, f"frames_{split}.csv"), index=False)
        for gap in CHUNK_IDX:
            pairs = build_pairs(fs, a.out_root, split, gap)
            summary[f"{split}_stride{gap}"] = pairs.groupby("subset").size().to_dict()
            summary[f"{split}_stride{gap}"]["episodes"] = int(pairs.episode_id.nunique())
    with open(os.path.join(a.out_root, "build_summary.json"), "w") as fh:
        json.dump({"args": vars(a), "pairs": summary, "seconds": time.time() - t0}, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
