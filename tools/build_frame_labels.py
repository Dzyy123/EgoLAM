#!/usr/bin/env python
"""SlotLAM labels for pre-extracted EgoVerse frames (egoverse_only_dt0p4).

The frames live in FRAMES_ROOT/frames/<builder>/<split>/epXXXXXX/{000000.jpg, ..., meta.json}: one JPEG every
12 RLDS steps (0.4 s at 30 fps), stored frame k == RLDS step 12k. The episode numbers are his read order, not the
RLDS `episode_index` field, so every RLDS record is matched to its folder by the MD5 of its first frame.

  1. index     hash 000000.jpg / 000001.jpg and read num_stored_frames of every episode -> OUT/first_frame_index.json
  2. labels    parse RLDS shards in parallel; per matched episode write OUT/<episode path>.npz with, on the stored
               grid (K frames):
                 ee [K,12] ee pose in the head frame (xyz + ZYX ypr), head [K,7] world head pose (xyz, qw qx qy qz;
                 NaN if absent), kp2d [K,2,21,2] float16 normalised image coords (NaN = invalid), kp_valid [K,2],
                 and for every requested grid gap g:  a{g} [K-g,12] hand action, n{g} [K-g,6] head motion,
                 ok{g} [K-g] physically plausible pair (limits scaled from the LARY-Causal 0.5 s limits).
  3. manifest  OUT/manifest.jsonl (one row per labelled episode), OUT/pairs_gap{g}_{split}.npz (episode id, k of
               every ok pair) and OUT/stats.json (mean/std of a and n over train pairs) for slotlam/train_frames.py.

Run with the system python that has TensorFlow (same as causal/build_egoverse_pairs.py), e.g.
  python slotlam/build_frame_labels.py --stage all --workers 20
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np

from build_egoverse_pairs import SUBSETS, project_keypoints, relative_hand_action, relative_head_motion  # noqa: E402

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
BUILDER = "EgoVerse_full__{}__ego_verse_infidata__1.0.0"
SUBDIR_TO_KEY = {v: k for k, v in SUBSETS.items()}
STRIDE = 12              # RLDS steps between stored frames
GRID_DT = 0.4            # seconds between stored frames
LIMITS_05S = {"state_xyz_m": 2.0, "hand_dxyz_m": 0.4, "hand_drot_rad": np.pi / 2, "head_dxyz_m": 0.5, "head_drot_rad": np.pi / 2}


def md5(b):
    return hashlib.md5(b).hexdigest()


# ----------------------------------------------------------------------------- stage 1
def build_index(frames_root, out):
    """{md5(000000.jpg): [[episode path, md5(000001.jpg) or "", num_stored_frames], ...]}.

    First frames are not unique (static openings, repeated recordings), so a record is matched on
    (first-frame hash, second-frame hash, number of stored frames)."""
    rows = [l.split("\t")[0] for l in open(os.path.join(frames_root, "episode_list.txt")) if l.strip()]
    rows = [p for p in rows if p.split("/")[1].startswith("EgoVerse_full__")]

    def one(p):
        d = os.path.join(frames_root, p)
        try:
            with open(os.path.join(d, "000000.jpg"), "rb") as fh:
                h0 = md5(fh.read())
            try:
                with open(os.path.join(d, "000001.jpg"), "rb") as fh:
                    h1 = md5(fh.read())
            except OSError:
                h1 = ""
            k = json.load(open(os.path.join(d, "meta.json")))["num_stored_frames"]
            return h0, [p, h1, int(k)]
        except (OSError, KeyError, ValueError):
            return None, p

    index, missing, t0 = {}, 0, time.time()
    with ThreadPoolExecutor(48) as ex:
        for i, (h, v) in enumerate(ex.map(one, rows, chunksize=64)):
            if h is None:
                missing += 1
            else:
                index.setdefault(h, []).append(v)
            if i % 10000 == 0:
                print(f"  index {i}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    keys = {(v[0].split("/")[0], h, v[1], v[2]) for h, vs in index.items() for v in vs}
    json.dump(index, open(out, "w"))
    print(f"index: {sum(len(v) for v in index.values())} episodes, {len(index)} distinct first frames, "
          f"{len(keys)} distinct (frame0, frame1, K) keys, {missing} unreadable -> {out}", flush=True)


def match(index, imgs, builder, split):
    """episode folders of an RLDS record: same first/second frame and length, same builder and split.

    EgoVerse's seen_test is almost entirely a copy of train episodes (identical frames), so a record is only
    assigned to folders of its own split; identical copies inside one split all get the labels."""
    n = len(imgs)
    if not n:
        return []
    k = (n + STRIDE - 1) // STRIDE
    h1 = md5(imgs[STRIDE]) if n > STRIDE else ""
    out = []
    for p, c1, ck in index.get(md5(imgs[0]), []):
        parts = p.split("/")
        if ck == k and c1 == h1 and parts[1] == builder and parts[2] == split:
            out.append(p)
    return out


# ----------------------------------------------------------------------------- stage 2
def pair_labels(ee, head, has_head, gap):
    """a, n, ok for grid pairs (k, k+gap); ee/head are on the stored grid."""
    k = len(ee) - gap
    if k <= 0:
        return np.zeros((0, 12), np.float32), np.zeros((0, 6), np.float32), np.zeros(0, bool)
    if has_head:
        n = relative_head_motion(head[:k].astype(np.float64), head[gap:].astype(np.float64))
    else:                        # fixed camera (eva robot): zero ego-motion is the true label
        n = np.zeros((k, 6))
    a = relative_hand_action(ee[:k].astype(np.float64), ee[gap:].astype(np.float64), n)
    sc = gap * GRID_DT / 0.5
    pos = [0, 1, 2, 6, 7, 8]
    ok = np.isfinite(a).all(1) & np.isfinite(ee[:k]).all(1) & (np.abs(ee[:k][:, pos]) <= LIMITS_05S["state_xyz_m"]).all(1)
    ok &= (np.abs(a[:, pos]) <= LIMITS_05S["hand_dxyz_m"] * sc).all(1)
    lim_rot = min(np.pi, LIMITS_05S["hand_drot_rad"] * sc)
    ok &= (np.linalg.norm(a[:, 3:6], axis=1) <= lim_rot) & (np.linalg.norm(a[:, 9:12], axis=1) <= lim_rot)
    if has_head:
        ok &= np.isfinite(n).all(1) & (np.linalg.norm(n[:, :3], axis=1) <= LIMITS_05S["head_dxyz_m"] * sc)
        ok &= np.linalg.norm(n[:, 3:], axis=1) <= min(np.pi, LIMITS_05S["head_drot_rad"] * sc)
    return a.astype(np.float32), n.astype(np.float32), ok


def process_shard(args):
    shard, subdir, split, index_path, out_root, gaps, extr_path = args
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    index = json.load(open(index_path))
    extr = json.load(open(extr_path)) if extr_path and os.path.exists(extr_path) else {}
    ext = extr.get(subdir) or extr.get(SUBDIR_TO_KEY.get(subdir))   # extrinsics.json is keyed by the RLDS subset folder name
    rows, unmatched = [], 0
    for raw in tf.data.TFRecordDataset([shard]):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        f = ex.features.feature
        imgs = f["steps/observation/images/front_1"].bytes_list.value
        n = len(imgs)
        paths = match(index, imgs, BUILDER.format(subdir), split)
        if not paths:
            unmatched += 1
            continue
        grid = np.arange(0, n, STRIDE)
        fl = lambda k, shape: np.asarray(f[k].float_list.value, dtype=np.float32).reshape(shape)  # noqa: E731
        ee = fl("steps/observation/ee_pose", (n, 12))
        has_head = "steps/source_float_vectors/obs_head_pose" in f
        head = fl("steps/source_float_vectors/obs_head_pose", (n, 7)) if has_head else np.full((n, 7), np.nan, np.float32)
        kp_world, kp_valid = {}, np.zeros((n, 2), bool)
        for h, side in enumerate(("left", "right")):
            key = f"steps/source_float_vectors/{side}_obs_keypoints"
            if key in f:
                kp = fl(key, (n, -1))
                kp_valid[:, h] = np.isfinite(kp).all(1) & (np.abs(kp).sum(1) > 0)
                kp_world[side] = kp.reshape(n, 21, 3)
        if ext is not None and has_head and len(kp_world) == 2:
            kp2d = project_keypoints(kp_world, head, grid.tolist(), ext).astype(np.float16)
        else:
            kp2d = np.full((len(grid), 2, 21, 2), np.nan, np.float16)
        out = {"ee": ee[grid], "head": head[grid], "kp2d": kp2d, "kp_valid": kp_valid[grid], "n_steps": np.int64(n),
               "rlds_episode_index": np.int64(f["episode_metadata/episode_index"].int64_list.value[0])}
        for g in gaps:
            out[f"a{g}"], out[f"n{g}"], out[f"ok{g}"] = pair_labels(out["ee"], out["head"], has_head, g)
        for p in paths:
            dst = os.path.join(out_root, p + ".npz")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.savez_compressed(dst, **out)
            rows.append({"path": p, "subset": SUBDIR_TO_KEY.get(subdir, subdir), "split": split, "K": int(len(grid)),
                         "has_head": bool(has_head), "kp_frac": float(out["kp_valid"].mean()), "copies": len(paths),
                         **{f"n_ok{g}": int(out[f"ok{g}"].sum()) for g in gaps}})
    return rows, unmatched


# ----------------------------------------------------------------------------- stage 3
def write_manifest(out_root, rows, gaps):
    """Consolidate per-episode npz into memory-mappable arrays under OUT/consolidated/ for train_frames.py."""
    c = os.path.join(out_root, "consolidated")
    os.makedirs(c, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["path"])
    with open(os.path.join(c, "episodes.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    K = np.array([r["K"] for r in rows], np.int64)
    off = np.concatenate([[0], np.cumsum(K)[:-1]]).astype(np.int64)
    F = int(K.sum())
    np.save(os.path.join(c, "frame_offset.npy"), off)
    np.save(os.path.join(c, "ep_of_frame.npy"), np.repeat(np.arange(len(rows), dtype=np.int32), K))
    mk = lambda name, shape, dt, fill: _filled(np.lib.format.open_memmap(os.path.join(c, name), "w+", dt, shape), fill)  # noqa: E731
    kp2d = mk("kp2d.npy", (F, 2, 21, 2), np.float16, np.nan)
    kpv = mk("kp_valid.npy", (F, 2), np.bool_, False)
    A = {g: mk(f"a{g}.npy", (F, 12), np.float32, np.nan) for g in gaps}
    N = {g: mk(f"n{g}.npy", (F, 6), np.float32, np.nan) for g in gaps}
    pairs = {(g, s): [] for g in gaps for s in {r["split"] for r in rows}}
    t0 = time.time()
    for i, r in enumerate(rows):
        d = np.load(os.path.join(out_root, r["path"] + ".npz"))
        o, k = off[i], r["K"]
        kp2d[o:o + k], kpv[o:o + k] = d["kp2d"], d["kp_valid"]
        for g in gaps:
            m = len(d[f"ok{g}"])
            A[g][o:o + m], N[g][o:o + m] = d[f"a{g}"], d[f"n{g}"]
            pairs[(g, r["split"])].append(o + np.flatnonzero(d[f"ok{g}"]))
        if i % 5000 == 0:
            print(f"  consolidate {i}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    for arr in [kp2d, kpv, *A.values(), *N.values()]:
        arr.flush()
    stats = {"gaps": gaps, "grid_dt": GRID_DT, "frames": F, "episodes": len(rows)}
    for (g, split), lst in pairs.items():
        idx = np.concatenate(lst).astype(np.int64) if lst else np.zeros(0, np.int64)
        np.save(os.path.join(c, f"pairs_gap{g}_{split}.npy"), idx)
        stats[f"pairs_gap{g}_{split}"] = int(len(idx))
        print(f"  pairs gap{g} {split}: {len(idx)}", flush=True)
    rng = np.random.default_rng(0)
    for g in gaps:
        tr = pairs.get((g, "train"), [])
        idx = np.concatenate(tr) if tr else np.zeros(0, np.int64)
        idx = np.sort(rng.choice(idx, size=min(500000, len(idx)), replace=False)) if len(idx) else idx
        a_s, n_s = np.asarray(A[g][idx]), np.asarray(N[g][idx])
        n_s = n_s[np.isfinite(n_s).all(1)]
        head_rows = np.array([rows[e]["has_head"] for e in np.searchsorted(off, idx, side="right") - 1]) if len(idx) else np.zeros(0, bool)
        n_s = np.asarray(N[g][idx[head_rows]]) if len(idx) else n_s
        stats[f"gap{g}"] = {"a_mean": a_s.mean(0).tolist(), "a_std": (a_s.std(0) + 1e-6).tolist(),
                            "n_mean": n_s.mean(0).tolist(), "n_std": (n_s.std(0) + 1e-6).tolist(), "n_sample_pairs": int(len(a_s))}
    json.dump(stats, open(os.path.join(c, "stats.json"), "w"), indent=1)
    print(f"consolidated: {len(rows)} episodes, {F} frames -> {c}", flush=True)


def _filled(arr, fill):
    arr[:] = fill
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--rlds-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--extrinsics", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "extrinsics.json"))
    ap.add_argument("--stage", default="all", choices=["index", "labels", "manifest", "all"])
    ap.add_argument("--subsets", default=",".join(SUBSETS.values()))
    ap.add_argument("--splits", default="train,seen_test,unseen_test")
    ap.add_argument("--gaps", default="3", help="grid gaps to label; 3 x 0.4 s = 1.2 s (stored frame interval)")
    ap.add_argument("--max-shards", type=int, default=0, help="per subset and split, 0 = all (smoke tests use a few)")
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    gaps = [int(g) for g in a.gaps.split(",")]
    os.makedirs(a.out_root, exist_ok=True)
    index_path = os.path.join(a.out_root, "first_frame_index.json")
    if a.stage in ("index", "all") and not os.path.exists(index_path):
        build_index(a.frames_root, index_path)
    rows_path = os.path.join(a.out_root, "episodes_done.jsonl")
    if a.stage in ("labels", "all"):
        jobs = []
        for subdir in a.subsets.split(","):
            for split in a.splits.split(","):
                shards = sorted(glob.glob(os.path.join(a.rlds_root, subdir, "ego_verse_infidata", "1.0.0", f"*-{split}.tfrecord-*")))
                if a.max_shards:
                    shards = shards[:a.max_shards]
                jobs += [(s, subdir, split, index_path, a.out_root, gaps, a.extrinsics) for s in shards]
        print(f"labels: {len(jobs)} shards, {a.workers} workers", flush=True)
        t0, done, n_ep, n_un = time.time(), 0, 0, 0
        with ProcessPoolExecutor(a.workers) as ex, open(rows_path, "a") as fh:
            for fut in as_completed([ex.submit(process_shard, j) for j in jobs]):
                rows, un = fut.result()
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                done, n_ep, n_un = done + 1, n_ep + len(rows), n_un + un
                if done % 25 == 0 or done == len(jobs):
                    el = time.time() - t0
                    print(f"  shards {done}/{len(jobs)}  episodes {n_ep}  unmatched {n_un}  {el / 60:.1f} min, ETA {el / done * (len(jobs) - done) / 60:.0f} min", flush=True)
    if a.stage in ("manifest", "all"):
        rows = {}
        for l in open(rows_path):
            r = json.loads(l)
            rows[r["path"]] = r          # re-runs overwrite
        write_manifest(a.out_root, list(rows.values()), gaps)


if __name__ == "__main__":
    main()
