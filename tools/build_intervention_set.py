#!/usr/bin/env python
"""Interventional test set for latent actions (do-operations on the input images).

For a sample of val pairs we build, from the *same* source frame o_t, four image pairs:

  identity   (o_t, o_t)                  zero action, zero camera motion
  camera     (o_t, warp(o_t))            pure synthetic camera egomotion (similarity transform), hands do not move
  photo      (o_t, photometric(o_t))     pure appearance change (brightness / contrast / gamma), nothing moves
  real       (o_t, o_{t+k})              the recorded pair (hand action + real camera motion)

causal/intervention.py then compares how far z moves under each intervention relative to `real`.
A latent action that is causally about the hands should barely move under `camera` and `photo`.

Output: $DATA_DIR/regression/egoverse_ic/{images,egoverse_ic_metadata_val.csv}
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

KINDS = ["identity", "camera", "photo", "real"]


def warp_camera(img, rng, max_shift=0.06, max_rot=5.0, max_scale=0.04):
    h, w = img.shape[:2]
    ang = rng.uniform(-max_rot, max_rot)
    sc = 1 + rng.uniform(-max_scale, max_scale)
    tx, ty = rng.uniform(-max_shift, max_shift, 2) * np.array([w, h])
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    m[:, 2] += (tx, ty)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101), (tx / w, ty / h, ang, sc)


def photometric(img, rng):
    gamma = rng.uniform(0.7, 1.4)
    gain = rng.uniform(0.8, 1.2)
    bias = rng.uniform(-20, 20)
    x = (img.astype(np.float32) / 255.0) ** gamma * 255.0 * gain + bias
    return np.clip(x, 0, 255).astype(np.uint8), (gamma, gain, bias)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", required=True, help="pairs_val_stride15.csv")
    ap.add_argument("--pairs-root", required=True, help="$DATA_DIR/regression/egoverse")
    ap.add_argument("--out-root", required=True, help="$DATA_DIR/regression/egoverse_ic")
    ap.add_argument("--per-subset", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    df = pd.read_csv(a.pairs_csv)
    # only pairs with a valid, tracked action label so `real` is meaningful
    if {"kp_valid_left", "kp_valid_right"} <= set(df.columns):
        df = df[(df.subset == "eva") | (df.kp_valid_left & df.kp_valid_right)]
    sample = pd.concat([g.sample(min(a.per_subset, len(g)), random_state=a.seed) for _, g in df.groupby("subset")])
    rows = []
    for pid, r in enumerate(sample.itertuples(index=False)):
        src = cv2.imread(os.path.join(a.pairs_root, r.src_img))
        d = os.path.join(a.out_root, "images", r.subset, f"{pid:05d}")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "src.jpg"), src, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cam, cam_p = warp_camera(src, rng)
        cv2.imwrite(os.path.join(d, "camera.jpg"), cam, [cv2.IMWRITE_JPEG_QUALITY, 95])
        pho, pho_p = photometric(src, rng)
        cv2.imwrite(os.path.join(d, "photo.jpg"), pho, [cv2.IMWRITE_JPEG_QUALITY, 95])
        tgt = cv2.imread(os.path.join(a.pairs_root, r.tgt_img))
        cv2.imwrite(os.path.join(d, "real.jpg"), tgt, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rel = lambda name: os.path.relpath(os.path.join(d, name), a.out_root)
        base = {"pair_id": pid, "subset": r.subset, "embodiment": r.embodiment, "episode_id": r.episode_id,
                "src_frame": r.src_frame, "tgt_frame": r.tgt_frame, "dataset": "EgoVerse", "type": r.type, "perspective": "1st"}
        rows.append({**base, "kind": "identity", "src_img": rel("src.jpg"), "tgt_img": rel("src.jpg")})
        rows.append({**base, "kind": "camera", "src_img": rel("src.jpg"), "tgt_img": rel("camera.jpg"),
                     "cam_tx": cam_p[0], "cam_ty": cam_p[1], "cam_rot_deg": cam_p[2], "cam_scale": cam_p[3]})
        rows.append({**base, "kind": "photo", "src_img": rel("src.jpg"), "tgt_img": rel("photo.jpg"),
                     "photo_gamma": pho_p[0], "photo_gain": pho_p[1], "photo_bias": pho_p[2]})
        rows.append({**base, "kind": "real", "src_img": rel("src.jpg"), "tgt_img": rel("real.jpg"),
                     **{c: getattr(r, c) for c in df.columns if c.startswith(("a_", "n_", "s_"))}})
        if pid % 200 == 0:
            print(pid, flush=True)
    out = pd.DataFrame(rows)
    out["action"] = ""  # unused by extraction; keeps the LARY image-pair schema
    csv = os.path.join(a.out_root, "egoverse_ic_metadata_val.csv")
    out.to_csv(csv, index=False)
    print("wrote", csv, len(out), "rows", out.kind.value_counts().to_dict())


if __name__ == "__main__":
    main()
