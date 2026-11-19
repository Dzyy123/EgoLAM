#!/usr/bin/env python
"""Interventional Consistency (IC) from the egoverse_ic extraction.

For every source frame we have z for four pairs built from the same o_t (see build_intervention_set.py):
identity, camera (synthetic egomotion only), photo (appearance only), real (recorded pair).

  d_kind = || z_kind - z_identity ||   (per sample, after per-dimension standardisation of z on `real`)

Reported per model:
  IC_camera = median(d_camera) / median(d_real)     lower is better: pure camera motion should not look like an action
  IC_photo  = median(d_photo)  / median(d_real)     lower is better
  P(camera > real) = fraction of samples whose synthetic camera move shifts z more than the real action did

Also: R2 of a ridge probe from z_camera - z_identity to the synthetic camera parameters (tx, ty, rot, scale),
an interventional analogue of NL; with the diagnostic reported per intervention.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import RidgeProbe, load_latents, r2_score  # noqa: E402

CAM_COLS = ["cam_tx", "cam_ty", "cam_rot_deg", "cam_scale"]


def analyse(df, z):
    df = df.reset_index(drop=True)
    byk = {k: df.index[df.kind == k].to_numpy() for k in ["identity", "camera", "photo", "real"]}
    # align by pair_id
    order = {k: df.loc[byk[k]].sort_values("pair_id").index.to_numpy() for k in byk}
    zi, zc, zp, zr = (z[order[k]] for k in ["identity", "camera", "photo", "real"])
    sd = zr.std(0) + 1e-6
    d = {k: np.linalg.norm((zz - zi) / sd, axis=1) for k, zz in [("camera", zc), ("photo", zp), ("real", zr)]}
    res = {"n": int(len(zi)),
           "IC_camera": float(np.median(d["camera"]) / np.median(d["real"])),
           "IC_photo": float(np.median(d["photo"]) / np.median(d["real"])),
           "P_camera_gt_real": float(np.mean(d["camera"] > d["real"])),
           "P_photo_gt_real": float(np.mean(d["photo"] > d["real"])),
           "median_d": {k: float(np.median(v)) for k, v in d.items()}}
    sub = df.loc[order["real"], "subset"].to_numpy()
    res["IC_camera_by_subset"] = {s: float(np.median(d["camera"][sub == s]) / np.median(d["real"][sub == s])) for s in np.unique(sub)}
    # interventional nuisance probe: can the synthetic camera parameters be read from z_camera - z_identity?
    cam = df.loc[order["camera"], CAM_COLS].to_numpy(np.float64)
    groups = df.loc[order["camera"], "episode_id"].to_numpy()
    x = (zc - zi) / sd
    rng = np.random.default_rng(0)
    eps = np.unique(groups)
    hold = set(rng.choice(eps, size=max(1, len(eps) // 4), replace=False).tolist())
    m = np.array([g in hold for g in groups])
    p = RidgeProbe().fit(x[~m], cam[~m], groups[~m])
    res["R2_cam_params_from_dz"] = r2_score(cam[m], p.predict(x[m]))
    return res


