#!/usr/bin/env python
"""LARY-Causal probes.

Given frozen latent actions z for EgoVerse image pairs (extracted with `lary.cli extract`),
fit light probes and report how much of each causal variable z carries:

  AD   action decodability        R2(a_hand | z)          higher is better
  NL   nuisance leakage           R2(n_head | z)          lower is better (given AD)
  NLp  partial nuisance leakage   R2(n_head - E[n_head|a_hand] | z)   lower is better
  SL   state leakage              R2(s_t | z)             lower is better
  CET  cross-embodiment transfer  probe fit on human pairs, tested on robot pairs (and reverse)
  COMP compositionality           a(t->t+30) vs a(t->t+15) + a(t+15->t+30) decoded from z

Reference rows: R2(a | s) = "habit" baseline (how predictable the action is from state alone).

All probes are ridge regressions on standardized inputs/targets with the ridge strength chosen on
held-out training episodes, so numbers are deterministic and cheap. `--probe mlp` switches to a
2-layer MLP for a non-linear reading.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

A_COLS = [f"a_{h}_{c}" for h in ("l", "r") for c in ("x", "y", "z", "rx", "ry", "rz")]
S_COLS = [f"s_{h}_{c}" for h in ("l", "r") for c in ("x", "y", "z", "yaw", "pitch", "roll")]
N_COLS = [f"n_{c}" for c in ("tx", "ty", "tz", "rx", "ry", "rz")]
GROUPS = {
    "a": {"pos": [0, 1, 2, 6, 7, 8], "rot": [3, 4, 5, 9, 10, 11]},
    "n": {"trans": [0, 1, 2], "rot": [3, 4, 5]},
    "s": {"pos": [0, 1, 2, 6, 7, 8], "rot": [3, 4, 5, 9, 10, 11]},
}


# ----------------------------------------------------------------------------- data
# Physical plausibility limits for one 0.5 s / 1 s pair. Rows outside are label glitches
# (missing-hand sentinel poses of hundreds of metres, tracking jumps), not behaviour we want to score.
LIMITS = {"state_xyz_m": 2.0, "hand_dxyz_m": 0.4, "hand_drot_rad": np.pi / 2,
          "head_dxyz_m": 0.5, "head_drot_rad": np.pi / 2}   # per 0.5 s; motion limits scale with the frame gap
KEY_COLS = ["subset", "episode_id", "src_frame", "tgt_frame"]


def refresh_labels(df, pairs_csv):
    """Take la_path from an extraction CSV but every label column from the current pairs CSV."""
    fresh = pd.read_csv(pairs_csv)
    keep = df[KEY_COLS + ["la_path"]]
    merged = fresh.merge(keep, on=KEY_COLS, how="inner", validate="one_to_one")
    if len(merged) != len(df):
        raise ValueError(f"label refresh mismatch: {len(merged)} vs {len(df)} rows ({pairs_csv})")
    return merged


def clean_pairs(df, log=print, stride=15):
    """Drop physically implausible rows; keep NaN head motion (robot with fixed camera)."""
    scale = stride / 15.0
    lim_dxyz, lim_drot = LIMITS["hand_dxyz_m"] * scale, min(np.pi, LIMITS["hand_drot_rad"] * scale)
    lim_hxyz, lim_hrot = LIMITS["head_dxyz_m"] * scale, min(np.pi, LIMITS["head_drot_rad"] * scale)
    a = df[A_COLS].to_numpy(np.float64)
    s = df[S_COLS].to_numpy(np.float64)
    n = df[N_COLS].to_numpy(np.float64)
    pos = [0, 1, 2, 6, 7, 8]
    rot = [3, 4, 5, 9, 10, 11]
    ok = np.isfinite(a).all(1) & np.isfinite(s).all(1)
    ok &= (np.abs(s[:, pos]) <= LIMITS["state_xyz_m"]).all(1)
    ok &= (np.abs(a[:, pos]) <= lim_dxyz).all(1)
    ok &= (np.linalg.norm(a[:, 3:6], axis=1) <= lim_drot) & (np.linalg.norm(a[:, 9:12], axis=1) <= lim_drot)
    has_n = np.isfinite(n).all(1)
    ok &= ~has_n | ((np.linalg.norm(n[:, :3], axis=1) <= lim_hxyz) & (np.linalg.norm(n[:, 3:], axis=1) <= lim_hrot))
    # hand-tracking validity where the recording device provides keypoints (not on the eva robot)
    if {"kp_valid_left", "kp_valid_right"} <= set(df.columns):
        tracked = df.subset.ne("eva").to_numpy()
        ok &= ~tracked | (df.kp_valid_left.to_numpy(bool) & df.kp_valid_right.to_numpy(bool))
    kept = df[ok].reset_index(drop=True)
    frac = pd.Series(ok).groupby(df.subset.values).mean().round(3).to_dict()
    log(f"  clean: kept {ok.sum()}/{len(df)} rows; per subset {frac}")
    return kept


def pool_tokens(tok, grid=2, max_tokens=64):
    """Patch-token features (F, N, D) with many tokens -> per-frame `grid x grid` average pooling.

    LAM outputs such as LAPA (4, 1024) or LAQ (16, 32) are left untouched; raw encoder features
    such as DINOv3-origin (2, 196, 1024) become (2, grid*grid, 1024) so that ridge probes stay tractable
    while keeping coarse spatial layout.
    """
    tok = np.asarray(tok, dtype=np.float32)
    if tok.ndim == 3 and tok.shape[1] > max_tokens:
        f, n, d = tok.shape
        s = int(round(np.sqrt(n)))
        if s * s == n and s % grid == 0:
            tok = tok.reshape(f, grid, s // grid, grid, s // grid, d).mean(axis=(2, 4)).reshape(f, grid * grid, d)
        else:
            tok = tok.mean(axis=1, keepdims=True)
    return tok


def load_latents(df, la_root, pca=None, max_dim=4096, log=print):
    """Load and flatten `tokens` from every npz referenced in df.la_path -> (N, D) float32."""
    zs = []
    t0 = time.time()
    for i, p in enumerate(df["la_path"].tolist()):
        full = p if os.path.isabs(p) else os.path.join(la_root, p)
        with np.load(full) as d:
            tok = d["tokens"] if "tokens" in d else d[d.files[0]]
        zs.append(pool_tokens(tok).reshape(-1))
        if i and i % 5000 == 0:
            log(f"    loaded {i}/{len(df)} latents ({time.time() - t0:.0f}s)")
    z = np.stack(zs)
    log(f"  latents: {z.shape}, {time.time() - t0:.0f}s")
    return z


def fit_pca(z_train, k):
    """PCA via eigendecomposition of the feature covariance (GPU if torch+CUDA are available)."""
    mu = z_train.mean(0)
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        z = torch.tensor(z_train, device=dev)
        z -= torch.tensor(mu, device=dev)
        cov = torch.zeros(z.shape[1], z.shape[1], device=dev, dtype=torch.float64)
        for i in range(0, len(z), 8192):
            blk = z[i:i + 8192].double()
            cov += blk.T @ blk
        evals, evecs = torch.linalg.eigh(cov)
        comps = evecs[:, -k:].flip(1).T.float().cpu().numpy()
        del z, cov, evals, evecs
        torch.cuda.empty_cache()
    except Exception as e:  # numpy fallback
        print(f"  torch PCA unavailable ({e!r}); numpy fallback")
        zc = z_train - mu
        cov = zc.T @ zc
        evals, evecs = np.linalg.eigh(cov)
        comps = evecs[:, -k:][:, ::-1].T.astype(np.float32)
    return mu, comps


def apply_pca(z, mu, comps):
    out = np.empty((len(z), comps.shape[0]), np.float32)
    for i in range(0, len(z), 8192):
        out[i:i + 8192] = (z[i:i + 8192] - mu) @ comps.T
    return out


# ----------------------------------------------------------------------------- probes
class RidgeProbe:
    """Closed-form ridge with per-feature standardization; alpha picked on held-out episodes."""

    def __init__(self, alphas=(1e-2, 1e-1, 1, 10, 100, 1000)):
        self.alphas = alphas

    def _prep(self, x):
        return (x - self.x_mu) / self.x_sd

    def fit(self, x, y, groups):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        self.x_mu, self.x_sd = x.mean(0), x.std(0) + 1e-6
        self.y_mu, self.y_sd = y.mean(0), y.std(0) + 1e-8
        xs, ys = self._prep(x), (y - self.y_mu) / self.y_sd
        # held-out episodes for alpha selection
        uniq = np.unique(groups)
        rng = np.random.default_rng(0)
        hold = set(rng.choice(uniq, size=max(1, len(uniq) // 5), replace=False).tolist())
        m = np.array([g in hold for g in groups])
        best, best_alpha = -np.inf, None
        for a in self.alphas:
            r2 = r2_score(ys[m], xs[m] @ self._solve(xs[~m], ys[~m], a))
            if r2 > best:
                best, best_alpha = r2, a
        self.alpha = best_alpha
        self.w = self._solve(xs, ys, best_alpha)
        return self

    @staticmethod
    def _solve(x, y, alpha):
        """Ridge weights; dual form when N < D so tiny sets stay cheap."""
        n, d = x.shape
        if n < d:
            k = x @ x.T
            k[np.diag_indices(n)] += alpha
            return x.T @ np.linalg.solve(k, y)
        g = x.T @ x
        g[np.diag_indices(d)] += alpha
        return np.linalg.solve(g, x.T @ y)

    def predict(self, x):
        return (self._prep(np.asarray(x, np.float64)) @ self.w) * self.y_sd + self.y_mu


class MLPProbe:
    """2-layer MLP probe (torch), early-stopped on held-out episodes."""

    def __init__(self, hidden=512, epochs=30, lr=1e-3, device=None):
        self.hidden, self.epochs, self.lr = hidden, epochs, lr
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, x, y, groups):
        torch, dev = self.torch, self.device
        x = np.asarray(x, np.float32)
        y = np.asarray(y, np.float32)
        self.x_mu, self.x_sd = x.mean(0), x.std(0) + 1e-6
        self.y_mu, self.y_sd = y.mean(0), y.std(0) + 1e-8
        xs = torch.tensor((x - self.x_mu) / self.x_sd, device=dev)
        ys = torch.tensor((y - self.y_mu) / self.y_sd, device=dev)
        uniq = np.unique(groups)
        rng = np.random.default_rng(0)
        hold = set(rng.choice(uniq, size=max(1, len(uniq) // 5), replace=False).tolist())
        m = torch.tensor(np.array([g in hold for g in groups]), device=dev)
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], self.hidden), torch.nn.GELU(),
                                  torch.nn.Linear(self.hidden, self.hidden), torch.nn.GELU(),
                                  torch.nn.Linear(self.hidden, y.shape[1])).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        best, best_state = -np.inf, None
        xtr, ytr = xs[~m], ys[~m]
        bs = 512
        for ep in range(self.epochs):
            net.train()
            perm = torch.randperm(len(xtr), device=dev)
            for i in range(0, len(xtr), bs):
                idx = perm[i:i + bs]
                loss = torch.nn.functional.mse_loss(net(xtr[idx]), ytr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                r2 = r2_score(ys[m].cpu().numpy(), net(xs[m]).cpu().numpy())
            if r2 > best:
                best, best_state = r2, {k: v.clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        self.net = net.eval()
        return self

    def predict(self, x):
        torch = self.torch
        with torch.no_grad():
            xs = torch.tensor(((np.asarray(x, np.float32) - self.x_mu) / self.x_sd), device=self.device)
            out = np.concatenate([self.net(xs[i:i + 4096]).cpu().numpy() for i in range(0, len(xs), 4096)])
        return out * self.y_sd + self.y_mu


def make_probe(kind):
    return MLPProbe() if kind == "mlp" else RidgeProbe()


def r2_score(y, yhat):
    """Mean over dims of 1 - MSE/Var (variance of the evaluation set itself)."""
    y, yhat = np.asarray(y, np.float64), np.asarray(yhat, np.float64)
    var = y.var(0) + 1e-12
    return float(np.mean(1 - ((y - yhat) ** 2).mean(0) / var))


def r2_by_group(y, yhat, groups):
    out = {"all": r2_score(y, yhat)}
    for g, idx in groups.items():
        out[g] = r2_score(y[:, idx], yhat[:, idx])
    return out


# ----------------------------------------------------------------------------- metric blocks
def probe_target(kind, z_tr, y_tr, g_tr, evals, groups):
    """Fit one probe and evaluate on several (name -> (z, y)) sets."""
    p = make_probe(kind).fit(z_tr, y_tr, g_tr)
    res = {}
    for name, (z_ev, y_ev) in evals.items():
        if len(y_ev) == 0:
            continue
        res[name] = r2_by_group(y_ev, p.predict(z_ev), groups)
        res[name]["n"] = int(len(y_ev))
    if isinstance(p, RidgeProbe):
        res["alpha"] = p.alpha
    return res, p


def affine_calibrate(pred_fit, y_fit, pred_eval):
    """Per-dimension slope/intercept from (pred_fit -> y_fit), applied to pred_eval."""
    out = np.empty_like(pred_eval)
    for j in range(pred_fit.shape[1]):
        slope, icpt = np.polyfit(pred_fit[:, j], y_fit[:, j], 1)
        out[:, j] = slope * pred_eval[:, j] + icpt
    return out


def residualize(y, x, mask_fit):
    """y - E[y|x] with a linear fit on rows mask_fit (used for partial nuisance leakage)."""
    xa = np.concatenate([x, np.ones((len(x), 1))], 1)
    w, *_ = np.linalg.lstsq(xa[mask_fit], y[mask_fit], rcond=None)
    return y - xa @ w


def run(args, log=print):
    la_root = args.la_root or os.environ.get("LARY_LA_DIR")

    def tgt(df, cols):
        return df[cols].to_numpy(np.float32)

    def grp(df):
        return (df.subset + "/" + df.episode_id).to_numpy()

    dfs = {s: pd.read_csv(p) for s, p in [("train", args.train_csv), ("val", args.val_csv), ("unseen", args.unseen_csv)] if p}
    for s in list(dfs):
        d = dfs[s].dropna(subset=["la_path"])
        if args.pairs_dir:
            d = refresh_labels(d, os.path.join(args.pairs_dir, f"pairs_{s}_stride{args.stride}.csv"))
        log(f"{s}: {len(d)} pairs, subsets={d.subset.value_counts().to_dict()}")
        dfs[s] = clean_pairs(d, log=log, stride=args.stride)
    if args.oracle:
        # Positive/negative controls: replace z by label-derived features to check that each metric
        # reads what it claims (e.g. z=action -> AD~1, NLp~0; z=head -> NL~1; z=random -> all ~0).
        rng = np.random.default_rng(0)

        def oracle(d):
            parts = []
            for name in args.oracle.split("+"):
                if name == "action":
                    parts.append(tgt(d, A_COLS))
                elif name == "head":
                    n = tgt(d, N_COLS)
                    parts.append(np.nan_to_num(n))
                elif name == "state":
                    parts.append(tgt(d, S_COLS))
                elif name == "random":
                    parts.append(rng.standard_normal((len(d), 32)).astype(np.float32))
                else:
                    raise ValueError(name)
            x = np.concatenate(parts, 1)
            return x + args.oracle_noise * x.std(0) * rng.standard_normal(x.shape).astype(np.float32)

        zs = {s: oracle(d) for s, d in dfs.items()}
        log(f"  oracle latents: {args.oracle} (noise {args.oracle_noise}) dims={zs['train'].shape[1]}")
    else:
        zs = {s: load_latents(d, la_root, log=log) for s, d in dfs.items()}
    D = zs["train"].shape[1]
    pca = None
    if D > args.max_dim:
        log(f"  PCA {D} -> {args.max_dim} (fit on train)")
        mu, comps = fit_pca(zs["train"], args.max_dim)
        zs = {s: apply_pca(z, mu, comps) for s, z in zs.items()}

    tr, va = dfs["train"], dfs["val"]
    un = dfs.get("unseen")
    out = {"model": args.model, "stride": args.stride, "probe": args.probe, "latent_dim": int(D),
           "n": {s: int(len(d)) for s, d in dfs.items()}}

    # --- AD: action decodability (+ per-subset breakdown on val)
    evals = {"val": (zs["val"], tgt(va, A_COLS))}
    if un is not None:
        evals["unseen"] = (zs["unseen"], tgt(un, A_COLS))
    for sub in sorted(va.subset.unique()):
        m = (va.subset == sub).to_numpy()
        evals[f"val/{sub}"] = (zs["val"][m], tgt(va[m], A_COLS))
    out["AD"], p_action = probe_target(args.probe, zs["train"], tgt(tr, A_COLS), grp(tr), evals, GROUPS["a"])
    log(f"AD  R2(a|z)   val={out['AD']['val']['all']:.3f}" + (f"  unseen={out['AD']['unseen']['all']:.3f}" if un is not None else ""))

    # --- habit baseline: action from state alone / from state + z
    res, _ = probe_target(args.probe, tgt(tr, S_COLS), tgt(tr, A_COLS), grp(tr),
                          {"val": (tgt(va, S_COLS), tgt(va, A_COLS))}, GROUPS["a"])
    out["AD_state_only"] = res
    res, _ = probe_target(args.probe, np.concatenate([zs["train"], tgt(tr, S_COLS)], 1), tgt(tr, A_COLS), grp(tr),
                          {"val": (np.concatenate([zs["val"], tgt(va, S_COLS)], 1), tgt(va, A_COLS))}, GROUPS["a"])
    out["AD_state_plus_z"] = res
    log(f"    R2(a|s)={out['AD_state_only']['val']['all']:.3f}   R2(a|z,s)={out['AD_state_plus_z']['val']['all']:.3f}")

    # --- SL: state leakage
    out["SL"], _ = probe_target(args.probe, zs["train"], tgt(tr, S_COLS), grp(tr),
                                {"val": (zs["val"], tgt(va, S_COLS))}, GROUPS["s"])
    log(f"SL  R2(s|z)   val={out['SL']['val']['all']:.3f}")

    # --- NL: nuisance (camera egomotion) leakage, only pairs with head pose
    htr, hva = tr[N_COLS].notna().all(1).to_numpy(), va[N_COLS].notna().all(1).to_numpy()
    if htr.sum() > 100 and hva.sum() > 50:
        out["NL"], _ = probe_target(args.probe, zs["train"][htr], tgt(tr[htr], N_COLS), grp(tr[htr]),
                                    {"val": (zs["val"][hva], tgt(va[hva], N_COLS))}, GROUPS["n"])
        # partial: remove the part of head motion linearly explained by the hand action
        n_all = np.concatenate([tgt(tr[htr], N_COLS), tgt(va[hva], N_COLS)])
        a_all = np.concatenate([tgt(tr[htr], A_COLS), tgt(va[hva], A_COLS)])
        fit_mask = np.arange(len(n_all)) < htr.sum()
        n_perp = residualize(n_all, a_all, fit_mask)
        out["NLp"], _ = probe_target(args.probe, zs["train"][htr], n_perp[fit_mask], grp(tr[htr]),
                                     {"val": (zs["val"][hva], n_perp[~fit_mask])}, GROUPS["n"])
        # how much of head motion is explainable from hand action at all (reference)
        out["N_from_a"], _ = probe_target("ridge", tgt(tr[htr], A_COLS), tgt(tr[htr], N_COLS), grp(tr[htr]),
                                          {"val": (tgt(va[hva], A_COLS), tgt(va[hva], N_COLS))}, GROUPS["n"])
        log(f"NL  R2(n|z)   val={out['NL']['val']['all']:.3f}   partial NLp={out['NLp']['val']['all']:.3f}   (R2(n|a)={out['N_from_a']['val']['all']:.3f})")
        out["selectivity"] = out["AD"]["val"]["all"] - out["NLp"]["val"]["all"]

    # --- CET: cross-embodiment transfer of the action probe
    hum_tr, rob_tr = (tr.type == "human").to_numpy(), (tr.type == "robot").to_numpy()
    hum_va, rob_va = (va.type == "human").to_numpy(), (va.type == "robot").to_numpy()
    if hum_tr.sum() > 200 and rob_tr.sum() > 200:
        cet = {}
        rob_evals = {f"val/{sub}": (zs["val"][(va.subset == sub).to_numpy()], tgt(va[va.subset == sub], A_COLS))
                     for sub in sorted(va[rob_va].subset.unique())}
        rob_evals["val/robot"] = (zs["val"][rob_va], tgt(va[rob_va], A_COLS))
        hum_evals = {"val/human": (zs["val"][hum_va], tgt(va[hum_va], A_COLS))}
        cet["human->robot"], p_h = probe_target(args.probe, zs["train"][hum_tr], tgt(tr[hum_tr], A_COLS), grp(tr[hum_tr]), rob_evals, GROUPS["a"])
        cet["robot->robot"], _ = probe_target(args.probe, zs["train"][rob_tr], tgt(tr[rob_tr], A_COLS), grp(tr[rob_tr]), rob_evals, GROUPS["a"])
        cet["robot->human"], p_r = probe_target(args.probe, zs["train"][rob_tr], tgt(tr[rob_tr], A_COLS), grp(tr[rob_tr]), hum_evals, GROUPS["a"])
        cet["human->human"], _ = probe_target(args.probe, zs["train"][hum_tr], tgt(tr[hum_tr], A_COLS), grp(tr[hum_tr]), hum_evals, GROUPS["a"])
        # Embodiments differ in motion scale (the eva robot moves ~10x slower than a human hand), so a
        # probe fit on one embodiment is also scored after a per-dimension affine recalibration
        # (2 numbers per action dim, fit on the target embodiment's *train* pairs): this isolates
        # "does z carry the same action directions" from "does it carry the same action magnitudes".
        cet["human->robot_calibrated"] = r2_by_group(
            tgt(va[rob_va], A_COLS),
            affine_calibrate(p_h.predict(zs["train"][rob_tr]), tgt(tr[rob_tr], A_COLS), p_h.predict(zs["val"][rob_va])), GROUPS["a"])
        cet["robot->human_calibrated"] = r2_by_group(
            tgt(va[hum_va], A_COLS),
            affine_calibrate(p_r.predict(zs["train"][hum_tr]), tgt(tr[hum_tr], A_COLS), p_r.predict(zs["val"][hum_va])), GROUPS["a"])
        cet["transfer_ratio_h2r"] = cet["human->robot"]["val/robot"]["all"] / max(cet["robot->robot"]["val/robot"]["all"], 1e-6)
        cet["transfer_ratio_r2h"] = cet["robot->human"]["val/human"]["all"] / max(cet["human->human"]["val/human"]["all"], 1e-6)
        cet["transfer_ratio_h2r_calibrated"] = cet["human->robot_calibrated"]["all"] / max(cet["robot->robot"]["val/robot"]["all"], 1e-6)
        cet["transfer_ratio_r2h_calibrated"] = cet["robot->human_calibrated"]["all"] / max(cet["human->human"]["val/human"]["all"], 1e-6)
        out["CET"] = cet
        log(f"CET human->robot R2={cet['human->robot']['val/robot']['all']:.3f} calibrated={cet['human->robot_calibrated']['all']:.3f} (in-domain {cet['robot->robot']['val/robot']['all']:.3f})"
            f"   robot->human R2={cet['robot->human']['val/human']['all']:.3f} calibrated={cet['robot->human_calibrated']['all']:.3f} (in-domain {cet['human->human']['val/human']['all']:.3f})")

    # --- COMP: compositionality across strides (needs stride-15 model here and a stride-30 val csv)
    if args.comp_val_csv and args.stride == 15 and not args.oracle:
        d30 = pd.read_csv(args.comp_val_csv).dropna(subset=["la_path"])
        if args.pairs_dir:
            d30 = refresh_labels(d30, os.path.join(args.pairs_dir, "pairs_val_stride30.csv"))
        d30 = clean_pairs(d30, log=log, stride=30)
        key = lambda d: list(zip(d.subset, d.episode_id, d.src_frame))
        idx15 = {k: i for i, k in enumerate(key(va))}
        rows = []
        for i, (sub, ep, t) in enumerate(key(d30)):
            j1, j2 = idx15.get((sub, ep, t)), idx15.get((sub, ep, t + 15))
            if j1 is not None and j2 is not None:
                rows.append((i, j1, j2))
        if len(rows) > 50:
            rows = np.array(rows)
            a30 = tgt(d30.iloc[rows[:, 0]], A_COLS)
            pred = p_action.predict(zs["val"][rows[:, 1]]) + p_action.predict(zs["val"][rows[:, 2]])
            gt_sum = tgt(va.iloc[rows[:, 1]], A_COLS) + tgt(va.iloc[rows[:, 2]], A_COLS)
            out["COMP"] = {"n": int(len(rows)),
                           "R2_sum_of_decoded_vs_a30": r2_by_group(a30, pred, GROUPS["a"]),
                           "R2_sum_of_gt15_vs_a30 (label-only upper bound)": r2_by_group(a30, gt_sum, GROUPS["a"])}
            log(f"COMP R2(a15+a15 decoded vs a30)={out['COMP']['R2_sum_of_decoded_vs_a30']['all']:.3f}  (labels-only bound {out['COMP']['R2_sum_of_gt15_vs_a30 (label-only upper bound)']['all']:.3f})")

    return out


def summary_table(results):
    """One-line-per-model markdown table from a list of result dicts."""
    cols = ["model", "stride", "probe", "AD_val", "AD_unseen", "AD_pos", "AD_rot", "R2(a|s)", "SL", "NL", "NLp", "select", "CET_h2r", "CET_h2r_cal", "CET_r2h_cal", "COMP"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    g = lambda d, *ks: (lambda v: "" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))(_get(d, *ks))
    for r in results:
        lines.append("| " + " | ".join([
            str(r["model"]), str(r["stride"]), r["probe"], g(r, "AD", "val", "all"), g(r, "AD", "unseen", "all"),
            g(r, "AD", "val", "pos"), g(r, "AD", "val", "rot"), g(r, "AD_state_only", "val", "all"), g(r, "SL", "val", "all"),
            g(r, "NL", "val", "all"), g(r, "NLp", "val", "all"), g(r, "selectivity"),
            g(r, "CET", "transfer_ratio_h2r"), g(r, "CET", "transfer_ratio_h2r_calibrated"), g(r, "CET", "transfer_ratio_r2h_calibrated"), g(r, "COMP", "R2_sum_of_decoded_vs_a30", "all")]) + " |")
    return "\n".join(lines)


def _get(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--unseen-csv")
    ap.add_argument("--comp-val-csv", help="stride-30 val CSV with la_path of the same model (for COMP)")
    ap.add_argument("--la-root")
    ap.add_argument("--pairs-dir", help="directory with pairs_<split>_stride<k>.csv; labels are re-read from there")
    ap.add_argument("--probe", choices=["ridge", "mlp"], default="ridge")
    ap.add_argument("--max-dim", type=int, default=4096)
    ap.add_argument("--oracle", help="control run: use label features as z, e.g. action, head, state, random, action+head")
    ap.add_argument("--oracle-noise", type=float, default=0.0, help="relative gaussian noise added to oracle features")
    ap.add_argument("--out", required=True, help="json output path")
    args = ap.parse_args()
    res = run(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(summary_table([res]))


if __name__ == "__main__":
    main()
