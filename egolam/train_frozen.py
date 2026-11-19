"""Stage-3 training: SlotLAM's LAM on frozen object slots.

Slots come from the cache (no image read for s_t/s_tk), but the camera-invariance
term needs the warped next frame re-encoded by the frozen OC encoder, so images ARE
read for that branch.  That cost is deliberate (see frozen_lam.py).

usage:
  torchrun --nproc_per_node=4 -m egolam.train_frozen \
    --out runs/egolam --cache ... --inv-frac 1.0
"""
import argparse
import json
import os
import shutil
import time
from dataclasses import asdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

from egolam.model import SlotLAMConfig, random_similarity_warp
from egolam.frozen_lam import FrozenSlotLAM, frozen_losses, save_checkpoint

VAL_OC = [None]
VAL_CFG = [None]
N_SHARDS = None          # discovered from the cache directory (varies with world size)


class FrozenPairs(Dataset):
    """(frame k, frame k+gap) pairs -> labels; slots are fetched in the main process."""

    def __init__(self, labels_root, split, gap, grid=16, max_pairs=0, seed=0,
                 frames_root="", oc_size=518):
        c = os.path.join(labels_root, "consolidated")
        self.grid, self.gap = grid, gap
        self.frames_root, self.oc_size = frames_root, oc_size
        self.rows = [json.loads(l) for l in open(os.path.join(c, "episodes.jsonl"))]
        self.pairs = np.load(os.path.join(c, f"pairs_gap{gap}_{split}.npy"))
        if max_pairs and len(self.pairs) > max_pairs:
            self.pairs = np.sort(np.random.default_rng(seed).choice(self.pairs, max_pairs, replace=False))
        mm = lambda n: np.load(os.path.join(c, n), mmap_mode="r")  # noqa: E731
        self.ep, self.off = mm("ep_of_frame.npy"), np.load(os.path.join(c, "frame_offset.npy"))
        self.a, self.n = mm(f"a{gap}.npy"), mm(f"n{gap}.npy")
        self.kp2d, self.kpv = mm("kp2d.npy"), mm("kp_valid.npy")
        self.has_head = np.array([r["has_head"] for r in self.rows])
        self.is_eva = np.array([r["subset"] == "eva" for r in self.rows])
        ys, xs = np.mgrid[0:grid, 0:grid]
        self.cells = np.stack([xs + 0.5, ys + 0.5], -1).reshape(-1, 2).astype(np.float32)
        self.stats = None

    def set_stats(self, g):
        self.stats = g

    def __getattr__(self, k):                     # a_mu etc. resolved from stats lazily
        if k in ("a_mu", "a_sd", "n_mu", "n_sd"):
            g = self.__dict__.get("stats") or {}
            key = {"a_mu": "a_mean", "a_sd": "a_std", "n_mu": "n_mean", "n_sd": "n_std"}[k]
            return np.array(g.get(key, np.zeros(12 if k[0] == "a" else 6)), np.float32)
        raise AttributeError(k)

    def __len__(self):
        return len(self.pairs)

    def _heat(self, f):
        heat, valid = np.zeros((2, self.grid ** 2), np.float32), np.zeros(2, bool)
        for h in range(2):
            if not self.kpv[f, h]:
                continue
            pts = np.asarray(self.kp2d[f, h], np.float32)
            pts = pts[np.isfinite(pts).all(1)]
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < 1) & (pts[:, 1] >= 0) & (pts[:, 1] < 1)] * self.grid
            if len(pts) < 5:
                continue
            valid[h] = True
            d2 = ((self.cells[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
            heat[h] = np.exp(-d2 / 2.0).sum(1)
            heat[h] /= heat[h].max() + 1e-6
        return heat, valid

    def _img(self, e, k):
        p = os.path.join(self.frames_root, self.rows[e]["path"], f"{k:06d}.jpg")
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(p)
        im = cv2.cvtColor(cv2.resize(im, (self.oc_size, self.oc_size), interpolation=cv2.INTER_CUBIC),
                          cv2.COLOR_BGR2RGB)
        return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)))      # uint8 [3,H,W]

    def __getitem__(self, i):
        f = int(self.pairs[i])
        e = int(self.ep[f])
        k = f - int(self.off[e])
        a = (np.asarray(self.a[f], np.float32) - self.a_mu) / self.a_sd
        if self.has_head[e]:
            n, n_valid = (np.asarray(self.n[f], np.float32) - self.n_mu) / self.n_sd, True
        elif self.is_eva[e]:
            n, n_valid = (0.0 - self.n_mu) / self.n_sd, True
        else:
            n, n_valid = np.zeros(6, np.float32), False
        h_t, v_t = self._heat(f)
        h_tk, v_tk = self._heat(f + self.gap)
        out = {"f": f, "heat_t": torch.from_numpy(h_t), "heat_tk": torch.from_numpy(h_tk),
               "hand_valid": torch.from_numpy(v_t & v_tk),
               "action": torch.from_numpy(a.astype(np.float32)), "action_valid": True,
               "nuis": torch.from_numpy(np.asarray(n, np.float32)), "nuis_valid": n_valid}
        if self.frames_root:
            out["img_tk"] = self._img(e, k + self.gap)
        return out


class SlotStore:
    def __init__(self, cache_dir):
        self.lookup = np.load(os.path.join(cache_dir, "frame_lookup.npy"))
        import glob as _glob
        sp = sorted(_glob.glob(os.path.join(cache_dir, "slots_r*.npy")))
        self.slots = [np.load(p, mmap_mode="r") for p in sp]
        self.masks = [np.load(p.replace("slots_r", "masks_r"), mmap_mode="r") for p in sp]

    def gather(self, fids):
        loc = self.lookup[fids]
        if (loc < 0).any():
            raise KeyError(f"{int((loc < 0).sum())} frame ids missing from the cache")
        shard = (loc >> 32).astype(np.int64)
        row = (loc & 0xFFFFFFFF).astype(np.int64)
        K, D = self.slots[0].shape[1:]
        s = np.empty((len(fids), K, D), np.float16)
        m = np.empty((len(fids), K, 256), np.uint8)
        for r in np.unique(shard):
            sel = shard == r
            s[sel] = self.slots[r][row[sel]]
            m[sel] = self.masks[r][row[sel]]
        return s, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--oc-ckpt", required=True)
    ap.add_argument("--stats-from", default="configs/train.json")
    ap.add_argument("--gap", type=int, default=3)
    ap.add_argument("--val-split", default="unseen_test")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--num-train-steps", type=int, default=100005)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-pairs", type=int, default=4096)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--inv-frac", type=float, default=1.0, help="fraction of each batch that gets the warp path")
    ap.add_argument("--w-inv", type=float, default=1.0)
    ap.add_argument("--n-slots", type=int, default=0,
                    help="override n_slots when the slot cache has a different K")
    ap.add_argument("--z-all-slots", action="store_true",
                    help="build z from every slot temporal difference, not hand pools")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--stop-at", type=int, default=56000, help="Final update, independent of the LR horizon")
    a = ap.parse_args()
    if not 0 < a.stop_at <= a.num_train_steps:
        ap.error("stop-at must be within the LR horizon")

    base = json.load(open(a.stats_from))
    w = {"fdm": base["args"]["w_fdm"], "hand": base["args"]["w_hand"],
         "action": base["args"]["w_action"], "adv": base["args"]["w_adv"], "inv": a.w_inv}
    adv_lambda = base["args"]["adv_lambda"]
    stats = {"a_mean": base["a_stats"][0], "a_std": base["a_stats"][1],
             "n_mean": base["n_stats"][0], "n_std": base["n_stats"][1]}

    acc = Accelerator()
    dev, main_proc = acc.device, acc.is_main_process
    use_bf16 = not a.no_bf16
    if main_proc:
        os.makedirs(a.out, exist_ok=True)
        print(f"[frozen] steps={a.num_train_steps} bs/gpu={a.batch_size} world={acc.num_processes} "
              f"gbatch={a.batch_size*acc.num_processes} lr={a.lr} w={w} inv_frac={a.inv_frac}", flush=True)

    do_inv = a.inv_frac > 0
    ds = FrozenPairs(a.labels_root, "train", a.gap, frames_root=a.frames_root, oc_size=518)
    ds.set_stats(stats)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
                    pin_memory=True, drop_last=True, persistent_workers=a.num_workers > 0)
    store = SlotStore(a.cache)

    oc = None
    oc_cfg = None
    if True:                                  # fdm needs the decoder + features even without inv
        from slot.eval_oc_slots import load_model
        oc, oc_cfg = load_model(a.oc_ckpt)
        for p in oc.parameters():
            p.requires_grad_(False)
        oc.eval()
        VAL_OC[0], VAL_CFG[0] = oc, oc_cfg

    cfg = SlotLAMConfig(**base["config"])
    if a.n_slots:
        cfg.n_slots = a.n_slots
    if a.z_all_slots:
        cfg.z_from_all_slots = True
    torch.manual_seed(0)
    # --- consistency guards: cache K, oc-encoder K and model K must agree ---
    _k_cache = int(store.slots[0].shape[1])
    _d_cache = int(store.slots[0].shape[2])
    _k_oc = int(oc_cfg.n_slots) if oc_cfg is not None else -1
    if _k_oc != _k_cache:
        raise SystemExit(f"FATAL: slot cache K={_k_cache} but --oc-ckpt K={_k_oc}; "
                         f"pass the oc checkpoint that produced this cache")
    if int(cfg.n_slots) != _k_cache:
        raise SystemExit(f"FATAL: slot cache K={_k_cache} but model cfg n_slots={int(cfg.n_slots)}; pass --n-slots")
    if main_proc:
        print(f"[frozen] cache K={_k_cache} D={_d_cache} | oc K={_k_oc} | model K={int(cfg.n_slots)}", flush=True)

    lam = FrozenSlotLAM(cfg)
    if main_proc:
        print(f"[frozen] trainable params = {sum(p.numel() for p in lam.parameters())/1e6:.3f} M", flush=True)
    opt = torch.optim.AdamW(lam.parameters(), lr=a.lr, weight_decay=a.wd)
    total, warm = a.num_train_steps, max(1, int(a.warmup_frac * a.num_train_steps))

    def lr_at(s):
        if s < warm:
            return a.lr * s / warm
        p = (s - warm) / max(1, total - warm)
        return a.lr * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * p)))

    lam, opt, dl = acc.prepare(lam, opt, dl)
    lam.train()
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    step, t0 = 0, time.time()
    while step < a.stop_at:
        for b in dl:
            for gp in opt.param_groups:
                gp["lr"] = lr_at(step)
            fids = np.stack([b["f"].detach().cpu().numpy(),
                             b["f"].detach().cpu().numpy() + a.gap], 1).reshape(-1)
            s, m = store.gather(fids)
            K, D = s.shape[1:]
            B = len(b["f"])
            s = torch.from_numpy(s).to(dev, non_blocking=True).float().view(B, 2, K, D)
            m = torch.from_numpy(m).to(dev, non_blocking=True).float().view(B, 2, K, 256).div_(255.0)
            s_t, s_tk = s[:, 0], s[:, 1]
            masks_t, masks_tk = m[:, 0], m[:, 1]

            img_tk = b["img_tk"].to(dev, non_blocking=True).float().div_(255.0)      # [B,3,H,W]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                feat_tk = oc.backbone((img_tk - MEAN) / STD)["vit_block12"]          # [B,N,768]

            s_tk_w, inv_mask = None, None
            if do_inv:
                k = max(1, int(round(a.inv_frac * B)))
                inv_mask = torch.zeros(B, dtype=torch.bool, device=dev)
                inv_mask[:k] = True
                img = img_tk[:k]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    warped, _ = random_similarity_warp(img)
                    xw = (warped - MEAN) / STD
                    feats_w = oc.backbone(xw)["vit_block12"]           # [k,N,768]
                    proj_w = oc.proj(feats_w)                          # [k,N,128]
                    state = oc.predictor(s_t[:k].detach())             # == baseline init_slots=s_t
                    s_tk_w = oc.grouper(state, proj_w, n_iters=oc_cfg.n_iters)["slots"].float()

            with acc.accumulate():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    out = lam(s_t, s_tk, b["nuis"], b["nuis_valid"].bool(),
                              adv_lambda=adv_lambda, s_tk_warp=s_tk_w)
                out = {k_: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                       for k_, v in out.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    out["recon_pred_tk"] = oc.decoder(out["pred_tk"])["reconstruction"][..., :oc_cfg.feat_dim]
                out["feat_tk"] = feat_tk
                L = frozen_losses(out, s_tk, masks_t, masks_tk, b["heat_t"], b["heat_tk"], b["hand_valid"],
                                  b["action"], b["action_valid"], b["nuis"], b["nuis_valid"], w,
                                  inv_mask=inv_mask)
                acc.backward(L["total"])
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.log_every == 0 and main_proc:
                rec = {"step": step, **{k_: round(float(v), 4) for k_, v in L.items() if k_ != "total"},
                       "total": round(float(L["total"]), 4), "lr": lr_at(step),
                       "s_per_step": round((time.time() - t0) / a.log_every, 3)}
                t0 = time.time()
                print(json.dumps(rec), flush=True)
                with open(os.path.join(a.out, "log.jsonl"), "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            if step % a.val_every == 0:
                validate(acc, lam, a, stats, w, adv_lambda, store, main_proc, step)
            if step % a.save_every == 0 or step == a.stop_at:
                acc.wait_for_everyone()
                if main_proc:
                    save_checkpoint(os.path.join(a.out, "frozen_lam.pt"), acc.unwrap_model(lam), cfg,
                                    {"step": step, "a_stats": base["a_stats"], "n_stats": base["n_stats"],
                                     "slot_source": a.cache})
                    ck = os.path.join(a.out, "ckpts", f"step_{step:06d}")
                    os.makedirs(ck, exist_ok=True)
                    shutil.copyfile(os.path.join(a.out, "frozen_lam.pt"), os.path.join(ck, "frozen_lam.pt"))
                    json.dump({"args": vars(a), "config": asdict(cfg), "w": w, "adv_lambda": adv_lambda},
                              open(os.path.join(a.out, "run.json"), "w"), indent=1)
            if step >= a.stop_at:
                break
    acc.wait_for_everyone()
    if main_proc:
        print("FROZEN_TRAIN_DONE", flush=True)


@torch.no_grad()
def validate(acc, lam, a, stats, w, adv_lambda, store, main_proc, step):
    """Losses + val_AD_proxy on the held-out split, straight from the cache (no inv term:
    a warped frame would need a fresh encoder pass, that is what training does)."""
    m = acc.unwrap_model(lam)
    m.eval()
    dv = FrozenPairs(a.labels_root, a.val_split, a.gap, max_pairs=a.val_pairs, seed=123,
                     frames_root=a.frames_root, oc_size=518)
    dv.set_stats(stats)
    vl = DataLoader(dv, batch_size=128, shuffle=False, num_workers=4)
    agg, n, zs, ys = {}, 0, [], []
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=acc.device).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=acc.device).view(1, 3, 1, 1)
    for b in vl:
        fids = np.stack([b["f"].detach().cpu().numpy(),
                         b["f"].detach().cpu().numpy() + a.gap], 1).reshape(-1)
        s, mk = store.gather(fids)
        K, D = s.shape[1:]
        B = len(b["f"])
        s = torch.from_numpy(s).to(acc.device).float().view(B, 2, K, D)
        mk = torch.from_numpy(mk).to(acc.device).float().view(B, 2, K, 256).div_(255.0)
        bb = {k: (v.to(acc.device) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(s[:, 0], s[:, 1], bb["nuis"], bb["nuis_valid"].bool(), adv_lambda=adv_lambda)
        out = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()}
        img_tk = bb["img_tk"].float().div_(255.0)
        feat_tk = VAL_OC[0].backbone((img_tk - MEAN) / STD)["vit_block12"]
        out["recon_pred_tk"] = VAL_OC[0].decoder(out["pred_tk"])["reconstruction"][..., :VAL_CFG[0].feat_dim]
        out["feat_tk"] = feat_tk
        L = frozen_losses(out, s[:, 1], mk[:, 0], mk[:, 1], bb["heat_t"], bb["heat_tk"], bb["hand_valid"],
                          bb["action"], bb["action_valid"], bb["nuis"], bb["nuis_valid"], w)
        for k, v in L.items():
            agg[k] = agg.get(k, 0.0) + float(v) * B
        n += B
        zs.append(out["z"].cpu().double())
        ys.append(bb["action"].cpu().double())
    m.train()
    res = {k: v / n for k, v in agg.items()}
    z, y = torch.cat(zs), torch.cat(ys)
    perm = torch.randperm(len(z), generator=torch.Generator().manual_seed(0))
    z, y = z[perm], y[perm]
    h = len(z) // 2
    x = (z - z[:h].mean(0)) / (z[:h].std(0) + 1e-6)
    xa = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1)
    wgt = torch.linalg.solve(xa[:h].T @ xa[:h] + torch.eye(xa.shape[1], dtype=x.dtype), xa[:h].T @ y[:h])
    res["val_AD_proxy"] = float(1 - ((xa[h:] @ wgt - y[h:]) ** 2).mean(0).div(y[h:].var(0) + 1e-8).mean())
    res["z_std"] = float(z.std(0).mean())
    if main_proc:
        print(json.dumps({"step": step, "val": {k: round(v, 4) for k, v in res.items()}}), flush=True)
        with open(os.path.join(a.out, "val.jsonl"), "a") as fh:
            fh.write(json.dumps({"step": step, "val": {k: round(v, 4) for k, v in res.items()}}) + "\n")


if __name__ == "__main__":
    main()

