"""Freeze the 100k OC slots into a per-frame cache, so the downstream LAM never reads images.

Runs under torchrun (one rank per GPU).  Each rank owns the episodes with
`episode_idx % world == rank` and writes its own shard, so no two processes ever write
the same file.  A merge step (slot/merge_slot_cache.py) then builds the frame_id -> row
lookup used by the trainer.

Per frame we store
  slots  [6, 128]  float16   the slot vectors (what the LAM consumes)
  masks  [6, 256]  uint8     decoder attention masks downsampled 37x37 -> 16x16.
                             Only used by the hand BCE, which lives on the 16x16 grid
                             of the DINOv3 features -- keeping it there means the loss
                             is byte-identical to the SlotLAM baseline's hand term.
  frames [ ]       int64     global frame id (episode offset + local k)
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class FrameSet(Dataset):
    """All frames of the episodes this rank owns."""

    def __init__(self, frames_root, rows, ep_idxs, off, n_frames_total, size):
        self.root, self.size = frames_root, size
        self.paths = {e: rows[e]["path"] for e in ep_idxs}
        self.off, self.n_total = off, n_frames_total
        items = []
        for e in ep_idxs:
            k0 = int(off[e])
            k1 = int(off[e + 1]) if e + 1 < len(off) else n_frames_total
            for k in range(k1 - k0):
                items.append((e, k))
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        e, k = self.items[i]
        p = os.path.join(self.root, self.paths[e], f"{k:06d}.jpg")
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(p)
        im = cv2.cvtColor(cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_CUBIC),
                          cv2.COLOR_BGR2RGB)
        return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))), int(self.off[e]) + k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many frames per rank")
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    from slot.eval_oc_slots import load_model
    model, cfg = load_model(a.ckpt)
    size, grid = cfg.img_size, int(round(cfg.n_patches ** 0.5))
    K, D = cfg.n_slots, cfg.slot_dim
    if rank == 0:
        print(f"[cache] K={K} D={D} size={size} grid={grid} ckpt={os.path.basename(a.ckpt)}", flush=True)

    C = os.path.join(a.labels_root, "consolidated")
    rows = [json.loads(l) for l in open(os.path.join(C, "episodes.jsonl"))]
    off = np.load(os.path.join(C, "frame_offset.npy"))
    n_frames_total = int(len(np.load(os.path.join(C, "ep_of_frame.npy"), mmap_mode="r")))
    n_ep = len(rows)
    mine = [e for e in range(n_ep) if e % world == rank]

    ds = FrameSet(a.frames_root, rows, mine, off, n_frames_total, size)
    if a.limit:
        ds.items = ds.items[: a.limit]
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                    pin_memory=True, drop_last=False)
    if rank == 0:
        print(f"[cache] this rank: {len(ds)} frames over {len(mine)} episodes", flush=True)

    os.makedirs(a.out, exist_ok=True)
    n = len(ds)
    sp = os.path.join(a.out, f"slots_r{rank}.npy")
    mp = os.path.join(a.out, f"masks_r{rank}.npy")
    fp = os.path.join(a.out, f"frames_r{rank}.npy")
    slots = np.lib.format.open_memmap(sp, mode="w+", dtype=np.float16, shape=(n, K, D))
    masks = np.lib.format.open_memmap(mp, mode="w+", dtype=np.uint8, shape=(n, K, 16 * 16))
    fids = np.lib.format.open_memmap(fp, mode="w+", dtype=np.int64, shape=(n,))

    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    at = 0
    for bi, (img, fid) in enumerate(dl):
        x = img.cuda().float().div_(255.0)
        x = (x - mean) / std
        # bf16 autocast: the checkpoint was TRAINED under autocast, so this is also the
        # numerically matching regime -- and it is ~5x faster than fp32 here.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x[:, None], compute_decoder=True)     # [B,1,3,H,W]
            s = out["slots"][:, 0].float()                    # [B,K,D]
            m = out["decoder_masks"][:, 0].float()            # [B,K,N]
            m = m.reshape(m.shape[0], K, grid, grid)
            m16 = F.adaptive_avg_pool2d(m, 16).reshape(m.shape[0], K, 256)
        b = s.shape[0]
        slots[at:at + b] = s.half().cpu().numpy()
        masks[at:at + b] = (m16.clamp(0, 1) * 255).round().byte().cpu().numpy()
        fids[at:at + b] = fid.numpy()
        at += b
        if rank == 0 and bi % 200 == 0:
            print(f"[cache] batch {bi}/{len(dl)} rows={at}/{n}", flush=True)

    slots.flush(); masks.flush(); fids.flush()
    dist.barrier()
    if rank == 0:
        print(f"[cache] DONE rows_this_rank={n}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

