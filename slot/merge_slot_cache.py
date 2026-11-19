"""Merge the per-rank slot-cache shards into one frame_id -> (shard, row) lookup.

Each rank wrote frames_r{r}.npy (global frame ids, in the order it processed them).
We concatenate them, then build a dense int64 lookup of length n_frames_total whose
value encodes shard and row, or -1 if that frame was never cached.
"""
import glob
import json
import os

import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("cache")
parser.add_argument("labels_root")
args = parser.parse_args()
CACHE = args.cache
LABELS = os.path.join(args.labels_root, "consolidated")

shards = sorted(glob.glob(os.path.join(CACHE, "frames_r*.npy")))
print(f"shards: {len(shards)}")
n_total = len(np.load(os.path.join(LABELS, "ep_of_frame.npy"), mmap_mode="r"))
lookup = np.full(n_total, -1, dtype=np.int64)

sizes = []
for p in shards:
    r = int(os.path.basename(p).split("_r")[1].split(".")[0])
    f = np.load(p)
    s = np.load(os.path.join(CACHE, f"slots_r{r}.npy"), mmap_mode="r")
    assert len(f) == s.shape[0], f"rank{r}: frames {len(f)} != slots {s.shape[0]}"
    dup = lookup[f] != -1
    if dup.any():
        raise SystemExit(f"rank{r}: {int(dup.sum())} duplicate frame ids")
    lookup[f] = (r << 32) | np.arange(len(f), dtype=np.int64)
    sizes.append((r, len(f)))
    print(f"  rank{r}: {len(f):>9,} rows  slots{s.shape}")

missing = int((lookup == -1).sum())
np.save(os.path.join(CACHE, "frame_lookup.npy"), lookup)
meta = {"n_frames_total": n_total, "n_cached": int((lookup != -1).sum()),
        "missing": missing, "shards": sizes,
        "K": int(s.shape[1]), "D": int(s.shape[2]), "mask_grid": 16}
json.dump(meta, open(os.path.join(CACHE, "meta.json"), "w"), indent=1)
print(f"\ncached {meta['n_cached']:,} / {n_total:,} frames, missing {missing:,}")
print("wrote", os.path.join(CACHE, "frame_lookup.npy"))

