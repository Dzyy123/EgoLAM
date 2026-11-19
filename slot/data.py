"""Frame-chunk dataset for slot pretraining (no labels needed)."""
import json, os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FrameChunks(Dataset):
    """Return T consecutive stored frames (0.4 s apart) from a random episode."""

    def __init__(self, frames_root, labels_root, image_size=224, n_frames=4, n_samples=100000, seed=0):
        c = os.path.join(labels_root, "consolidated")
        self.root = frames_root
        self.rows = [json.loads(l) for l in open(os.path.join(c, "episodes.jsonl"))]
        off = np.load(os.path.join(c, "frame_offset.npy"))
        end = np.append(off[1:], len(np.load(os.path.join(c, "ep_of_frame.npy"), mmap_mode="r")))
        lens = (end - off).astype(np.int64)
        keep = np.nonzero(lens > n_frames + 1)[0]
        self.rows = [self.rows[i] for i in keep]
        self.lens = torch.as_tensor(lens[keep], dtype=torch.float64)
        self.size, self.T, self.n, self.seed = image_size, n_frames, n_samples, seed

    def __len__(self):
        return self.n

    def _img(self, e, k):
        p = os.path.join(self.root, self.rows[e]["path"], f"{k:06d}.jpg")
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:                       # missing / undecodable frame
            raise FileNotFoundError(p)
        im = cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i * 7919 + self.seed)
        e = int(torch.multinomial(self.lens, 1, generator=g))
        K = int(self.lens[e].item())
        k0 = int(torch.randint(0, K - self.T + 1, (1,), generator=g))   # inclusive upper bound
        frames = np.stack([self._img(e, k0 + t) for t in range(self.T)])
        return torch.from_numpy(np.ascontiguousarray(frames.transpose(0, 3, 1, 2)))   # [T,3,H,W] uint8