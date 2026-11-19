"""Native EgoLAM image-pair preprocessing for frozen readouts."""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def image(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(cv2.resize(bgr,(518,518),interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2RGB)
    return torch.from_numpy((rgb / 255.).astype(np.float32)).permute(2,0,1)

class Pairs(Dataset):
    def __init__(self, frame):
        self.paths = list(zip(frame.src_img,frame.tgt_img))
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        return i, torch.stack([image(p) for p in self.paths[i]])
