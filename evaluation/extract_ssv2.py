"""Create the eight-pair cache for a composition or effect manifest."""
import argparse
import csv
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from abilities.train import pack
from egolam.inference import encode, load_models

def read_clip(path, indices):
    cap=cv2.VideoCapture(str(path))
    frames={}
    last=-1
    try:
        for i in range(max(indices)+1):
            ok, bgr=cap.read()
            if not ok:
                break
            last=i
            last_bgr=bgr
            if i in indices:
                frames[i]=bgr
    finally:
        cap.release()
    if last<0:
        raise ValueError(f"Empty video: {path}")
    frames[last]=last_bgr
    return [cv2.cvtColor(cv2.resize(frames[min(i,last)],(518,518),interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2RGB) for i in indices]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True,type=Path)
    p.add_argument("--video-root",type=Path,default=Path("."))
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    oc,lam=load_models(os.environ["MODEL_DIR"])
    videos=json.loads(a.manifest.read_text())["videos"]
    a.output.mkdir(parents=True,exist_ok=False)
    csv_path=a.output/"clips.csv"
    with csv_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["sample_id","video_path","sample_indices","la_path"])
        writer.writeheader()
        for i,v in enumerate(videos):
            indices=v["frame_indices"]
            if len(indices)!=9 or min(indices)<0:
                raise ValueError("Expected nine valid frame indices")
            x=torch.from_numpy(np.stack(read_clip(a.video_root/v["video_path"],indices))).permute(0,3,1,2).float()/255.
            torch.manual_seed(42+i)
            z=encode(oc,lam,torch.stack((x[:-1],x[1:]),dim=1)).cpu().numpy()
            if z.shape!=(8,1,128) or not np.isfinite(z).all():
                raise ValueError("Unexpected EgoLAM features")
            dst=(a.output/f"{i:06d}.npz").resolve()
            np.savez(dst,tokens=z)
            writer.writerow(dict(sample_id=v["sample_id"],video_path=v["video_path"],sample_indices=",".join(map(str,indices)),la_path=str(dst)))
    pack(a.manifest,csv_path,a.output,a.output/"packed")

if __name__=="__main__":
    main()
