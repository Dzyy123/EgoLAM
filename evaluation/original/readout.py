"""Run the single-GPU native LARYBench readout on pre-extracted EgoLAM codes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    strides={"calvin":5,"vlabench":5,"robocoin":10,"agibotbeta":45}
    classes={"human_1st":123,"robot_1st":54}
    p.add_argument("task",choices=[*classes,*strides])
    for key in ("train","val","output","lary-root"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--stats-json",type=Path)
    a=p.parse_args()
    a.output=a.output.resolve()
    runtime=a.output/"data"
    runtime.mkdir(parents=True,exist_ok=False)
    for split,src in [("train",a.train),("val",a.val)]:
        if not src.is_file(): raise FileNotFoundError(src)
        (runtime/f"{split}_la_{a.task}_EgoLAM.csv").symlink_to(src.resolve())
    source=Path(__file__).resolve().parent
    env=dict(os.environ,LARY_ROOT=str(a.output),LARY_LOG_DIR=str(a.output),
             PYTHONPATH=str(a.lary_root.resolve())+os.pathsep+os.environ.get("PYTHONPATH",""),WANDB_MODE="disabled")
    if a.task in classes:
        cmd=[sys.executable,str(source/"classification_main.py"),"--fname",str(source/"manipulation.yaml"),
             "--lam","EgoLAM","--dataset",a.task,"--dim","128","--classes",str(classes[a.task]),
             "--batch_size","256","--devices","cuda:0"]
    else:
        cmd=[sys.executable,str(source/"regression_main.py"),"--model_type","mlp","--train_csv",str(a.train.resolve()),
             "--val_csv",str(a.val.resolve()),"--save_dir",str(a.output/"readout"),"--batch_size","256","--epochs","50",
             "--lr","0.0001","--dataset",a.task,"--stride",str(strides[a.task]),"--action_mode","absolute"]
        if a.stats_json:cmd+=["--global_stats_json",str(a.stats_json.resolve())]
    subprocess.run(cmd,env=env,check=True)
if __name__=="__main__":
    main()
