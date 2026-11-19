"""Extract EgoLAM codes from one official LARYBench split CSV."""
import argparse
from pathlib import Path
import pandas as pd
from lary.extract import ExtractionConfig, LatentActionExtractor, setup_seed
from evaluation.lary_adapter import FrozenLamLaryWrap

class EgoExtractor(LatentActionExtractor):
    def __init__(self, config):
        self.config = config
        self.model = FrozenLamLaryWrap()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("csv","dataset","split","output"):
        parser.add_argument("--"+key, required=True)
    parser.add_argument("--mode",choices=("image","video"),required=True)
    parser.add_argument("--stride",type=int,default=5)
    parser.add_argument("--batch-size",type=int,default=1)
    a=parser.parse_args()
    setup_seed(42)
    root=Path(a.output).resolve()
    out=root/a.dataset/a.split/"EgoLAM"
    out.mkdir(parents=True,exist_ok=False)
    cfg=ExtractionConfig(model="EgoLAM",dataset=a.dataset,split=a.split,mode=a.mode,
                         stride=a.stride,batch_size=a.batch_size)
    df=pd.read_csv(a.csv)
    df["la_path"]=""
    result=EgoExtractor(cfg).extract(df,str(out))
    if result.la_path.isna().any() or (result.la_path=="").any():
        raise ValueError("Incomplete split extraction")
    result["la_path"]=[str(root/p) for p in result.la_path]
    manifest=root/f"{a.split}_la_{a.dataset}_EgoLAM.csv"
    result.to_csv(manifest,index=False)
    print(manifest)
if __name__=="__main__":
    main()
