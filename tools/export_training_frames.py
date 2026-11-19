"""Export unmodified front-camera JPEGs every 12 raw steps from EgoVerse RLDS."""
import argparse
import json
from pathlib import Path
from build_egoverse_pairs import SUBSETS
from build_frame_labels import BUILDER, STRIDE

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlds-root",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/"episode_list.txt").open("w") as manifest:
        for subset in SUBSETS.values():
            for split in ("train","seen_test","unseen_test"):
                base=args.rlds_root/subset/"ego_verse_infidata"/"1.0.0"
                index=0
                for shard in sorted(base.glob(f"*-{split}.tfrecord-*")):
                    for raw in tf.data.TFRecordDataset([str(shard)]):
                        ex=tf.train.Example.FromString(raw.numpy())
                        images=ex.features.feature["steps/observation/images/front_1"].bytes_list.value
                        if not images:
                            raise ValueError(f"Empty front camera in {shard}")
                        rel=Path("frames")/BUILDER.format(subset)/split/f"ep{index:06d}"
                        target=args.output/rel
                        target.mkdir(parents=True)
                        for k,t in enumerate(range(0,len(images),STRIDE)):
                            (target/f"{k:06d}.jpg").write_bytes(images[t])
                        (target/"meta.json").write_text(json.dumps({"num_stored_frames":k+1}))
                        manifest.write(str(rel)+"\n")
                        index+=1
    if not (args.output/"episode_list.txt").stat().st_size:
        raise ValueError("No matching RLDS shards in the supplied root")

if __name__=="__main__":
    main()
