"""Bind the existing EgoVerse Causal cohort and extract final z with native wrappers."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_SHA = {
    "train": "4da021741452d06349e4b5440bed42d74713a59d02bfff932d06b4770a0c4ce3",
    "val": "a9947720ec67b9401a6e30572cda5b48291a1d531af360dbe0dbca01820e4d1c",
    "unseen": "383337219610f898dede522f758462edc0c6eb62a97e1b6eab49e786cee2d13f",
}
METRIC_SHA = "b84dc37c53f54e05ce5f4275918cf276cfb01dd2dde11c75dcc58c18985b79be"


def sha(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def metrics(root):
    path = Path(root) / "causal/metrics.py"
    if sha(path) != METRIC_SHA:
        raise ValueError("Causal metric source differs from the recorded evaluation")
    spec = importlib.util.spec_from_file_location("aligned_causal_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args):
    metric = metrics(args.lary_root)
    data = Path(args.data)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    frames = {}
    groups = {}
    for split, digest in PAIR_SHA.items():
        path = data / f"pairs_{split}_stride15.csv"
        if sha(path) != digest:
            raise ValueError(f"Reference labels changed: {split}")
        frames[split] = metric.clean_pairs(pd.read_csv(path))
        frame = frames[split]
        if frame.duplicated(metric.KEY_COLS).any():
            raise ValueError("Duplicate source pair")
        groups[split] = set(frame.subset + "/" + frame.episode_id)
    overlap = groups["train"] & (groups["val"] | groups["unseen"])
    if groups["val"] & groups["unseen"]:
        raise ValueError("Evaluation episodes overlap")
    train = frames["train"]
    frames["train"] = train[~(train.subset + "/" + train.episode_id).isin(overlap)].reset_index(drop=True)
    counts = {s: len(df) for s, df in frames.items()}
    if counts != {"train": 82346, "val": 8904, "unseen": 9322}:
        raise ValueError(f"Unexpected prior cohort: {counts}")
    for split, frame in frames.items():
        frame["la_path"] = [f"{split}:{i}" for i in range(len(frame))]
        frame.to_csv(out / f"{split}.csv", index=False)
    comp = pd.read_csv(data / "pairs_val_stride30.csv")
    comp["la_path"] = "label_join_only"
    comp.to_csv(out / "comp.csv", index=False)
    ic_root = data.parent / "egoverse_ic"
    interventions = pd.read_csv(ic_root / "egoverse_ic_metadata_val.csv")
    if interventions.duplicated(["pair_id", "kind"]).any():
        raise ValueError("Duplicate intervention")
    if len(interventions)!=6000 or interventions.pair_id.nunique()!=1500:
        raise ValueError("Unexpected legacy intervention cohort")
    interventions.to_csv(out / "ic_legacy.csv", index=False)
    real = interventions[interventions.kind.eq("real")]
    matched = real.merge(frames["val"][metric.KEY_COLS], on=metric.KEY_COLS, validate="one_to_one")
    ic = interventions[interventions.pair_id.isin(matched.pair_id)].copy()
    if len(matched) != 605 or len(ic) != 2420:
        raise ValueError("Matched IC coverage differs")
    for _, group in ic.groupby("pair_id"):
        if set(group.kind) != {"identity", "camera", "photo", "real"} or group.src_img.nunique() != 1:
            raise ValueError("Invalid matched intervention group")
    ic.to_csv(out / "ic.csv", index=False)
    dump(out / "cohort.json", dict(counts=counts, matched_ic_pairs=len(matched),
         matched_ic_episodes=int(matched.groupby(["subset", "episode_id"]).ngroups),
         removed_train_episodes=sorted(overlap), labels_sha256=PAIR_SHA,
         source_root=str(data), native_metrics_sha256=METRIC_SHA,
         csv_sha256={s: sha(out / f"{s}.csv") for s in [*frames, "comp", "ic","ic_legacy"]}))


def extract(args):
    import torch
    import cv2
    from torch.utils.data import DataLoader
    from evaluation.pairs import Pairs
    from egolam.inference import load_models, encode
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    base = Path(args.cohort)
    lock = json.loads((base / "cohort.json").read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    recorded = json.loads(Path(args.weights).read_text())
    for name, proof in recorded.items():
        if sha(Path(os.environ["MODEL_DIR"]) / name) != proof["sha256"]:
            raise ValueError("Checkpoint differs from the LARY extension run")
    oc, lam = load_models(os.environ["MODEL_DIR"])
    source = Path(lock["source_root"])
    for split in ["val", "ic", "train", "unseen"]:
        csv_key = "ic_legacy" if split == "ic" else split
        if sha(base / f"{csv_key}.csv") != lock["csv_sha256"][csv_key]:
            raise ValueError("Cohort changed")
        frame = pd.read_csv(base / f"{csv_key}.csv")
        image_root = source.parent / "egoverse_ic" if split == "ic" else source
        for k in ("src_img", "tgt_img"):
            frame[k] = [str(image_root / p) for p in frame[k]]
        dataset = Pairs(frame)
        if args.smoke:
            dataset = torch.utils.data.Subset(dataset, range(min(8, len(dataset))))
        loader = DataLoader(dataset, batch_size=32, num_workers=4)
        name = f"smoke_{split}" if args.smoke else split
        target = out / f"{name}.npy"
        marker = out / f"{name}.json"
        identity = dict(model=args.model, weights=recorded, backbone_record_sha256="55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b",
                        cohort_sha256=sha(base / "cohort.json"),
                        rows=len(dataset), smoke=args.smoke, extraction_seed=42)
        if marker.exists():
            prior = json.loads(marker.read_text())
            if prior["identity"] != identity or prior["sha256"] != sha(target):
                raise ValueError(f"Unexpected existing cache: {target}")
            continue
        tmp = target.with_suffix(".partial.npy")
        count = 0
        array = None
        with torch.inference_mode():
            for batch in loader:
                if batch is None:
                    raise ValueError("Native extraction skipped an image pair")
                indices, images = batch
                if list(indices) != list(range(count, count + len(indices))):
                    raise ValueError("Missing or reordered image pairs")
                tokens = encode(oc, lam, images).cpu().numpy()
                z = np.asarray(tokens, np.float32).reshape(len(indices), -1)
                expected_width = 128
                if z.shape[1] != expected_width or not np.isfinite(z).all():
                    raise ValueError("Invalid native feature width or nonfinite features")
                if array is None:
                    array = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(dataset), z.shape[1]))
                array[count:count+len(z)] = z
                count += len(z)
        if count != len(dataset):
            raise ValueError("Incomplete feature cache")
        array.flush()
        tmp.replace(target)
        dump(marker, dict(identity=identity, sha256=sha(target), shape=list(array.shape)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for k in ("lary-root", "data", "output"):
        p.add_argument("--"+k, required=True)
    p = sub.add_parser("extract")
    for k in ("model", "cohort", "output", "weights"):
        p.add_argument("--"+k, required=True)
    p.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    (prepare if args.command == "prepare" else extract)(args)
