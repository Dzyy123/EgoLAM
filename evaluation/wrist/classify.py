"""Multi-GPU refit on existing LARY extension caches; outputs are separate from v1."""
import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from extension_v1 import train as composition
from extension_v1 import train_context as effect
from extension_v1.inventory import sha, write
from extension_v1.native import GRID, Probe, optimizer
from .distributed import initialize, updates, BalancedRows


def fit(args):
    initialize()
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world not in (4, 8):
        raise ValueError("The reference profile requires four or eight physical ranks")
    task = args.task
    engine = composition if task == "composition" else effect
    manifest, rows, features, index = engine.setup(args)
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    vocab = manifest.get("composition", {})
    nv, nn = (len(vocab.get(k, [])) for k in ("verbs", "nouns"))
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    epochs = 1 if args.smoke else 20
    steps = math.ceil(len(train) / 2048)
    models, optimizers, clocks = [], [], []
    grid = GRID[:1] if args.smoke else GRID
    labels = [(r["verb"],r["noun"]) if task=="composition" else r["label"] for r in train]
    balanced = BalancedRows(labels)
    for lr, wd in grid:
        probe = Probe(args.lary_root, features.shape[-1], task, nv, nn).cuda()
        model = DDP(probe, device_ids=[torch.cuda.current_device()], static_graph=True)
        opt, lr_schedule, wd_schedule = optimizer(model, lr, wd, steps * epochs, args.native_eval)
        models.append(model)
        optimizers.append(opt)
        clocks.append((lr_schedule, wd_schedule))
    best = [-1.] * len(grid)
    states = [None] * len(grid)
    selected_epochs = [None] * len(grid)
    history = []
    count_updates = 0
    for epoch in range(epochs):
        for model in models:
            model.train()
        for parts in updates(len(train), epoch, balanced=balanced):
            for schedulers in clocks:
                for schedule in schedulers:
                    schedule.step()
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            for ids in parts:
                batch = [train[i] for i in ids]
                ix = torch.tensor([index[r["sample_id"]] for r in batch], device="cuda")
                keys = ("verb", "noun") if task == "composition" else ("label",)
                targets = {key: torch.tensor([r[key] for r in batch], device="cuda") for key in keys}
                inp = features.index_select(0, ix)
                for model in models:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pred = model(inp)
                        loss = model.module.loss(pred, targets)
                    if not torch.isfinite(loss) or not torch.isfinite(pred).all():
                        raise FloatingPointError("Nonfinite head; abort the complete job")
                    (loss / len(parts)).backward()
            for opt in optimizers:
                opt.step()
            count_updates += 1
        if rank == 0:
            values = []
            for j, model in enumerate(models):
                result, _ = engine.evaluate(model.module, features, dev, index)
                value = result["all"]["accuracy"] if task == "composition" else result["family_macro_balanced_accuracy"]
                values.append(value)
                if value > best[j]:
                    best[j] = value
                    selected_epochs[j] = epoch + 1
                    states[j] = {k: v.detach().cpu().clone() for k, v in model.module.state_dict().items()}
            history.append({"epoch": epoch + 1, "development_scores": values, "optimizer_updates": count_updates})
            write(output / "progress.json", history[-1])
        dist.barrier()
    if rank == 0:
        selected = max(range(len(best)), key=best.__getitem__)
        record = dict(training_profile="lary-classify-8x256", physical_world_size=world,
                      reference_ranks=8, per_rank_batch=256, accumulation=8 // world,
                      nominal_global_batch=2048, updates_per_epoch=steps, updates=count_updates,
                      train_unique_sources=len(train), sampler_epoch_padded=8 * math.ceil(len(train) / 8),
                      seed=args.seed, sampler_seed=0, epochs=epochs, smoke=args.smoke,
                      sampler="upstream DistributedWeightedSampler, replacement",
                      sampler_label="verb-noun pair" if task=="composition" else "binary effect label",
                      data_protocol=manifest["protocol_version"], manifest_sha256=sha(args.manifest),
                      cache_manifest_sha256=sha(Path(args.cache) / "manifest.json"),
                      native_eval_sha256=sha(args.native_eval), development_best=best,
                      grid=grid, selected_grid=grid[selected], selected_epoch=selected_epochs[selected],
                      history=history, train_precision="bf16", evaluation_precision="fp32", label_smoothing=.1)
        ck = dict(state_dict=states[selected], input_dim=features.shape[-1], nverbs=nv,
                  nnouns=nn, seed=args.seed, manifest_sha256=record["manifest_sha256"])
        torch.save(ck, output / "probe.pt")
        record["checkpoint_sha256"] = sha(output / "probe.pt")
        write(output / "selection.json", record)
    dist.destroy_process_group()


def score(args):
    output = Path(args.output)
    sel = json.loads((output / "selection.json").read_text())
    if sel["smoke"] or sel["epochs"] != 20 or sel["nominal_global_batch"] != 2048:
        raise ValueError("Only completed reference-profile selections may be scored")
    if sel["seed"] != args.seed or sel["cache_manifest_sha256"] != sha(Path(args.cache) / "manifest.json"):
        raise ValueError("Changed seed or cache")
    path = output / "result.json"
    if path.exists():
        raise FileExistsError("This held-out result is already committed")
    engine = composition if args.task == "composition" else effect
    with tempfile.TemporaryDirectory(dir=output) as temp:
        staging = Path(temp)
        for item in ("probe.pt","selection.json"):
            (staging/item).symlink_to((output/item).resolve())
        engine.score(argparse.Namespace(**{**vars(args),"output":temp}))
        result = json.loads((staging/"result.json").read_text())
        result.update(training_profile=sel["training_profile"], global_batch=2048,
                      physical_world_size=sel["physical_world_size"])
        write(staging/"result.json", result)
        os.replace(staging/"predictions.npz",output/"predictions.npz")
        os.replace(staging/"result.json",path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "score"])
    parser.add_argument("--task", choices=["composition", "effect"], required=True)
    for name in ("lary-root", "manifest", "cache", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--native-eval")
    parser.add_argument("--validation")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    (fit if args.mode == "train" else score)(args)
