"""Distributed fit of the pinned official ActionExpertMLP on one Causal target."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from .distributed import initialize


def regression_updates(length, epoch):
    """Use the same Accelerate batch sharder as the reference regression loader."""
    from accelerate.data_loader import BatchSamplerShard
    from torch.utils.data import BatchSampler, RandomSampler

    shards = []
    for virtual_rank in range(dist.get_rank(), 8, dist.get_world_size()):
        generator = torch.Generator().manual_seed(42 + epoch)
        sampler = RandomSampler(range(length), generator=generator)
        batches = BatchSampler(sampler, batch_size=256, drop_last=False)
        shards.append(BatchSamplerShard(batches, num_processes=8, process_index=virtual_rank,
                                        split_batches=False, even_batches=True))
    return zip(*shards, strict=True)


def official_model(source, input_dim, output_dim, hidden=4096):
    source = Path(source)
    raw = source.read_bytes()
    # Our text snapshot has one additional trailing LF. The canonical bytes
    # reproduce the full Git blob ID advertised by the pinned upstream tree.
    raw = raw.rstrip(b"\n") + b"\n"
    identity = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if identity != "9ece94c6027ce5da90c3932c8a4167a59c3254ad":
        raise ValueError("Regression source differs from the pinned upstream blob")
    names = {"MLPResNetBlock", "MLPResNet", "ActionExpertMLP"}
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("Missing source classes")
    ns = {"torch": torch, "nn": nn, "F": F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
    return ns["ActionExpertMLP"](input_dim, action_dim=output_dim, action_steps=1,
                                 hidden_dim=hidden, num_blocks=2)


def fit(a):
    initialize()
    rank = dist.get_rank()
    if dist.get_world_size() not in (4, 8):
        raise ValueError("Use four or eight ranks")
    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    work = Path(a.work)
    x, y = [torch.from_numpy(np.load(work / (key + ".npy"), mmap_mode="c")).cuda() for key in ("x", "y")]
    fit_rows = torch.from_numpy(np.load(work / "fit.npy")).cuda()
    hold_rows = torch.from_numpy(np.load(work / "hold.npy")).cuda()
    model = DDP(official_model(a.source, x.shape[1], y.shape[1]).cuda(),
                device_ids=[torch.cuda.current_device()])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epochs = 1 if a.smoke else 20
    steps = math.ceil(len(fit_rows) / 2048)
    # Upstream builds warmup from the unsharded loader. AcceleratedScheduler
    # advances it eight times after each eight-rank optimizer update.
    warmup = int(.1 * epochs * math.ceil(len(fit_rows) / 256))
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1., 8*s / warmup) if warmup else 1.)
    best = float("inf")
    for epoch in range(epochs):
        model.train()
        for parts in regression_updates(len(fit_rows), epoch):
            opt.zero_grad(set_to_none=True)
            for ids in parts:
                ix = fit_rows[torch.as_tensor(ids, device="cuda")]
                loss = model.module.loss(model(x[ix]), y[ix])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite Causal readout")
                (loss / len(parts)).backward()
            opt.step()
            schedule.step()
        if rank == 0:
            model.module.eval()
            with torch.inference_mode():
                sq = sum(float((model.module(x[ix])-y[ix]).square().sum())
                         for ix in hold_rows.split(256))
            mse = sq / (len(hold_rows) * y.shape[1])
            if mse < best:
                best = mse
                torch.save(model.module.state_dict(), work / "best.pt")
                (work / "selection.json").write_text(json.dumps(dict(epoch=epoch+1, hold_mse=mse,
                    epochs=epochs, fit_rows=len(fit_rows), hold_rows=len(hold_rows), seed=a.seed,
                    optimizer_updates_per_epoch=steps, global_batch=2048, per_rank_batch=256,
                    reference_warmup_steps=warmup, physical_ranks=dist.get_world_size(), smoke=a.smoke)))
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    fit(p.parse_args())
