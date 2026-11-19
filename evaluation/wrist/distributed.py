"""Map a reference DDP sampler and its optimizer updates onto fewer physical GPUs."""
import itertools
import os
import hashlib
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, DistributedSampler


def initialize(backend="nccl"):
    if backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend)


class BalancedRows:
    """The upstream sampler only needs the dataset size and its per-row weights."""
    def __init__(self, labels):
        counts = Counter(labels)
        self.sample_weights = torch.tensor([1. / (len(counts)*counts[k]) for k in labels],
                                           dtype=torch.float64)

    def __len__(self):
        return len(self.sample_weights)


def updates(length, epoch, reference_ranks=8, batch=256, balanced=None):
    """One tuple per optimizer step. All ref-ranks have equal tail-batch lengths."""
    physical = dist.get_world_size()
    if length <= 0 or reference_ranks % physical:
        raise ValueError("Need nonempty training data and an integral virtual-rank mapping")
    dataset = range(length) if balanced is None else balanced
    sampler_class = DistributedSampler
    if balanced is not None:
        from classification.src.datasets.utils import weighted_sampler
        raw = Path(weighted_sampler.__file__).read_bytes()
        blob = hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
        if blob != "f68dd9e70f722fe26119c7b9fc3dd988cc81c7ef":
            raise ValueError("Native weighted sampler differs from the upstream reference")
        sampler_class = weighted_sampler.DistributedWeightedSampler
    streams = []
    for rank in range(dist.get_rank(), reference_ranks, physical):
        sampler = sampler_class(dataset, num_replicas=reference_ranks, rank=rank, seed=0,
                                shuffle=True, drop_last=False)
        sampler.set_epoch(epoch)
        streams.append(BatchSampler(sampler, batch_size=batch, drop_last=False))
    return zip(*streams, strict=True)


def smoke_equivalence():
    """Real four-rank all-reduce against a serial full-batch CPU reference."""
    from torch.nn.parallel import DistributedDataParallel as DDP
    initialize("gloo")
    torch.set_num_threads(1)
    torch.manual_seed(23)
    model = DDP(torch.nn.Linear(3, 1).double())
    reference = torch.nn.Linear(3, 1).double()
    reference.load_state_dict(model.module.state_dict())
    opt = torch.optim.SGD(model.parameters(), lr=.01)
    baseline_opt = torch.optim.SGD(reference.parameters(), lr=.01)
    x = torch.randn(2059, 3, dtype=torch.float64)
    y = torch.randn(2059, 1, dtype=torch.float64)
    sizes = []
    for parts in updates(len(x), 0):
        indexes = list(itertools.chain.from_iterable(parts))
        groups = [None] * dist.get_world_size()
        dist.all_gather_object(groups, indexes)
        combined = list(itertools.chain.from_iterable(groups))
        opt.zero_grad()
        baseline_opt.zero_grad()
        for ids in parts:
            loss = (model(x[ids]) - y[ids]).square().mean()
            (loss / len(parts)).backward()
        (reference(x[combined]) - y[combined]).square().mean().backward()
        opt.step()
        baseline_opt.step()
        for actual, wanted in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, wanted, rtol=1e-12, atol=1e-12)
        sizes.append(len(combined))
    assert sizes == [2048, 16]  # DistributedSampler pads 2059 to 2064.
    import numpy as np
    dataset = BalancedRows([i % 3 if i<200 else 3 for i in range(len(x))])
    probabilities = dataset.sample_weights.numpy()
    expected = np.random.default_rng(3).choice(len(x), size=2064, replace=True,
                                                p=probabilities/probabilities.sum()).tolist()
    for step, parts in enumerate(updates(len(x), 3, balanced=dataset)):
        local = list(itertools.chain.from_iterable(parts))
        groups = [None]*dist.get_world_size()
        dist.all_gather_object(groups, local)
        assert sorted(itertools.chain.from_iterable(groups)) == sorted(expected[step*2048:(step+1)*2048])
    from .causal_fit import regression_updates
    generator = torch.Generator().manual_seed(42)
    permutation = torch.randperm(len(x), generator=generator).tolist()
    expected = (permutation * 2)[:4096]
    reg_sizes = []
    for step, parts in enumerate(regression_updates(len(x), 0)):
        local = list(itertools.chain.from_iterable(parts))
        groups = [None] * dist.get_world_size()
        dist.all_gather_object(groups, local)
        combined = list(itertools.chain.from_iterable(groups))
        assert sorted(combined) == sorted(expected[step*2048:(step+1)*2048])
        reg_sizes.append(len(combined))
    assert reg_sizes == [2048, 2048]  # Accelerate pads complete batches.
    if dist.get_rank() == 0:
        print("CPU parity passed: four ranks, two microbatches, updates of 2048 and 16")
        print("Official weighted sampler passed: inverse frequency and replacement, seed + epoch")
        print("Regression sampler passed: same 2048-sample batches including the padded tail")
    dist.destroy_process_group()


if __name__ == "__main__":
    smoke_equivalence()
