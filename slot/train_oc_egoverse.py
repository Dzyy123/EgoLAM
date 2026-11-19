"""Stage-1 OC pretraining on EgoVerse — thin driver around the OCLAL reference model.

Model/losses are the reference implementation (`slot/oc/slot_model.py`, unmodified
except import paths). This file only supplies: data, optimizer schedule, gradient
clipping (missing in the reference driver), DDP and checkpointing.

Reference hyper-parameters (OCLAL Table 5, as recorded in the reference code):
  chunk 3 frames, lr 3e-4 @ batch 256, 100k steps, warmup 2500, exp decay 0.1.
  We scale lr linearly with the global batch (videosaur convention):
      lr = 3e-4 * global_batch / 256
  and clip gradients at 0.05 (VideoSAUR's `gradient_clip_val`), which the
  reference Stage-1 driver omits.
"""
import argparse, json, os, sys, time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slot.data import FrameChunks                      # noqa: E402
from slot.oc.slot_model import ObjectSlotConfig, ObjectSlotEncoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--n-slots", type=int, default=4)
    ap.add_argument("--slot-dim", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=3)
    ap.add_argument("--sim-temp", type=float, default=0.25)
    ap.add_argument("--sim-weight", type=float, default=0.1,
                    help="official ytvis.yml SIM_WEIGHT = 0.1")
    ap.add_argument("--rec-weight", type=float, default=1.0,
                    help="official ytvis.yml loss_featrec = 1.0")
    ap.add_argument("--batch-size", type=int, default=8, help="per GPU chunks")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--lr", type=float, default=None, help="default 3e-4 * gbatch/256")
    ap.add_argument("--warmup", type=int, default=2500)
    ap.add_argument("--clip", type=float, default=0.05)
    ap.add_argument("--decay-rate", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=2500)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="resume from the newest checkpoint in --out")
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev, main_proc = torch.device("cuda", rank), rank == 0
    os.makedirs(a.out, exist_ok=True)
    gbatch = a.batch_size * world
    lr = a.lr if a.lr is not None else 3e-4 * gbatch / 256.0

    def log(m):
        if main_proc:
            print(m, flush=True)

    cfg = ObjectSlotConfig(model_name=a.model_name, img_size=a.img_size, n_slots=a.n_slots,
                           slot_dim=a.slot_dim, sim_weight=a.sim_weight, rec_weight=a.rec_weight,
                           sim_temp=a.sim_temp,
                           pretrained_backbone=True)
    torch.manual_seed(1000 + a.seed)
    model = ObjectSlotEncoder(cfg).to(dev)
    model.train()
    model.backbone.eval()   # frozen=True does not call eval(); keep dropout off
    start_step = 0
    if a.resume:
        import glob
        cks = sorted(glob.glob(os.path.join(a.out, "oc_slots_k*_[0-9]*.pt")))
        if cks:
            d = torch.load(cks[-1], map_location="cpu", weights_only=False)
            model.load_state_dict(d["state_dict"])
            start_step = int(d.get("step", 0))
            log(f"[oc] resumed from {cks[-1]} at step {start_step}")
    opt = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (min(1.0, (s + 1) / max(1, a.warmup)) * a.decay_rate ** (s / max(1, a.steps))))

    sched.last_epoch = start_step          # continue the LR schedule, not restart warmup
    ds = FrameChunks(a.frames_root, a.labels_root, image_size=a.img_size, n_frames=a.chunk,
                     n_samples=a.steps * a.batch_size + 2000,
                     seed=a.seed * 100 + rank + (start_step if start_step else 0))
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                                     drop_last=True, pin_memory=True, persistent_workers=a.num_workers > 0,
                                     prefetch_factor=2 if a.num_workers > 0 else None)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
    n_tr = sum(p.numel() for p in model.trainable_parameters())
    log(f"[oc] K={a.n_slots} d={a.slot_dim} patches={cfg.n_patches} img={a.img_size} chunk={a.chunk} "
        f"lr={lr:.2e} clip={a.clip} temp={cfg.sim_temp} w_sim={cfg.sim_weight} "
        f"gbatch={gbatch} trainable={n_tr/1e6:.2f}M")

    _MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 1, 3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 1, 3, 1, 1)
    step, t0 = start_step, time.time()
    fh = open(os.path.join(a.out, "log.jsonl"), "a") if main_proc else None
    for frames in dl:
        frames = frames.to(dev, non_blocking=True).float().div_(255.0)   # [B,F,3,H,W]
        # ImageNet normalisation: TimmExtractor does NOT normalise internally, the
        # reference dataset does it (egoverse_native.IMAGENET_MEAN/STD).
        frames = (frames - _MEAN) / _STD
        if step == 0 and main_proc:
            log(f"[oc] input {tuple(frames.shape)}")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = ddp(frames)
        # losses (CE over 1369 classes) in fp32 for stability
        out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
        losses = ddp.module.losses(out)
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(ddp.module.trainable_parameters(), a.clip)
        opt.step(); sched.step(); step += 1
        if step % a.log_every == 0:
            r = {"step": step, "feat": round(float(losses["loss_feat"]), 4),
                 "sim": round(float(losses["loss_timesim"]), 4),
                 "loss": round(float(losses["loss"]), 4),
                 "lr": sched.get_last_lr()[0], "s_per_step": round((time.time() - t0) / step, 3),
                 "mem_gb": round(torch.cuda.max_memory_allocated(dev) / 1e9, 2)}
            if main_proc:
                fh.write(json.dumps(r) + "\n"); fh.flush()
                if step % (a.log_every * 4) == 0:
                    log(f"step {step} feat {r['feat']:.4f} sim {r['sim']:.4f} "
                        f"{r['s_per_step']}s/it {r['mem_gb']}GB")
        if step % a.save_every == 0 or step >= a.steps:
            if main_proc:
                torch.save({"state_dict": model.state_dict(), "config": vars(cfg), "args": vars(a),
                            "step": step}, os.path.join(a.out, f"oc_slots_k{a.n_slots}_{step:06d}.pt"))
                log(f"saved step {step}")
        if step >= a.steps:
            break
    if main_proc:
        fh.write("OC_DONE\n"); fh.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()