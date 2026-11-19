"""Load the released encoder and extract one continuous code per RGB pair."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from slot.oc.slot_model import ObjectSlotConfig, ObjectSlotEncoder
from .model import SlotLAMConfig
from .frozen_lam import FrozenSlotLAM

def load_models(directory, device="cuda"):
    root = Path(directory)
    cfg = json.loads((root / "config.json").read_text())
    oc = ObjectSlotEncoder(ObjectSlotConfig(**cfg["oc_slots"]["config"]))
    state = load_file(root / "oc_slots.safetensors")
    missing, extra = oc.load_state_dict(state, strict=False)
    expected = {name for name in oc.state_dict() if name.startswith("backbone.")}
    if set(missing) != expected or extra:
        raise ValueError("Slot checkpoint does not match the encoder")
    lam = FrozenSlotLAM(SlotLAMConfig(**cfg["frozen_lam"]["config"]))
    lam.load_state_dict(load_file(root / "frozen_lam.safetensors"), strict=True)
    return oc.requires_grad_(False).to(device).eval(), lam.requires_grad_(False).to(device).eval()

@torch.inference_mode()
def encode(oc, lam, frames):
    """Float RGB in [0,1], shaped B x 2 x 3 x 518 x 518."""
    if frames.ndim != 5 or tuple(frames.shape[1:]) != (2,3,518,518):
        raise ValueError("Expected B x 2 x 3 x 518 x 518")
    x = frames.to(next(lam.parameters()).device, dtype=torch.float32)
    mean = x.new_tensor([.485,.456,.406]).view(1,1,3,1,1)
    std = x.new_tensor([.229,.224,.225]).view(1,1,3,1,1)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
        slots = oc((x-mean)/std, compute_decoder=False)["slots"].float()
    return lam.encode_z(slots[:,0],slots[:,1])[0][:,None,:]
