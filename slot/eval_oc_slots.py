"""Load a slot-training checkpoint, including its frozen visual front end."""
import torch
from .oc.slot_model import ObjectSlotConfig, ObjectSlotEncoder

def load_model(path, device="cuda"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = ObjectSlotConfig(**checkpoint["config"])
    model = ObjectSlotEncoder(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), config
