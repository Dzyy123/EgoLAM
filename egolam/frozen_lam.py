"""Stage-3: SlotLAM's latent-action model on FROZEN, precomputed object slots.

Same z machinery as `slotlam.model.SlotLAM` (IDM -> two hand queries -> temporal
difference -> LayerNorm MLP), but the slot encoder is frozen: s_t / s_tk come from the
OC checkpoint through a lookup table.

Losses kept identical in form to the baseline wherever the data exists:

  hand    BCE(hand_mask(att, masks), heat)   masks cached at 16x16, same pos_weight
  action  motion_head(z) vs a
  adv     adv_head(GRL(z)) vs n
  inv     MSE(z_warp, z)                      camera-invariance, RESTORED:
          z_warp = encode_z(s_t, OCencode(warp(img_tk))), i.e. one extra frozen
          encoder pass on a warped frame -- this is why the cache no longer saves
          the encoder cost, and it is deliberate.
  fdm     MSE(pred_tk, s_tk) in SLOT space (OC-LAL's L_dyn). The baseline's second
          half (decode to DINOv3 features) needs the decoder, which frozen slots do
          not have; `recon` is dropped for the same reason (it is constant w.r.t.
          every trainable parameter once the slots and decoder are frozen).
"""
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn

from egolam.model import GradReverse, SlotLAM, SlotLAMConfig, hand_mask  # noqa: F401


class FrozenSlotLAM(SlotLAM):
    """SlotLAM without its slot encoder; slots are supplied by the caller."""

    def __init__(self, c: SlotLAMConfig):
        super().__init__(c)
        del self.encoder                       # never used, never saved

    def forward(self, s_t, s_tk, nuis, nuis_valid, adv_lambda=1.0, s_tk_warp=None):
        z, att_t, att_tk = self.encode_z(s_t, s_tk)
        pred_tk = self.predict(s_t, z, nuis, nuis_valid)
        out = {"z": z, "slots_t": s_t, "slots_tk": s_tk,
               "hand_att_t": att_t, "hand_att_tk": att_tk, "pred_tk": pred_tk,
               "motion": self.motion_head(z),
               "adv": self.adv_head(GradReverse.apply(z, adv_lambda))}
        if s_tk_warp is not None:
            # camera invariance: same s_t, warped next frame -> z must not move
            out["z_warp"] = self.encode_z(s_t, s_tk_warp)[0]
        return out


def frozen_losses(out, s_tk, masks_t, masks_tk, heat_t, heat_tk, hand_valid,
                  action, action_valid, nuis, nuis_valid, w, inv_mask=None):
    L = {}
    zero = out["z"].sum() * 0
    # L_dyn, same split as the baseline: feature space (the FROZEN decoder applied to
    # the PREDICTED slots -- still differentiable, because pred_tk is trainable) plus
    # 0.1 x slot space.
    if "recon_pred_tk" in out:
        L["fdm"] = (F.mse_loss(out["recon_pred_tk"].float(), out["feat_tk"].float())
                    + 0.1 * F.mse_loss(out["pred_tk"], s_tk))
    else:
        L["fdm"] = F.mse_loss(out["pred_tk"], s_tk)
    if hand_valid.any():
        tgt_t, tgt_tk = (heat_t > 0.3).float(), (heat_tk > 0.3).float()
        m_t = hand_mask(out["hand_att_t"], masks_t).clamp(1e-5, 1 - 1e-5)
        m_tk = hand_mask(out["hand_att_tk"], masks_tk).clamp(1e-5, 1 - 1e-5)
        pw = 8.0
        bce = lambda m, t: F.binary_cross_entropy(m, t, weight=1 + (pw - 1) * t)  # noqa: E731
        L["hand"] = 0.5 * (bce(m_t[hand_valid], tgt_t[hand_valid]) + bce(m_tk[hand_valid], tgt_tk[hand_valid]))
    else:
        L["hand"] = zero
    am = action_valid
    L["action"] = F.mse_loss(out["motion"][am], action[am]) if am.any() else out["motion"].sum()*0
    nm = nuis_valid
    L["adv"] = F.mse_loss(out["adv"][nm], torch.nan_to_num(nuis)[nm]) if nm.any() else out["adv"].sum()*0
    # camera invariance -- same row layout as the baseline: MSE(z_warp, z.detach())
    if "z_warp" in out:
        z, zw = out["z"], out["z_warp"]
        if inv_mask is not None:
            z, zw = z[inv_mask], zw[inv_mask]
        L["inv"] = F.mse_loss(zw, z.detach()) if len(z) else zero
    else:
        L["inv"] = zero
    L["total"] = (w["fdm"] * L["fdm"] + w["hand"] * L["hand"] + w["action"] * L["action"]
                  + w["adv"] * L["adv"] + w["inv"] * L["inv"])
    return L


def save_checkpoint(path, model, cfg, extra=None):
    torch.save({"config": asdict(cfg), "state_dict": model.state_dict(), **(extra or {})}, path)

