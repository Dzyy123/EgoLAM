"""Slot-attention latent action model (SlotLAM), v2.

    frames (o_t, o_{t+k})  --frozen DINOv3-B/16-->  patch tokens [256, 768] (per-token LayerNorm)
        --> SlotEncoder: proj + slot attention (K slots; slots at t+k are initialised from slots at t through a
            predictor, so slot identity is preserved) + spatial-broadcast feature decoder (reconstruction loss)
        --> SlotLAM
              IDM : transformer over the 2K slot tokens; two learned HAND QUERIES (left / right) attend over each
                    frame's slots. Their attention, pushed through the slot masks, gives a predicted hand mask on the
                    patch grid that is supervised with projected hand-keypoint heat-maps (BCE).
                    z = MLP(q_{t+k} - q_t)  -- TEMPORAL DIFFERENCE ONLY (absolute slot summaries collapse z), LayerNorm-ed.
              FDM : predicts slots_{t+k} from slots_t + z token + nuisance token (camera egomotion n, train only; zeros
                    for a fixed camera). The prediction is DECODED to DINO features and compared with f_{t+k}, so
                    copying slots_t is not a solution and z has to carry the change.
              aux : motion head z -> 12-D hand action (masked); adversarial head z -> n through gradient reversal;
                    camera-invariance: z(o_t, warp(o_{t+k})) must equal z(o_t, o_{t+k}) for a random similarity warp.

Inference needs only the two images: latent(f_t, f_tk) -> z [B, latent_dim].
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class SlotLAMConfig:
    dino_path: str = ""
    image_size: int = 256
    feat_dim: int = 768
    n_slots: int = 6
    slot_dim: int = 128
    slot_iters: int = 3
    latent_dim: int = 32
    n_heads: int = 4
    depth: int = 2
    action_dim: int = 12
    nuisance_dim: int = 6
    n_hands: int = 2
    z_from_all_slots: bool = False   # if True, z is built from every slot's temporal difference


# ----------------------------------------------------------------------------- frozen frontend
class DinoV3Frontend(nn.Module):
    def __init__(self, path):
        super().__init__()
        from transformers import DINOv3ViTModel
        self.model = DINOv3ViTModel.from_pretrained(path).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        """x: [B, 3, H, W] in [0, 1] -> patch tokens [B, N, 768] (penultimate layer, CLS + registers dropped)."""
        x = (x - self.mean) / self.std
        out = self.model(pixel_values=x, output_hidden_states=True)
        tok = out.hidden_states[-2][:, 5:, :]
        # affine-free LayerNorm per token (LaWAM's norm_latents): raw DINO features have magnitudes ~20,
        # which would let the reconstruction MSE swamp every other loss term.
        return F.layer_norm(tok, (tok.shape[-1],))


def random_similarity_warp(img, max_shift=0.06, max_rot_deg=5.0, max_scale=0.04):
    """GPU similarity warp of a batch [B,3,H,W] (synthetic camera egomotion); returns warped images and params [B,4]."""
    b = img.shape[0]
    dev = img.device
    ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(max_rot_deg)
    sc = 1 + (torch.rand(b, device=dev) * 2 - 1) * max_scale
    tx = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2   # affine_grid works in [-1,1] coords -> x2
    ty = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)   # [B,2,3]
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection", align_corners=False), torch.stack([tx, ty, ang, sc], 1)


# ----------------------------------------------------------------------------- slot encoder
class SlotAttention(nn.Module):
    def __init__(self, dim, iters=3):
        super().__init__()
        self.iters, self.scale = iters, dim ** -0.5
        self.norm_in, self.norm_slot, self.norm_mlp = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, inputs, slots):
        inputs = self.norm_in(inputs)
        k, v = self.k(inputs), self.v(inputs)
        attn = None
        for _ in range(self.iters):
            prev = slots
            q = self.q(self.norm_slot(slots))
            logits = torch.einsum("bkd,bnd->bkn", q, k) * self.scale
            attn = logits.softmax(1)                                   # competition over slots
            w = attn / (attn.sum(-1, keepdim=True) + 1e-8)
            upd = torch.einsum("bkn,bnd->bkd", w, v)
            slots = self.gru(upd.reshape(-1, upd.shape[-1]), prev.reshape(-1, prev.shape[-1])).view_as(prev)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn


class SlotEncoder(nn.Module):
    def __init__(self, c: SlotLAMConfig):
        super().__init__()
        n = (c.image_size // 16) ** 2
        self.proj = nn.Sequential(nn.LayerNorm(c.feat_dim), nn.Linear(c.feat_dim, c.slot_dim), nn.GELU(), nn.Linear(c.slot_dim, c.slot_dim))
        self.pos = nn.Parameter(torch.randn(1, n, c.slot_dim) * 0.02)
        self.init_mu = nn.Parameter(torch.randn(1, c.n_slots, c.slot_dim) * 0.5)
        self.init_logsigma = nn.Parameter(torch.zeros(1, c.n_slots, c.slot_dim))
        self.sa = SlotAttention(c.slot_dim, c.slot_iters)
        self.predictor = nn.TransformerEncoderLayer(c.slot_dim, 4, 2 * c.slot_dim, dropout=0.0, batch_first=True, norm_first=True)
        # spatial-broadcast decoder: slot + position -> feature + alpha logit
        self.dec_pos = nn.Parameter(torch.randn(1, 1, n, c.slot_dim) * 0.02)
        self.dec = nn.Sequential(nn.Linear(c.slot_dim, 4 * c.slot_dim), nn.GELU(), nn.Linear(4 * c.slot_dim, 4 * c.slot_dim), nn.GELU(),
                                 nn.Linear(4 * c.slot_dim, c.feat_dim + 1))

    def initial_slots(self, b):
        mu = self.init_mu.expand(b, -1, -1)
        if not self.training:            # deterministic slots (and z) at extraction / evaluation time
            return mu
        return mu + self.init_logsigma.exp() * torch.randn_like(mu)

    def encode(self, feats, init_slots=None):
        x = self.proj(feats) + self.pos
        slots = self.initial_slots(feats.shape[0]) if init_slots is None else self.predictor(init_slots)
        return self.sa(x, slots)                                       # slots [B,K,D], attn [B,K,N]

    def decode(self, slots):
        h = self.dec(slots[:, :, None, :] + self.dec_pos)              # [B,K,N,768+1]
        feat, alpha = h[..., :-1], h[..., -1].softmax(1)               # alpha over slots
        return (alpha[..., None] * feat).sum(1), alpha                 # recon [B,N,768], masks [B,K,N]


# ----------------------------------------------------------------------------- LAM
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class SlotLAM(nn.Module):
    def __init__(self, c: SlotLAMConfig):
        super().__init__()
        self.c = c
        d = c.slot_dim
        self.encoder = SlotEncoder(c)
        self.time_emb = nn.Parameter(torch.randn(2, 1, 1, d) * 0.02)
        self.idm = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, c.n_heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True), c.depth)
        self.hand_query = nn.Parameter(torch.randn(1, c.n_hands, d) * 0.02)      # left, right
        self.hq, self.hk = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        # z from the temporal difference of the per-hand pooled representation only (see module docstring)
        zin = (c.n_slots if c.z_from_all_slots else c.n_hands) * d
        self.to_z = nn.Sequential(nn.Linear(zin, d), nn.GELU(), nn.Linear(d, c.latent_dim), nn.LayerNorm(c.latent_dim))
        self.z_in = nn.Linear(c.latent_dim, d)
        self.nuis_in = nn.Linear(c.nuisance_dim, d)
        self.nuis_missing = nn.Parameter(torch.zeros(1, 1, d))
        self.fdm = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, c.n_heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True), c.depth)
        self.fdm_out = nn.Linear(d, d)
        self.motion_head = nn.Sequential(nn.Linear(c.latent_dim, d), nn.GELU(), nn.Linear(d, c.action_dim))
        self.adv_head = nn.Sequential(nn.Linear(c.latent_dim, d), nn.GELU(), nn.Linear(d, c.nuisance_dim))

    # -- pieces
    def slots_for_pair(self, f_t, f_tk):
        s_t, a_t = self.encoder.encode(f_t)
        s_tk, a_tk = self.encoder.encode(f_tk, init_slots=s_t)
        return s_t, a_t, s_tk, a_tk

    def hand_attend(self, tokens):
        """hand queries over one frame's K tokens -> pooled [B,H,D], attention over slots [B,H,K]."""
        q = self.hq(self.hand_query).expand(tokens.shape[0], -1, -1)
        att = torch.einsum("bhd,bkd->bhk", q, self.hk(tokens)) * tokens.shape[-1] ** -0.5
        att = att.softmax(-1)
        return torch.einsum("bhk,bkd->bhd", att, tokens), att

    def encode_z(self, s_t, s_tk):
        k = s_t.shape[1]
        h = self.idm(torch.cat([s_t + self.time_emb[0], s_tk + self.time_emb[1]], 1))
        q_t, att_t = self.hand_attend(h[:, :k])
        q_tk, att_tk = self.hand_attend(h[:, k:])
        if self.c.z_from_all_slots:
            # every slot's own temporal difference feeds z; the hand attention is kept
            # only for the hand-mask loss, so z is no longer a hand-only bottleneck
            z = self.to_z((h[:, k:] - h[:, :k]).flatten(1))
        else:
            z = self.to_z((q_tk - q_t).flatten(1))
        return z, att_t, att_tk

    def predict(self, s_t, z, nuis, nuis_valid):
        b = s_t.shape[0]
        nt = torch.where(nuis_valid[:, None, None], self.nuis_in(torch.nan_to_num(nuis))[:, None], self.nuis_missing.expand(b, -1, -1))
        h = self.fdm(torch.cat([s_t + self.time_emb[0], self.z_in(z)[:, None], nt], 1))
        return s_t + self.fdm_out(h[:, :s_t.shape[1]])

    # -- training forward
    def forward(self, f_t, f_tk, nuis, nuis_valid, f_tk_warp=None, adv_lambda=1.0):
        s_t, a_t, s_tk, a_tk = self.slots_for_pair(f_t, f_tk)
        rec_t, _ = self.encoder.decode(s_t)
        rec_tk, _ = self.encoder.decode(s_tk)
        z, att_t, att_tk = self.encode_z(s_t, s_tk)
        pred_tk = self.predict(s_t, z, nuis, nuis_valid)
        rec_pred, _ = self.encoder.decode(pred_tk)
        out = {"z": z, "slots_t": s_t, "slots_tk": s_tk, "attn_t": a_t, "attn_tk": a_tk, "recon_t": rec_t, "recon_tk": rec_tk,
               "hand_att_t": att_t, "hand_att_tk": att_tk, "pred_tk": pred_tk, "recon_pred_tk": rec_pred,
               "motion": self.motion_head(z), "adv": self.adv_head(GradReverse.apply(z, adv_lambda))}
        if f_tk_warp is not None:                    # camera-invariance branch: same action, synthetic egomotion
            s_tk_w, _ = self.encoder.encode(f_tk_warp, init_slots=s_t)
            out["z_warp"] = self.encode_z(s_t, s_tk_w)[0]
        return out

    @torch.no_grad()
    def latent(self, f_t, f_tk):
        s_t, _, s_tk, _ = self.slots_for_pair(f_t, f_tk)
        return self.encode_z(s_t, s_tk)[0]


def hand_mask(hand_att, attn):
    """predicted per-hand mask on the patch grid: [B,H,K] x [B,K,N] -> [B,H,N] (probabilities in [0,1])."""
    return torch.einsum("bhk,bkn->bhn", hand_att, attn)


def losses(out, f_t, f_tk, heat_t, heat_tk, hand_valid, action, action_valid, nuis, nuis_valid, w):
    """heat_*: [B,H,N] in [0,1]; hand_valid: [B,H] bool."""
    L = {}
    zero = out["z"].sum() * 0
    L["recon"] = F.mse_loss(out["recon_t"], f_t) + F.mse_loss(out["recon_tk"], f_tk)
    # FDM through the decoder: the predicted future slots must explain the future DINO features
    L["fdm"] = F.mse_loss(out["recon_pred_tk"], f_tk) + 0.1 * F.mse_loss(out["pred_tk"], out["slots_tk"].detach())
    if hand_valid.any():
        tgt_t, tgt_tk = (heat_t > 0.3).float(), (heat_tk > 0.3).float()
        m_t = hand_mask(out["hand_att_t"], out["attn_t"]).clamp(1e-5, 1 - 1e-5)
        m_tk = hand_mask(out["hand_att_tk"], out["attn_tk"]).clamp(1e-5, 1 - 1e-5)
        pw = 8.0                                                    # hands cover a few % of the patches
        bce = lambda m, t: F.binary_cross_entropy(m, t, weight=1 + (pw - 1) * t)
        L["hand"] = 0.5 * (bce(m_t[hand_valid], tgt_t[hand_valid]) + bce(m_tk[hand_valid], tgt_tk[hand_valid]))
    else:
        L["hand"] = zero
    am = action_valid
    L["action"] = F.mse_loss(out["motion"][am], action[am]) if am.any() else zero
    nm = nuis_valid
    L["adv"] = F.mse_loss(out["adv"][nm], torch.nan_to_num(nuis)[nm]) if nm.any() else zero
    L["inv"] = F.mse_loss(out["z_warp"], out["z"].detach()) if "z_warp" in out else zero
    L["total"] = L["recon"] + w["fdm"] * L["fdm"] + w["hand"] * L["hand"] + w["action"] * L["action"] + w["adv"] * L["adv"] + w["inv"] * L["inv"]
    return L


def save_checkpoint(path, model, cfg, extra=None):
    torch.save({"config": asdict(cfg), "state_dict": model.state_dict(), **(extra or {})}, path)


def load_lam(path, dino_path=None, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = SlotLAMConfig(**ck["config"])
    if dino_path:
        cfg.dino_path = dino_path
    lam = SlotLAM(cfg)
    lam.load_state_dict(ck["state_dict"], strict=True)
    return lam.to(device).eval(), DinoV3Frontend(cfg.dino_path).to(device), cfg
