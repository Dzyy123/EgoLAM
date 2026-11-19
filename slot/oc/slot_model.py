"""Stage 1: object-centric pretraining model (VideoSAUR backbone, OCLAL Sec. 3.1).

Pipeline for a chunk of F frames:

    video [B, F, 3, H, W]
      -> TimmExtractor (frozen DINO ViT): block-12 patch features `vit_block12`
         and attention keys `vit_block_keys12`           [B*F, N, D_e]
      -> two-layer MLP projection to slot_dim            [B*F, N, D_s]
      -> SlotAttention applied recurrently over time,
         slots initialized by `FixedInit` (per-slot mu/sigma, OCLAL)
      -> SlotMixerDecoder: per-frame reconstruction
         [B, F, N, D_e + N]  (features + cross-patch similarity logits)

Losses (identical to the VideoSAUR recipe, configs/videosaur/ytvis.yml):
    L_feat    = MSE(reconstruction[..., :D_e], block-12 features)      weight 1.0
    L_timesim = CE(reconstruction[..., D_e:] (frames 0..F-2),
                    FeatureTimeSimilarity(block-12 keys))              weight 0.1

Only the initializer, the projection MLP, the SlotAttention and the decoder are
trained; the DINO backbone stays frozen (OCLAL Stage 1 protocol).
"""

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# Official VideoSAUR modules, vendored verbatim (diff-verified identical)
# so that no pytorch_lightning dependency is pulled in.
from slot.vendor import nets as networks
from slot.vendor.decoders import SlotMixerDecoder
from slot.vendor.encoders import TimmExtractor
from slot.vendor.groupers import SlotAttention
from slot.vendor.m_utils import FeatureTimeSimilarity

from slot.oc.initializer import FixedInit


@dataclass
class ObjectSlotConfig:
    # Paper-aligned default: OCLAL uses VideoSAUR with a DINOv2 ViT-B/14 encoder
    # and upscales inputs to 518px (518 = 14 * 37). The cheaper VideoSAUR-release
    # option is model_name="vit_base_patch16_224_dino", img_size=224.
    model_name: str = "vit_base_patch14_dinov2.lvd142m"
    img_size: int = 518
    # Number of slots K and slot dimension D_s.
    # Paper Table 5: slot dimension 128 (ytvis release: 64). K is per-task
    # (the paper's probe figures use 3-8 slots depending on the environment).
    n_slots: int = 4
    slot_dim: int = 128
    # SlotAttention iterations (VideoSAUR: 3 on the first frame, 2 afterwards).
    n_iters_first_frame: int = 3
    n_iters: int = 2
    # Time-similarity loss hyperparameters. Paper Table 5: temperature 0.075
    # (VideoSAUR's ytvis release uses 0.25 -- we follow the OCLAL table).
    rec_weight: float = 1.0
    sim_weight: float = 0.1
    sim_temp: float = 0.075
    sim_threshold: float = 0.0
    # Use random timm weights (True) for smoke tests without downloads.
    pretrained_backbone: bool = True
    # Required opt-in when pretrained_backbone=False (see the check in __init__).
    allow_random_backbone: bool = False

    @property
    def feat_dim(self) -> int:
        return 768 if "base" in self.model_name else 384

    @property
    def n_patches(self) -> int:
        patch = 16 if "patch16" in self.model_name else 14
        return (self.img_size // patch) ** 2


class ObjectSlotEncoder(nn.Module):
    """VideoSAUR-style object-centric encoder with OCLAL fixed initialization."""

    def __init__(self, config: ObjectSlotConfig | None = None):
        super().__init__()
        self.config = config = config or ObjectSlotConfig()
        d_e, d_s = config.feat_dim, config.slot_dim
        n_patches = config.n_patches

        # A frozen *randomly initialised* backbone silently produces meaningless
        # slots (no error, just garbage). Refuse to build one by accident.
        if not config.pretrained_backbone and not config.allow_random_backbone:
            raise ValueError(
                "pretrained_backbone=False with frozen=True gives a random frozen backbone. "
                "Set allow_random_backbone=True only if you will load real weights afterwards."
            )
        # Frozen self-supervised backbone (DINO). `frozen=True` disables grads
        # and wraps the forward in no_grad, matching VideoSAUR.
        local_backbone = os.environ.get("EGOLAM_BACKBONE")
        self.backbone = TimmExtractor(
            model=config.model_name,
            pretrained=config.pretrained_backbone and not local_backbone,
            checkpoint_path=local_backbone,
            frozen=True,
            features=["vit_block12", "vit_block_keys12"],
        )
        # Input adapter to the grouper (VideoSAUR `two_layer_mlp` output transform,
        # which maps its `layer_norm` config onto `initial_layer_norm`).
        self.proj = networks.MLP(d_e, d_s, [2 * d_e], initial_layer_norm=True)
        # Fixed per-slot initialization (OCLAL Appendix C).
        self.initializer = FixedInit(config.n_slots, d_s)
        # Slot attention over projected patch features.
        self.grouper = SlotAttention(
            inp_dim=d_s,
            slot_dim=d_s,
            n_iters=config.n_iters,
            use_mlp=False,
        )
        # Frame-to-frame slot predictor (VideoSAUR `predictor`, ytvis config:
        # 1-block TransformerEncoder). The next frame's slots are initialized
        # from predictor(corrected_slots), not from the corrected slots directly.
        self.predictor = networks.TransformerEncoder(dim=d_s, n_blocks=1, n_heads=4)
        # Slot mixer decoder: reconstructs DINO features + similarity logits.
        self.decoder = SlotMixerDecoder(
            inp_dim=d_s,
            outp_dim=d_e + n_patches,
            embed_dim=d_s,
            n_patches=n_patches,
            allocator=networks.TransformerEncoder(dim=d_s, memory_dim=d_s, n_blocks=3, n_heads=4),
            renderer=networks.MLP(d_s, 1024, [1024, 1024], final_activation=True),
            renderer_dim=1024,
            use_layer_norms=True,
            pos_embed_mode="add",
        )

    def trainable_parameters(self):
        """Everything except the frozen DINO backbone (Stage 1 trains these only)."""
        return [
            p
            for n, p in self.named_parameters()
            if not n.startswith("backbone") and p.requires_grad
        ]

    def forward(
        self, video: torch.Tensor, compute_decoder: bool = True
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            video: [B, F, 3, H, W] float in [0, 1] (or normalized consistently
                with the timm transform used at training time).
            compute_decoder: set False for slot extraction only -- skips the
                (relatively expensive) decoder and omits reconstruction/masks.

        Returns dict with:
            slots:              [B, F, K, D_s]   per-frame object slots
            reconstruction:     [B, F, N, D_e+N] decoder output (feats + sim logits)
            decoder_masks:      [B, F, K, N]     decoder attention masks (per slot,
                                VideoSAUR convention [batch, slots, patches])
            backbone_features:  [B, F, N, D_e]   frozen block-12 features (targets)
            backbone_keys:      [B, F, N, D_e]   frozen block-12 keys (timesim target)

            (reconstruction / decoder_masks only present when compute_decoder=True)
        """
        assert video.ndim == 5, f"expected [B, F, 3, H, W], got {tuple(video.shape)}"
        B, F = video.shape[:2]
        cfg = self.config

        flat = video.flatten(0, 1)  # [B*F, 3, H, W]
        feats = self.backbone(flat)
        # TimmExtractor returns a dict when multiple features are requested.
        main_feats = feats["vit_block12"]  # [B*F, N, D_e]
        keys = feats["vit_block_keys12"]  # [B*F, N, D_e]

        proj = self.proj(main_feats).unflatten(0, (B, F))  # [B, F, N, D_s]

        slots, recons, masks = [], [], []
        state = self.initializer(B)  # [B, K, D_s]
        for t in range(F):
            n_iters = cfg.n_iters_first_frame if t == 0 else cfg.n_iters
            corrected = self.grouper(state, proj[:, t], n_iters=n_iters)["slots"]  # [B, K, D_s]
            if compute_decoder:
                dec = self.decoder(corrected)  # reconstruction [B, N, D_e+N], masks [B, K, N]
                recons.append(dec["reconstruction"])
                masks.append(dec["masks"])
            slots.append(corrected)
            # VideoSAUR recursion: next frame's slot initialization is the
            # *predicted* state; the next corrector call then refines it with
            # that frame's features.
            state = self.predictor(corrected)

        outputs = {
            "slots": torch.stack(slots, dim=1),  # [B, F, K, D_s]
            "backbone_features": main_feats.unflatten(0, (B, F)),  # [B, F, N, D_e]
            "backbone_keys": keys.unflatten(0, (B, F)),  # [B, F, N, D_e]
        }
        if compute_decoder:
            outputs["reconstruction"] = torch.stack(recons, dim=1)  # [B, F, N, D_e+N]
            outputs["decoder_masks"] = torch.stack(masks, dim=1)  # [B, F, K, N]
        return outputs

    def losses(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """VideoSAUR losses: feature reconstruction + temporal feature similarity."""
        cfg = self.config
        d_e = cfg.feat_dim

        rec = outputs["reconstruction"]  # [B, F, N, D_e+N]
        target_feats = outputs["backbone_features"].detach()  # [B, F, N, D_e]
        loss_feat = F.mse_loss(rec[..., :d_e], target_feats)

        # Similarity logits: predict, from frame t, the patch-similarity structure
        # of frame t+1. The last frame has no successor -> drop it (as in
        # VideoSAUR's `remove_last_n_frames: 1`).
        #   sim_logits: [B, F-1, P_src, P_dst] (classes = destination patches)
        sim_logits = rec[..., d_e:][:, :-1]
        timesim = FeatureTimeSimilarity(
            time_shift=1,
            temperature=cfg.sim_temp,
            threshold=cfg.sim_threshold,
            softmax=True,
        )
        target_sim = timesim(outputs["backbone_keys"].detach())  # [B, F-1, P_src, P_dst]
        # torch.cross_entropy expects (classes, positions) order with a
        # probabilistic target of the same shape (matches VideoSAUR's
        # CrossEntropyLoss wrapper, which transposes both tensors).
        loss_timesim = F.cross_entropy(
            sim_logits.transpose(-2, -1).flatten(0, 1),
            target_sim.transpose(-2, -1).flatten(0, 1),
        )

        return {
            "loss_feat": loss_feat,
            "loss_timesim": loss_timesim,
            "loss": cfg.rec_weight * loss_feat + cfg.sim_weight * loss_timesim,
        }
