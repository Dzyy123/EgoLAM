import re
from typing import Any, Dict, List, Optional, Union

import einops
import timm
import torch
import torchvision
from torch import nn

from slot.vendor import m_utils as utils
from slot.vendor.shim import config_as_kwargs, make_build_fn


@make_build_fn(__name__, "encoder")
def build(config, name: str):
    if name == "FrameEncoder":
        pos_embed = None
        if config.get("pos_embed"):
            pos_embed = utils.build_module(config.pos_embed)

        output_transform = None
        if config.get("output_transform"):
            output_transform = utils.build_module(config.output_transform)

        return FrameEncoder(
            backbone=utils.build_module(config.backbone, default_group="encoders"),
            pos_embed=pos_embed,
            output_transform=output_transform,
            **config_as_kwargs(config, ("backbone", "pos_embed", "output_transform")),
        )
    else:
        return None


class FrameEncoder(nn.Module):
    """Module reducing image to set of features."""

    def __init__(
        self,
        backbone: nn.Module,
        pos_embed: Optional[nn.Module] = None,
        output_transform: Optional[nn.Module] = None,
        spatial_flatten: bool = False,
        main_features_key: str = "vit_block12",
    ):
        super().__init__()
        self.backbone = backbone
        self.pos_embed = pos_embed
        self.output_transform = output_transform
        self.spatial_flatten = spatial_flatten
        self.main_features_key = main_features_key

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # images: batch x n_channels x height x width
        backbone_features = self.backbone(images)
        if isinstance(backbone_features, dict):
            features = backbone_features[self.main_features_key].clone()
        else:
            features = backbone_features.clone()

        if self.pos_embed:
            features = self.pos_embed(features)

        if self.spatial_flatten:
            features = einops.rearrange(features, "b c h w -> b (h w) c")
        if self.output_transform:
            features = self.output_transform(features)

        assert (
            features.ndim == 3
        ), f"Expect output shape (batch, tokens, dims), but got {features.shape}"
        if isinstance(backbone_features, dict):
            for k, backbone_feature in backbone_features.items():
                if self.spatial_flatten:
                    backbone_features[k] = einops.rearrange(backbone_feature, "b c h w -> b (h w) c")
                assert (
                    backbone_feature.ndim == 3
                ), f"Expect output shape (batch, tokens, dims), but got {backbone_feature.shape}"
            main_backbone_features = backbone_features[self.main_features_key]

            return {
                "features": features,
                "backbone_features": main_backbone_features,
                **backbone_features,
            }
        else:
            if self.spatial_flatten:
                backbone_features = einops.rearrange(backbone_features, "b c h w -> b (h w) c")
            assert (
                backbone_features.ndim == 3
            ), f"Expect output shape (batch, tokens, dims), but got {backbone_features.shape}"

            return {
                "features": features,
                "backbone_features": backbone_features,
            }


def _with_fx_trace_fixes(fn, *a, **kw):
    """Trace a timm model with torchvision's fx extractor, tolerating timm>=1.0 internals."""
    import timm.layers.attention as _ta
    import torch as _torch

    orig_attn = _ta.Attention.forward
    orig_assert = _torch._assert

    def attn_fwd(self, x, attn_mask=None, is_causal=False, **k):
        return orig_attn(self, x, attn_mask=attn_mask, is_causal=False, **k)

    def assert_noop(condition, message="", *aa, **kk):
        return None

    _ta.Attention.forward = attn_fwd
    _torch._assert = assert_noop
    try:
        return fn(*a, **kw)
    finally:
        _ta.Attention.forward = orig_attn
        _torch._assert = orig_assert


class _HookExtractor(nn.Module):
    """Run the full model and collect the outputs of named submodules as a dict."""

    def __init__(self, model, names_to_aliases):
        super().__init__()
        self.model = model
        submodules = dict(model.named_modules())
        self._handles = []
        self._buf = {}
        for name, alias in names_to_aliases:
            if name not in submodules:
                raise ValueError(f"node {name!r} not found in model")
            self._handles.append(submodules[name].register_forward_hook(self._make_hook(alias)))
        if not self._handles:
            raise ValueError("no feature nodes requested")

    def _make_hook(self, alias):
        def hook(_module, _inp, out):
            self._buf[alias] = out
        return hook

    def forward(self, x):
        self._buf = {}
        self.model(x)
        return dict(self._buf)

    def __del__(self):
        for h in getattr(self, "_handles", []):
            h.remove()


def _alias_and_node(name):
    """Resolve a user-facing feature alias (vit_blockN / vit_block_keysN) to a module path."""
    m = re.match(r"vit_block_keys(\d+)$", name)
    if m:
        return f"blocks.{int(m.group(1)) - 1}.attn.qkv", name
    m = re.match(r"vit_block(\d+)$", name)
    if m:
        return f"blocks.{int(m.group(1)) - 1}", name
    return name, name


class TimmExtractor(nn.Module):
    """Feature extractor utilizing models from timm library."""

    # Convenience aliases for feature keys
    FEATURE_ALIASES = {
        **{f"resnet_block{i}": f"layer{i}" for i in range(1, 5)},
        **{f"vit_block{i + 1}": f"blocks.{i}" for i in range(12)},
        **{f"vit_block_values{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_queries{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_keys{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        "vit_output": "norm",
    }
    FEATURE_MAPPING = {
        **{f"layer{i}": f"resnet_block{i}" for i in range(1, 5)},
        **{f"blocks.{i}": f"vit_block{i + 1}" for i in range(12)},
        **{f"blocks.{i}.attn.qkv": f"vit_block_keys{i + 1}" for i in range(12)},
        "norm": "vit_output",
    }

    def __init__(
        self,
        model: str,
        pretrained: bool = False,
        frozen: bool = False,
        features: Optional[Union[str, List[str]]] = None,
        checkpoint_path: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        model_name = model
        self.frozen = frozen
        self.features = [features] if isinstance(features, str) else features
        self.is_vit = model_name.startswith("vit")

        model = TimmExtractor._create_model(model_name, pretrained, checkpoint_path, model_kwargs)

        if self.features is not None:
            mapping = []
            for name in self.features:
                if name in TimmExtractor.FEATURE_ALIASES:
                    name = TimmExtractor.FEATURE_ALIASES[name]
                node, alias = _alias_and_node(name)
                mapping.append((node, alias))
            model = _HookExtractor(model, mapping)

        self.model = model

        if self.frozen:
            self.requires_grad_(False)

    @staticmethod
    def _create_model(
        model_name: str,
        pretrained: bool,
        checkpoint_path: Optional[str],
        model_kwargs: Optional[Dict[str, Any]],
        trials: int = 0,
    ) -> nn.Module:
        if model_kwargs is None:
            model_kwargs = {}

        try:
            model = timm.create_model(
                model_name, pretrained=pretrained, checkpoint_path=checkpoint_path, **model_kwargs
            )
        except (FileExistsError, FileNotFoundError):
            # Timm uses Hugginface hub for loading the files, which does some symlinking in the
            # background when loading the checkpoint. When multiple concurrent jobs attempt to
            # load the checkpoint, this can create conflicts, because the symlink is first removed,
            # then created again by each job. We attempt to catch the resulting errors here, and
            # retry creating the model, up to 3 times.
            if trials == 2:
                raise
            else:
                model = None

        if model is None:
            model = TimmExtractor._create_model(
                model_name, pretrained, checkpoint_path, model_kwargs, trials=trials + 1
            )

        return model

    def forward(self, inp):
        if self.frozen:
            with torch.no_grad():
                outputs = self.model(inp)
        else:
            outputs = self.model(inp)

        if self.features is not None:
            if self.is_vit:
                outputs = {k: v[:, 1:] for k, v in outputs.items()}  # Remove CLS token
            outputs = {self.FEATURE_MAPPING[key]: value for key, value in outputs.items()}
            for name in self.features:
                if ("keys" in name) or ("queries" in name) or ("values" in name):
                    feature_name = name.replace("queries", "keys").replace("values", "keys")
                    B, N, C = outputs[feature_name].shape
                    qkv = outputs[feature_name].reshape(
                        B, N, 3, C // 3
                    )  # outp has shape B, N, 3 * H * (C // H)
                    q, k, v = qkv.unbind(2)
                    if "keys" in name:
                        outputs[name] = k
                    elif "queries" in name:
                        outputs[name] = q
                    elif "values" in name:
                        outputs[name] = v
                    else:
                        raise ValueError(f"Unknown feature name {name}.")

            if len(outputs) == 1:
                # Unpack single output for now
                return next(iter(outputs.values()))
            else:
                return outputs
        else:
            return outputs
