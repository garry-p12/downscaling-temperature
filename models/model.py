"""DownscalingModel: Swin backbone + temperature head, and the arch dispatcher.

Predictions are in normalized space (the trainer denormalizes for metrics).

Scope is temperature only. The precipitation heads that used to sit alongside
the temperature head were never trained and there is no ``ppt`` channel in the
dataset, so they were removed rather than left as permanently-disabled stubs;
git history has them if precip is ever revived.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_swin import SwinUNet
from .temp_head import TempHead

# State-dict prefixes written by checkpoints from before the precip heads were
# removed. They are dropped on load so those checkpoints — and the results
# tables built from them — stay usable without a migration step.
_LEGACY_PREFIXES = ("precip_stage1.", "precip_stage2.")


class DownscalingModel(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        enc = model_cfg["encoder"]
        self.patch_size = model_cfg["patch_size"]
        self.backbone = SwinUNet(
            in_channels=model_cfg["in_channels"],
            patch_size=self.patch_size,
            embed_dim=enc["embed_dim"],
            depths=enc["depths"],
            num_heads=enc["num_heads"],
            window_size=enc["window_size"],
            mlp_ratio=enc["mlp_ratio"],
            drop_path=enc["drop_path"],
        )
        dim = enc["embed_dim"]
        self.temp_head = TempHead(dim, self.patch_size,
                                  model_cfg["temp_head"]["out_channels"])

    def forward(self, x: torch.Tensor, tasks: set[str] | None = None) -> dict:
        """x: (B, C_in, H, W) normalized. Returns dict of (B, 1, H, W) tensors.

        H and W are padded up to a multiple of ``patch_size`` and the heads'
        outputs cropped back. Without this, an odd grid silently loses a row:
        the strided patch-embed conv floors 49 -> 24 and the pixel-shuffle head
        returns 48, so predictions and targets would be misaligned in shape.
        Real domains do produce odd grids (the Colorado 4x4 deg default is
        49 x 40 at 10 km), so this is the common case, not an edge case.
        """
        # `is None` rather than a truthiness test: an explicitly EMPTY task set
        # means "run nothing", not "run the default", and the two must not
        # collapse into each other.
        tasks = {"temp"} if tasks is None else tasks
        if "temp" not in tasks:
            return {}
        H, W = x.shape[-2:]
        ps = self.patch_size
        pad_h, pad_w = (-H) % ps, (-W) % ps
        if pad_h or pad_w:
            mode = "reflect" if (pad_h < H and pad_w < W) else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

        out = {"temp": self.temp_head(self.backbone(x))}
        if pad_h or pad_w:
            out = {k: v[..., :H, :W] for k, v in out.items()}
        return out


def build_model(model_cfg: dict) -> nn.Module:
    """Dispatch on ``arch`` so every model shares one trainer and one eval.

    All architectures take (B, C_in, H, W) normalized input plus an optional
    ``tasks`` set and return {"temp": (B, 1, H, W)} — identical contract, so
    the comparison is apples-to-apples by construction rather than by
    convention. ``arch`` defaults to 'swin' for backward compatibility with
    checkpoints written before the baselines existed.
    """
    arch = model_cfg.get("arch", "swin")
    if arch == "swin":
        return DownscalingModel(model_cfg)
    if arch == "unet":
        from .unet import UNetDownscaler

        cfg = model_cfg.get("unet", {})
        return UNetDownscaler(model_cfg["in_channels"],
                              base=cfg.get("base", 96),
                              depth=cfg.get("depth", 4))
    if arch == "deepsd":
        from .deepsd import DeepSD

        cfg = model_cfg.get("deepsd", {})
        return DeepSD(model_cfg["in_channels"],
                      n_stages=cfg.get("n_stages", 3),
                      n1=cfg.get("n1", 64), n2=cfg.get("n2", 32))

    from . import zoo

    builders = {
        "swinir": zoo.SwinIR,
        "restormer": zoo.Restormer,
        "segformer": zoo.SegFormer,
        "maxvit": zoo.MaxViT,
        "vit": zoo.ViT,
        "edsr": zoo.EDSR,
        "esrt": zoo.ESRT,
        "swinir_light": zoo.SwinIR,
        "convnext": zoo.ConvNeXtUNet,
    }
    if arch in builders:
        kwargs = dict(model_cfg.get(arch, {}) or {})
        # YAML gives lists; these constructors expect tuples for the size specs.
        kwargs = {k: tuple(v) if isinstance(v, list) else v
                  for k, v in kwargs.items()}
        return builders[arch](model_cfg["in_channels"], **kwargs)

    raise ValueError(
        f"unknown arch {arch!r}; expected one of: swin, unet, deepsd, "
        f"{', '.join(builders)}")


def load_checkpoint(path: str | Path, device="cpu", fallback_cfg: dict | None = None):
    """Rebuild the trained network from a checkpoint. Returns (model, model_cfg).

    The architecture comes from the config STORED IN THE CHECKPOINT, not from
    configs/model.yaml: otherwise evaluating a unet checkpoint after the yaml
    has moved on to another arch silently constructs the wrong network. Every
    evaluation entry point goes through here so that rule holds in one place.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    mcfg = state.get("cfg") or fallback_cfg
    if mcfg is None:
        raise ValueError(f"{path} stores no model cfg and no fallback was given")
    model = build_model(mcfg).to(device)
    sd = {k: v for k, v in state["model"].items()
          if not k.startswith(_LEGACY_PREFIXES)}
    model.load_state_dict(sd)
    model.eval()
    return model, mcfg
