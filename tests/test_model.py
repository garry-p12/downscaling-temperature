"""The I/O contract every architecture must honour, plus checkpoint loading.

Contract: (B, C_in, H, W) -> {"temp": (B, 1, H, W)} with H, W preserved
EXACTLY. Odd grids are the common case here (Colorado is 49x40, the Austin crop
35x31), so a silently floored dimension would misalign predictions and targets.
"""
from __future__ import annotations

import pytest
import torch

from models.model import DownscalingModel, build_model, load_checkpoint

# One cheap config per family. Sizes are minimal — this is a shape/contract
# test, not a capacity test.
ZOO_CFGS = {
    "unet": {"unet": {"base": 8, "depth": 2}},
    "deepsd": {"deepsd": {"n_stages": 2, "n1": 8, "n2": 4}},
    "swinir": {"swinir": {"dim": 24, "depths": [1, 1], "num_heads": 2, "window_size": 4}},
    "restormer": {"restormer": {"dim": 8, "depths": [1, 1, 1, 1], "heads": [1, 1, 2, 2]}},
    "segformer": {"segformer": {"dims": [8, 16, 24, 32], "depths": [1, 1, 1, 1],
                                "heads": [1, 2, 3, 4], "srs": [4, 2, 1, 1],
                                "decoder_dim": 32}},
    "maxvit": {"maxvit": {"dim": 24, "depth": 1, "num_heads": 2, "p": 7}},
    "vit": {"vit": {"dim": 24, "depth": 1, "num_heads": 2, "patch_size": 4}},
    "edsr": {"edsr": {"dim": 16, "n_blocks": 2}},
    "esrt": {"esrt": {"dim": 16, "depth": 1, "num_heads": 2, "reduction": 4}},
    "convnext": {"convnext": {"base": 16, "depths": [1, 1, 1, 1]}},
}
ODD_GRID = (49, 40)     # the shipped Colorado domain


def _cfg(arch: str, in_ch: int = 4) -> dict:
    return {"arch": arch, "in_channels": in_ch, **ZOO_CFGS[arch]}


@pytest.mark.slow
@pytest.mark.parametrize("arch", sorted(ZOO_CFGS))
def test_every_arch_preserves_the_grid_exactly(arch):
    h, w = ODD_GRID
    model = build_model(_cfg(arch))
    out = model(torch.randn(1, 4, h, w))
    assert set(out) == {"temp"}
    assert out["temp"].shape == (1, 1, h, w), f"{arch} changed the grid"


@pytest.mark.slow
def test_swin_preserves_the_grid_exactly(tiny_model_cfg):
    h, w = ODD_GRID
    out = build_model(tiny_model_cfg)(torch.randn(1, 4, h, w))
    assert out["temp"].shape == (1, 1, h, w)


@pytest.mark.slow
@pytest.mark.parametrize("shape", [(35, 31), (48, 48), (17, 23)])
def test_odd_and_small_grids_survive_the_padding_path(shape, tiny_model_cfg):
    h, w = shape
    out = build_model(tiny_model_cfg)(torch.randn(1, 4, h, w))
    assert out["temp"].shape == (1, 1, h, w)


def test_unknown_arch_names_the_valid_options():
    with pytest.raises(ValueError, match="unknown arch"):
        build_model({"arch": "nope", "in_channels": 4})


def test_arch_defaults_to_swin_for_pre_baseline_checkpoints(tiny_model_cfg):
    cfg = dict(tiny_model_cfg)
    cfg.pop("arch")
    assert isinstance(build_model(cfg), DownscalingModel)


def test_empty_task_set_returns_no_heads(tiny_model_cfg):
    out = build_model(tiny_model_cfg)(torch.randn(1, 4, 16, 16), tasks=set())
    assert out == {}


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #
def test_load_checkpoint_rebuilds_from_the_stored_cfg(tmp_path, tiny_model_cfg):
    """Not from configs/model.yaml — otherwise evaluating an old checkpoint
    after the yaml moved on silently builds the wrong network."""
    model = build_model(_cfg("unet"))
    path = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "cfg": _cfg("unet")}, path)
    loaded, mcfg = load_checkpoint(path, "cpu", fallback_cfg=tiny_model_cfg)
    assert mcfg["arch"] == "unet"
    assert not loaded.training


def test_load_checkpoint_drops_legacy_precip_weights(tmp_path, tiny_model_cfg):
    """Checkpoints written before the precip heads were removed carry
    precip_stage*.* keys. They must keep loading, unmigrated."""
    sd = build_model(tiny_model_cfg).state_dict()
    sd["precip_stage2._noop"] = torch.zeros(1)
    sd["precip_stage1.out.bias"] = torch.zeros(1)
    path = tmp_path / "legacy.pt"
    torch.save({"model": sd, "cfg": tiny_model_cfg}, path)
    model, _ = load_checkpoint(path, "cpu")
    assert model(torch.randn(1, 4, 20, 20))["temp"].shape == (1, 1, 20, 20)


def test_load_checkpoint_still_rejects_a_genuine_weight_mismatch(tmp_path, tiny_model_cfg):
    """The legacy filter must not turn into a blanket strict=False."""
    sd = build_model(tiny_model_cfg).state_dict()
    sd.pop(next(iter(sd)))
    path = tmp_path / "broken.pt"
    torch.save({"model": sd, "cfg": tiny_model_cfg}, path)
    with pytest.raises(RuntimeError):
        load_checkpoint(path, "cpu")


def test_load_checkpoint_without_cfg_or_fallback_raises(tmp_path, tiny_model_cfg):
    path = tmp_path / "nocfg.pt"
    torch.save({"model": build_model(tiny_model_cfg).state_dict()}, path)
    with pytest.raises(ValueError, match="no model cfg"):
        load_checkpoint(path, "cpu")
