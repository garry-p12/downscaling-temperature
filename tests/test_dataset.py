"""Spatial holdout geometry and the arch/checkpoint path map.

The holdout is the whole basis of the reported result — if a training patch can
see an Austin cell, the number in README.md means nothing — so its bounds are
tested directly rather than through a full Dataset (which would need a store).
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from common.config import load_config
from common.grid import build_target_grid
from training.dataset import holdout_bounds
from training.run_all import ALL_ARCHS, TIER_A, ckpt_path


def _store_like(domain_cfg: str = "configs/data_southcentral.yaml") -> xr.Dataset:
    """An empty Dataset carrying only the coords holdout_bounds reads."""
    cfg = load_config(domain_cfg)
    g = build_target_grid(cfg["domain"], cfg["grid"])
    ds = xr.Dataset(coords={"x": g.x, "y": g.y})
    ds.attrs["crs"] = cfg["grid"]["target_crs"]
    return ds


def test_holdout_bounds_are_inside_the_grid():
    cfg = load_config("configs/data_southcentral.yaml")
    ds = _store_like()
    i0, i1, j0, j1 = holdout_bounds(ds, cfg["holdout"])
    assert 0 <= i0 < i1 <= ds.sizes["y"]
    assert 0 <= j0 < j1 <= ds.sizes["x"]


def test_holdout_is_a_small_interior_crop_not_the_whole_domain():
    cfg = load_config("configs/data_southcentral.yaml")
    ds = _store_like()
    i0, i1, j0, j1 = holdout_bounds(ds, cfg["holdout"])
    frac = ((i1 - i0) * (j1 - j0)) / (ds.sizes["y"] * ds.sizes["x"])
    assert 0.01 < frac < 0.25, f"holdout covers {frac:.1%} of the domain"


def test_buffer_only_ever_grows_the_excluded_region():
    """The buffer exists so patch EDGES stay clear; shrinking it would silently
    let training patches touch the evaluation crop."""
    cfg = load_config("configs/data_southcentral.yaml")
    ds = _store_like()
    box = dict(cfg["holdout"])
    tight = holdout_bounds(ds, {**box, "buffer_deg": 0.0})
    wide = holdout_bounds(ds, {**box, "buffer_deg": 1.0})
    assert wide[0] <= tight[0] and wide[1] >= tight[1]
    assert wide[2] <= tight[2] and wide[3] >= tight[3]


def test_holdout_outside_the_domain_raises_rather_than_silently_emptying():
    ds = _store_like()
    off_grid = {"lon_min": 10.0, "lon_max": 11.0,
                "lat_min": 50.0, "lat_max": 51.0, "buffer_deg": 0.0}
    with pytest.raises(ValueError, match="does not intersect"):
        holdout_bounds(ds, off_grid)


def test_holdout_bounds_cover_every_cell_in_the_lat_lon_box():
    """Albers rotates against lat/lon, so the index extent must be the
    conservative superset of the matching cells — never a subset."""
    cfg = load_config("configs/data_southcentral.yaml")
    ds = _store_like()
    hold = cfg["holdout"]
    i0, i1, j0, j1 = holdout_bounds(ds, hold)

    from pyproj import Transformer
    xx, yy = np.meshgrid(ds["x"].values, ds["y"].values)
    lon, lat = Transformer.from_crs(ds.attrs["crs"], "EPSG:4326",
                                    always_xy=True).transform(xx, yy)
    buf = hold["buffer_deg"]
    inside = ((lon >= hold["lon_min"] - buf) & (lon <= hold["lon_max"] + buf)
              & (lat >= hold["lat_min"] - buf) & (lat <= hold["lat_max"] + buf))
    rows, cols = np.where(inside)
    assert rows.min() >= i0 and rows.max() < i1
    assert cols.min() >= j0 and cols.max() < j1


# --------------------------------------------------------------------------- #
# Sweep bookkeeping
# --------------------------------------------------------------------------- #
def test_every_listed_arch_gets_a_distinct_checkpoint_path():
    """Comparison runs must not clobber each other's best.pt."""
    paths = [ckpt_path(a) for a in ALL_ARCHS]
    assert len(set(paths)) == len(paths)


def test_tier_a_is_a_subset_of_all_archs():
    assert set(TIER_A) <= set(ALL_ARCHS)


def test_seed_repeat_runs_get_their_own_directory():
    assert ckpt_path("swin_s1") != ckpt_path("swin")


def test_every_arch_in_the_sweep_is_buildable():
    """run_all would otherwise fail hours into a sweep on a typo."""
    from models.model import build_model

    cfg = load_config("model")
    for arch in ALL_ARCHS:
        build_model({**cfg, "arch": arch, "in_channels": 3, "encoder":
                     {**cfg["encoder"], "embed_dim": 24, "depths": [1, 1, 1, 1],
                      "num_heads": [1, 1, 1, 1], "window_size": 4}})


# --------------------------------------------------------------------------- #
# Channel subsetting — the mechanism every covariate ablation rests on
# --------------------------------------------------------------------------- #
def _tiny_store(tmp_path, channels=("coarse_tmp", "dem", "land_mask")):
    """A minimal but real Zarr store + norm_stats, enough to drive the Dataset."""
    import json

    from common import Normalizer
    from training.dataset import DownscaleDataset  # noqa: F401

    T, H, W = 4, 12, 10
    rng = np.random.default_rng(0)
    inp = rng.normal(size=(T, len(channels), H, W)).astype("float32")
    inp[:, channels.index("land_mask")] = 1.0
    ds = xr.Dataset(
        {"input": (("time", "channel_in", "y", "x"), inp),
         "target": (("time", "channel_out", "y", "x"),
                    rng.normal(size=(T, 1, H, W)).astype("float32")),
         "split": ("time", np.array(["train"] * T, dtype="<U5"))},
        coords={"time": np.arange(T), "channel_in": list(channels),
                "channel_out": ["tmp"], "y": np.arange(H, dtype="float64"),
                "x": np.arange(W, dtype="float64")})
    store = tmp_path / "s.zarr"
    ds.to_zarr(store, mode="w", consolidated=True)

    nz = Normalizer()
    for c in channels:
        nz.fit(f"in::{c}", inp[:, channels.index(c)])
    nz.fit("out::tmp", ds["target"].values)
    norm = tmp_path / "norm_stats.json"
    norm.write_text(json.dumps(nz.stats))
    return str(store), str(norm)


def test_use_channels_selects_and_reorders(tmp_path):
    from training.dataset import DownscaleDataset

    store, norm = _tiny_store(tmp_path)
    full = DownscaleDataset(store, norm, "train")
    assert full.in_names == ["coarse_tmp", "dem", "land_mask"]
    assert full[0][0].shape[0] == 3

    sub = DownscaleDataset(store, norm, "train",
                           use_channels=["land_mask", "coarse_tmp"])
    assert sub.in_names == ["land_mask", "coarse_tmp"]
    assert sub[0][0].shape[0] == 2


def test_use_channels_takes_the_right_data_not_just_the_right_count(tmp_path):
    """A subset must carry the SAME values the full stack had for those
    channels — an off-by-one in the index map would be silent otherwise."""
    from training.dataset import DownscaleDataset

    store, norm = _tiny_store(tmp_path)
    full = DownscaleDataset(store, norm, "train")
    sub = DownscaleDataset(store, norm, "train", use_channels=["dem"])
    assert np.allclose(sub[0][0][0].numpy(),
                       full[0][0][full.in_names.index("dem")].numpy())


def test_unknown_channel_raises_rather_than_training_on_the_wrong_stack(tmp_path):
    from training.dataset import DownscaleDataset

    store, norm = _tiny_store(tmp_path)
    with pytest.raises(KeyError, match="elev_std"):
        DownscaleDataset(store, norm, "train", use_channels=["dem", "elev_std"])


def test_mask_index_follows_the_subset(tmp_path):
    """land_mask moves position when channels are dropped; the loss mask must
    follow it, or the wrong channel gets used to exclude ocean."""
    from training.dataset import DownscaleDataset

    store, norm = _tiny_store(tmp_path)
    sub = DownscaleDataset(store, norm, "train",
                           use_channels=["land_mask", "dem"])
    assert sub.mask_index() == 0
    assert DownscaleDataset(store, norm, "train",
                            use_channels=["dem"]).mask_index() is None


# --------------------------------------------------------------------------- #
# Dead-channel guard
# --------------------------------------------------------------------------- #
def _ds_with(channels, arrays):
    import xarray as xr
    inp = np.stack(arrays, axis=1).astype("float32")     # (T, C, y, x)
    return xr.Dataset(
        {"input": (("time", "channel_in", "y", "x"), inp)},
        coords={"time": np.arange(inp.shape[0]), "channel_in": list(channels),
                "y": np.arange(inp.shape[2]), "x": np.arange(inp.shape[3])})


def test_constant_channel_fails_the_build():
    """A silently dead channel would make an ablation on it report 'no effect'."""
    from data.build_dataset import check_no_dead_channels

    rng = np.random.default_rng(0)
    live = rng.normal(size=(2, 4, 4))
    dead = np.zeros((2, 4, 4))
    with pytest.raises(ValueError, match="coastal_dist"):
        check_no_dead_channels(_ds_with(["coarse_tmp", "coastal_dist"],
                                        [live, dead]))


def test_land_mask_may_be_constant_on_an_all_land_domain():
    from data.build_dataset import check_no_dead_channels

    rng = np.random.default_rng(0)
    check_no_dead_channels(_ds_with(
        ["coarse_tmp", "land_mask"],
        [rng.normal(size=(2, 4, 4)), np.ones((2, 4, 4))]))


def test_a_healthy_stack_passes():
    from data.build_dataset import check_no_dead_channels

    rng = np.random.default_rng(1)
    check_no_dead_channels(_ds_with(["a", "b"], [rng.normal(size=(2, 4, 4)),
                                                 rng.normal(size=(2, 4, 4))]))
