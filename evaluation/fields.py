"""Shared field loading for the map figures.

Every map figure in the repo needs the same four things: the store, the target
grid's lat/lon, the land/holdout masks, and the model's output on EVERY day
(not just the test split — the offset estimator is fit on the train days). That
last one is the expensive part, so it is cached to disk keyed by checkpoint
mtime: a figure script re-run after a tweak to its layout should not re-run
1826 forward passes.

Everything returned is in DEGREES C. The store holds anomalies relative to a
day-of-year climatology fitted on the training years, and mixing the two spaces
silently changes every number on a figure, so the conversion happens once here.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common import load_config  # noqa: E402
from common.grid import build_target_grid  # noqa: E402
from common.normalize import Normalizer  # noqa: E402
from data.build_dataset import NO_NORMALIZE  # noqa: E402
from models.model import load_checkpoint  # noqa: E402
from training.dataset import holdout_bounds  # noqa: E402

CACHE = REPO / ".field_cache"


class Store:
    """The store plus the geography every figure draws on top of it."""

    def __init__(self, zarr: str, cfg_name: str | None = None,
                 holdout: dict | None = None):
        self.path = Path(zarr)
        # load_config("data") resolves to the Colorado prototype unless
        # DOWNSCALE_CONFIG_data is exported. Every map figure is South-Central,
        # so name that config directly rather than depending on the caller's
        # environment — a figure drawn on the wrong domain is not obviously
        # wrong to look at.
        # SIM112 wants this upper-cased; it cannot be. common/config.py builds
        # the variable name as DOWNSCALE_CONFIG_<config stem>, and the stem is
        # "data".
        self.cfg = load_config(cfg_name or os.environ.get(
            "DOWNSCALE_CONFIG_data", "configs/data_sc_super.yaml"))  # noqa: SIM112
        self.ds = xr.open_zarr(zarr, consolidated=True)
        self.nz = Normalizer.load(str(self.path.parent / "norm_stats.json"))
        self.stored = self.ds["channel_in"].values.tolist()
        self.LAT, self.LON = build_target_grid(self.cfg["domain"], self.cfg["grid"]).latlon()

        self.box = holdout_bounds(self.ds, holdout or self.cfg["holdout"])
        i0, i1, j0, j1 = self.box
        self.land = self.ds["input"].values[0, self.stored.index("land_mask")] > 0.5
        hold = np.zeros_like(self.land)
        hold[i0:i1, j0:j1] = True
        self.m_hold = hold & self.land
        self.m_train = (~hold) & self.land          # disjoint by construction

        self.split = self.ds["split"].values
        self.i_train = np.where(self.split == "train")[0]
        self.i_test = np.where(self.split == "test")[0]
        self.times = pd.DatetimeIndex(self.ds["time"].values)
        self.doy = self.times.dayofyear.values - 1
        self.clim = self.ds["clim_tmp"].values
        self.clim_coarse = self.ds["clim_coarse_tmp"].values

    @property
    def shape(self):
        return self.land.shape

    def truth(self, absolute: bool = False) -> np.ndarray:
        """(T, H, W) target field. Anomaly unless absolute=True."""
        t = np.nan_to_num(self.ds["target"].values[:, 0]).astype("float32")
        return t + self.clim[self.doy] if absolute else t

    def coarse(self, absolute: bool = False) -> np.ndarray:
        """(T, H, W) the interpolated POWER input — the no-model baseline."""
        c = self.ds["input"].values[:, self.stored.index("coarse_tmp")].astype("float32")
        return c + self.clim_coarse[self.doy] if absolute else c

    def to_absolute(self, anom: np.ndarray) -> np.ndarray:
        return anom + self.clim[self.doy]


def predict_all(store: Store, arch: str, ckpt: str = "last",
                device: str = "cpu", use_cache: bool = True) -> np.ndarray:
    """(T, H, W) model output in ANOMALY degC for every day in the store.

    Cached on (checkpoint bytes-mtime, store path, ckpt tag). The estimator in
    5.7 is fit on the train split, so the train days are needed too — this is
    deliberately the whole store, not the test split.
    """
    path = REPO / "checkpoints" / arch / f"{ckpt}.pt"
    if not path.exists():
        raise FileNotFoundError(path)

    key = hashlib.sha1(
        f"{path}:{path.stat().st_mtime_ns}:{store.path}:{store.ds.sizes['time']}"
        .encode()).hexdigest()[:16]
    cached = CACHE / f"{arch}_{ckpt}_{key}.npy"
    if use_cache and cached.exists():
        return np.load(cached)

    model, mcfg = load_checkpoint(path, device, load_config("model"))
    names = list(mcfg.get("use_channels") or store.stored)
    idx = [store.stored.index(n) for n in names]
    n = store.ds.sizes["time"]
    out = np.empty((n, *store.shape), "float32")
    for k in range(n):
        x = store.ds["input"].isel(time=k).values[idx].copy()
        for c, nm in enumerate(names):
            if nm not in NO_NORMALIZE:
                x[c] = store.nz.transform(f"in::{nm}", x[c])
        np.nan_to_num(x, copy=False)
        with torch.no_grad():
            y = model(torch.from_numpy(x).unsqueeze(0).to(device),
                      {"temp"})["temp"][0, 0].cpu().numpy()
        out[k] = store.nz.inverse("out::tmp", y)

    if use_cache:
        CACHE.mkdir(exist_ok=True)
        np.save(cached, out)
    return out


def decompose(resid: np.ndarray, mask: np.ndarray):
    """Split a residual stack into the three terms of the 5.4 error budget.

    resid : (T, H, W) truth - prediction
    mask  : (H, W) the cells the budget is quoted over

    Returns (mu, s, eps, shares) where mu is (T,) the spatially-uniform daily
    offset, s is (H, W) the static spatial pattern, eps is (T, H, W) the
    space-time remainder, and shares are the three variance fractions. The
    terms are orthogonal by construction — mu is removed before s is measured
    and both are removed before eps — so the fractions sum to 1 and nothing is
    double counted.
    """
    mu = resid[:, mask].mean(1)                              # (T,)
    dm = resid - mu[:, None, None]
    s = np.where(mask, dm.mean(0), np.nan)                   # (H, W)
    eps = dm - np.nan_to_num(s)[None]

    v_mu = float(np.var(mu))
    v_s = float(np.var(s[mask]))
    v_eps = float(np.var(eps[:, mask]))
    tot = v_mu + v_s + v_eps
    return mu, s, eps, dict(offset=v_mu / tot, static=v_s / tot,
                            remainder=v_eps / tot, total_var=tot)


def highpass(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Field minus its Gaussian-smoothed self: the sub-(2 pi sigma) detail.

    Applied per time step. 'nearest' at the edges rather than zero-padding, so
    the domain rim does not manufacture a rim of spurious high-pass amplitude.
    """
    from scipy.ndimage import gaussian_filter

    f = np.asarray(field, "float32")
    if f.ndim == 2:
        return f - gaussian_filter(f, sigma_cells, mode="nearest")
    sm = np.empty_like(f)
    for k in range(f.shape[0]):
        sm[k] = gaussian_filter(f[k], sigma_cells, mode="nearest")
    return f - sm


def cellwise_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-cell Pearson r over time between two (T, H, W) stacks."""
    az = a - a.mean(0)
    bz = b - b.mean(0)
    num = (az * bz).sum(0)
    den = np.sqrt((az ** 2).sum(0) * (bz ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return np.where(mask & np.isfinite(r), r, np.nan)
