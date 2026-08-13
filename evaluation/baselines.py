"""Cheap standard downscaling baselines the transformer must beat.

  * bilinear   — the coarse predictor already interpolated onto the target grid
                 (in this pipeline that is simply the coarse_* input channel).
  * BCSD       — Bias-Correction Spatial Disaggregation: monthly quantile
                 mapping of the coarse field to the truth climatology (fit on
                 train), then applied per timestep.

Both operate directly on the paired Zarr, so they are trivially comparable to
the model on identical splits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def _channel(ds: xr.Dataset, kind: str, name: str) -> xr.DataArray:
    dim = "channel_in" if kind == "input" else "channel_out"
    var = "input" if kind == "input" else "target"
    return ds[var].sel({dim: name})


def bilinear_baseline(ds: xr.Dataset, coarse_name: str) -> np.ndarray:
    """Prediction = the (already target-gridded) coarse channel."""
    return _channel(ds, "input", coarse_name).values


class LapseRateBaseline:
    """Bilinear + a global linear elevation correction (lapse rate).

    Fits truth - coarse ≈ a·dem + b by least squares on the train split, so
    ``a`` is the empirical lapse rate (°C per DEM unit; ~ -6.5 °C/km if DEM is
    in km). This is the *hard* temperature floor: it captures both the
    large-scale gradient (bilinear) and the elevation dependence a DEM gives
    for free. A transformer must beat this, not just bilinear.
    """

    def __init__(self):
        self.a = 0.0
        self.b = 0.0

    def fit(self, ds: xr.Dataset, coarse_name: str, truth_name: str,
            dem_name: str = "dem") -> LapseRateBaseline:
        train = ds.sel(time=ds["split"] == "train")
        coarse = _channel(train, "input", coarse_name).values
        truth = _channel(train, "target", truth_name).values
        dem = _channel(train, "input", dem_name).values
        resid = (truth - coarse).ravel()
        d = dem.ravel()
        mask = np.isfinite(resid) & np.isfinite(d)
        A = np.column_stack([d[mask], np.ones(mask.sum())])
        (self.a, self.b), *_ = np.linalg.lstsq(A, resid[mask], rcond=None)
        return self

    def predict(self, ds: xr.Dataset, coarse_name: str,
                dem_name: str = "dem") -> np.ndarray:
        coarse = _channel(ds, "input", coarse_name).values
        dem = _channel(ds, "input", dem_name).values
        return coarse + self.a * dem + self.b


class BCSD:
    """Quantile-mapping BCSD, fit per calendar month on the train split.

    For each month we build the empirical CDF of the coarse and truth fields
    (pooled over space+time) and remap coarse quantiles onto truth quantiles.
    Spatial disaggregation is implicit here because both fields share the
    target grid; on a true coarse->fine setup add a climatological scaling
    factor per cell.
    """

    def __init__(self, n_quantiles: int = 100):
        self.q = np.linspace(0, 1, n_quantiles)
        self.maps: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, ds: xr.Dataset, coarse_name: str, truth_name: str) -> BCSD:
        train = ds.sel(time=ds["split"] == "train")
        months = pd.DatetimeIndex(train["time"].values).month
        coarse = _channel(train, "input", coarse_name).values
        truth = _channel(train, "target", truth_name).values
        for m in range(1, 13):
            sel = months == m
            if not sel.any():
                continue
            cq = np.nanquantile(coarse[sel], self.q)
            tq = np.nanquantile(truth[sel], self.q)
            self.maps[m] = (cq, tq)
        return self

    def predict(self, ds: xr.Dataset, coarse_name: str) -> np.ndarray:
        months = pd.DatetimeIndex(ds["time"].values).month
        coarse = _channel(ds, "input", coarse_name).values
        out = np.empty_like(coarse)
        for i, m in enumerate(months):
            cq, tq = self.maps.get(m, self.maps[min(self.maps)])
            out[i] = np.interp(coarse[i], cq, tq)
        return out
