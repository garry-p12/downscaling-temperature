"""Torch Dataset over the paired Zarr store.

Normalization, patch sampling, and — the important part — a SPATIAL HOLDOUT.
Training patches are rejected if they overlap the evaluation region (Austin),
so evaluating there measures spatial generalization rather than memorized
per-cell corrections. A purely temporal split cannot show that: train and test
share every grid cell, so the network can learn "cell (i,j) runs 2 °C cold"
and score well without having learned anything transferable.
"""
from __future__ import annotations

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from common import Normalizer
from data.build_dataset import NO_NORMALIZE


def holdout_bounds(ds: xr.Dataset, holdout: dict) -> tuple[int, int, int, int]:
    """Grid-index bounds (i0, i1, j0, j1) of a lat/lon holdout box + buffer.

    The store is in a projected CRS, so the lat/lon box is converted by
    projecting every cell centre and taking the index extent of those inside.
    Albers rotates against lat/lon, so a lat/lon rectangle is not an index
    rectangle — taking the bounding extent of the matching cells is the
    conservative reading (it can only make the excluded region larger).
    """
    from pyproj import Transformer

    buf = float(holdout.get("buffer_deg", 0.0))
    xx, yy = np.meshgrid(ds["x"].values, ds["y"].values)
    tf = Transformer.from_crs(ds.attrs.get("crs", "EPSG:5070"), "EPSG:4326",
                              always_xy=True)
    lon, lat = tf.transform(xx, yy)
    inside = ((lon >= holdout["lon_min"] - buf) & (lon <= holdout["lon_max"] + buf)
              & (lat >= holdout["lat_min"] - buf) & (lat <= holdout["lat_max"] + buf))
    if not inside.any():
        raise ValueError("holdout box does not intersect the domain")
    rows, cols = np.where(inside)
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


class DownscaleDataset(Dataset):
    """
    holdout : dict | None
        Lat/lon box (+ ``buffer_deg``) defining the evaluation region.
    holdout_mode : 'exclude' | 'only' | None
        'exclude' — patches may cover the whole domain, but every cell inside
                    the box is ZEROED IN THE LOSS MASK, so it contributes no
                    gradient. Rejecting overlapping patches instead is
                    impossible here: the holdout sits mid-domain (rows 36-73 of
                    93), leaving no room for a 48-px patch above or below, and
                    shrinking patches until they fit would confine training to
                    thin edge strips — destroying the terrain diversity the
                    broad domain exists to provide. Masking gives the same
                    guarantee (no holdout cell is ever trained on) without
                    restricting where patches may be drawn.
        'only'    — return exactly the box as a fixed crop (evaluation).
    """

    def __init__(self, zarr_path: str, norm_path: str, split: str,
                 patch_size: int | None = None, seed: int = 0,
                 holdout: dict | None = None,
                 holdout_mode: str | None = None,
                 patches_per_sample: int = 1):
        self.ds = xr.open_zarr(zarr_path, consolidated=True)
        self.nz = Normalizer.load(norm_path)
        self.index = np.where(self.ds["split"].values == split)[0]
        self.patch = patch_size
        self.in_names = self.ds["channel_in"].values.tolist()
        self.out_names = self.ds["channel_out"].values.tolist()
        self.rng = np.random.RandomState(seed)
        self.H = self.ds.sizes["y"]
        self.W = self.ds.sizes["x"]
        self.patches_per_sample = max(1, int(patches_per_sample))

        self.holdout_mode = holdout_mode
        self.box = None
        self.holdout_grid = None
        if holdout and holdout_mode:
            self.box = holdout_bounds(self.ds, holdout)
            i0, i1, j0, j1 = self.box
            if holdout_mode == "only":
                self.patch = None            # fixed crop, no random sampling
            frac = (i1 - i0) * (j1 - j0) / (self.H * self.W)
            if holdout_mode == "exclude":
                g = np.ones((self.H, self.W), dtype="float32")
                g[i0:i1, j0:j1] = 0.0        # 0 = never contributes to loss
                self.holdout_grid = g
            print(f"[dataset] holdout '{holdout.get('name','eval')}' "
                  f"rows {i0}:{i1} cols {j0}:{j1} "
                  f"({100 * frac:.1f}% of domain), mode={holdout_mode}")

    def __len__(self) -> int:
        return len(self.index) * self.patches_per_sample

    # ------------------------------------------------------------------ #
    def _overlaps_holdout(self, i: int, j: int) -> bool:
        i0, i1, j0, j1 = self.box
        return not (i + self.patch <= i0 or i >= i1
                    or j + self.patch <= j0 or j >= j1)

    def _sample_origin(self) -> tuple[int, int]:
        return (self.rng.randint(0, max(1, self.H - self.patch + 1)),
                self.rng.randint(0, max(1, self.W - self.patch + 1)))

    def _crop(self, arr):                      # arr: (C, H, W)
        if self.holdout_mode == "only" and self.box is not None:
            i0, i1, j0, j1 = self.box
            return arr[:, i0:i1, j0:j1]
        if self.patch is None or (self.patch >= self.H and self.patch >= self.W):
            return arr
        i, j = self._sample_origin()
        return arr[:, i:i + self.patch, j:j + self.patch]

    # ------------------------------------------------------------------ #
    def __getitem__(self, k: int):
        t = int(self.index[k % len(self.index)])
        sample = self.ds.isel(time=t)
        x = sample["input"].values.astype("float32")     # (C_in, H, W)
        y = sample["target"].values.astype("float32")    # (C_out, H, W)

        for c, name in enumerate(self.in_names):
            if name in NO_NORMALIZE:            # keep 0/1 indicators intact
                continue
            x[c] = self.nz.transform(f"in::{name}", x[c])
        for c, name in enumerate(self.out_names):
            y[c] = self.nz.transform(f"out::{name}", y[c])
        # ERA5-Land is NaN over ocean. Zero them so no NaN reaches the network;
        # the land_mask channel is what excludes those cells from the loss, so
        # the zeros are never scored.
        np.nan_to_num(x, copy=False)
        np.nan_to_num(y, copy=False)

        # Loss mask: usable truth (land) AND outside the spatial holdout.
        mi = self.mask_index()
        m = x[mi:mi + 1].copy() if mi is not None \
            else np.ones((1, self.H, self.W), dtype="float32")
        if self.holdout_grid is not None:
            m = m * self.holdout_grid[None]

        both = self._crop(np.concatenate([x, y, m], axis=0))
        nx, ny = x.shape[0], y.shape[0]
        x, y, m = both[:nx], both[nx:nx + ny], both[nx + ny:]
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
        return t(x), t(y), t(m)

    # ------------------------------------------------------------------ #
    def target_index(self, name: str) -> int:
        return self.out_names.index(name)

    def mask_index(self) -> int | None:
        """Index of the land_mask input channel, if present."""
        return self.in_names.index("land_mask") \
            if "land_mask" in self.in_names else None

    def climatology(self, var: str = "tmp") -> np.ndarray | None:
        """(366, y, x) climatology for converting anomalies back to degC."""
        name = f"clim_{var}"
        return self.ds[name].values if name in self.ds else None

    def times(self) -> np.ndarray:
        return self.ds["time"].values[self.index]

    def clim_for_samples(self, var: str = "tmp") -> np.ndarray | None:
        """(N, y, x) climatology aligned to this split's sample order.

        Metrics must be reported in absolute degC. For model-vs-truth the
        climatology cancels in RMSE/MAE/bias, but NOT for the bilinear
        baseline: coarse and target have separate climatologies, so comparing
        them in anomaly space measures a different quantity. Converting both
        back to degC keeps every method on one scale.
        """
        clim = self.climatology(var)
        if clim is None:
            return None
        import pandas as pd
        doy = pd.DatetimeIndex(self.times()).dayofyear.values - 1
        out = clim[doy]
        if self.holdout_mode == "only" and self.box is not None:
            i0, i1, j0, j1 = self.box
            out = out[:, i0:i1, j0:j1]
        return out
