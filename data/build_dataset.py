"""Build the paired, chunked Zarr training dataset.

Aligns coarse predictors + static covariates -> high-res truth on the target
grid, one sample per timestep, and writes a single Zarr store plus a
normalization-stats JSON fit on the *train* split only.

Variables written:
  input   (time, channel, y, x)  float32  -- channel order = data.input_channels
  target  (time, channel, y, x)  float32  -- channel order = data.target_channels
  split   (time,)                <U5      -- 'train' | 'val' | 'test'

Real build:
    python -m data.build_dataset

Synthetic smoke build (no downloads; exercises model+train end-to-end):
    python -m data.build_dataset --synthetic --n 64 --size 100
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from common import Normalizer, build_target_grid, load_config
from common.grid import TargetGrid


# --------------------------------------------------------------------------- #
# Split assignment
# --------------------------------------------------------------------------- #
def assign_splits(times: pd.DatetimeIndex, time_cfg: dict) -> np.ndarray:
    split = np.full(len(times), "none", dtype="<U5")
    for name in ("train", "val", "test"):
        lo, hi = time_cfg[name]
        m = (times >= pd.Timestamp(lo)) & (times <= pd.Timestamp(hi))
        split[m] = name
    return split


# --------------------------------------------------------------------------- #
# Real source loaders (daily aggregation on the target grid)
# --------------------------------------------------------------------------- #
def load_coarse(cfg, tgrid: TargetGrid) -> xr.Dataset:
    """NASA POWER daily T2M (0.5 deg, ~50 km) -> bilinear onto the 10 km grid.

    POWER already publishes a daily value, so there is no temporal aggregation
    here; the only transform is the horizontal interpolation that turns the
    coarse field into the SR-style corrector's input channel.
    """
    from data.download_power import open_power
    from data.regrid import field_to_target

    param = cfg["sources"]["power"]["parameter"]
    ds = open_power(cfg)
    if param not in ds:
        raise KeyError(f"'{param}' not in POWER files (have {list(ds.data_vars)})")
    tmp = ds[param]
    if float(tmp.max()) > 200:                 # defensive: POWER T2M is degC
        tmp = tmp - 273.15
    method = cfg["dataset"].get("coarse_interp", "bilinear")
    tmp_t = field_to_target(tmp, tgrid, method).rename("coarse_tmp")
    return xr.merge([tmp_t])


def load_truth(cfg, tgrid: TargetGrid) -> xr.Dataset:
    """High-res daily mean temperature on the 10 km target grid."""
    which = cfg.get("truth_source", "aorc")
    if which == "era5_land":
        return _truth_era5_land(cfg, tgrid)
    if which == "aorc":
        return _truth_aorc(cfg, tgrid)
    if which == "prism":
        return _truth_prism(cfg, tgrid)
    raise ValueError(
        f"truth_source must be 'era5_land', 'aorc' or 'prism', got {which!r}")


def _truth_era5_land(cfg, tgrid: TargetGrid) -> xr.Dataset:
    """ERA5-Land 0.1 deg daily mean -> 10 km target grid.

    Bilinear, not conservative: ERA5-Land's native 0.1 deg is already at the
    target scale (~11 km lat / 8.7 km lon at 39N), so this is a reprojection
    between comparable grids, not an aggregation across a resolution gap.
    Area-weighted averaging here would smooth the truth below its own native
    resolution and quietly make the task easier.
    """
    from data.download_era5_land import open_era5_land
    from data.regrid import field_to_target

    ds = open_era5_land(cfg)
    var = "t2m" if "t2m" in ds else cfg["sources"]["era5_land"]["variable"]
    if var not in ds:
        raise KeyError(f"'{var}' not in ERA5-Land files "
                       f"(have {list(ds.data_vars)})")
    tmp = ds[var]
    if float(tmp.max()) > 200:                 # ERA5-Land ships Kelvin
        tmp = tmp - 273.15
    tmp_t = field_to_target(tmp, tgrid, "bilinear").rename("tmp")
    return xr.merge([tmp_t])


def _truth_aorc(cfg, tgrid: TargetGrid) -> xr.Dataset:
    """AORC hourly 1 km -> daily mean degC, conservatively aggregated to 10 km.

    Daily MEAN of the hourly field is the same statistic POWER reports as T2M,
    so coarse and truth are directly comparable (no definitional offset).
    """
    from data.regrid import field_to_target

    src = cfg["sources"]["aorc"]
    stores = sorted(Path(src["out_dir"]).glob("aorc_*.zarr"))
    if not stores:
        raise FileNotFoundError(f"No AORC zarr in {src['out_dir']}")
    ds = xr.open_mfdataset([str(s) for s in stores], engine="zarr",
                           combine="by_coords")
    tvar = src["variables"][0]
    tmp = ds[tvar].resample(time="1D").mean()
    if float(tmp.max()) > 200:  # Kelvin
        tmp = tmp - 273.15
    tmp_t = field_to_target(tmp, tgrid, "conservative").rename("tmp")
    return xr.merge([tmp_t])


def _truth_prism(cfg, tgrid: TargetGrid) -> xr.Dataset:
    """PRISM daily 4 km tmean BIL grids -> area-averaged onto the 10 km grid.

    NOTE: PRISM tmean is (tmin + tmax) / 2, not the mean of the diurnal cycle,
    so it carries a small systematic offset relative to POWER T2M. Fine as a
    cross-check; prefer AORC when you care about absolute bias.
    """
    import pandas as pd

    from data.regrid import raster_to_target

    src = cfg["sources"]["prism"]
    var = src["variables"][0]
    root = Path(src["out_dir"]) / var
    day_dirs = sorted(d for d in root.glob("[0-9]" * 8) if d.is_dir())
    if not day_dirs:
        raise FileNotFoundError(
            f"No PRISM {var} day folders in {root}. Run: "
            f"python -m data.download_prism --start ... --end ...")

    fields, times = [], []
    for d in day_dirs:
        bils = sorted(d.glob("*.bil"))
        if not bils:
            print(f"[prism] no .bil in {d}, skipping")
            continue
        # 'average' = area-weighted aggregation 4 km -> 10 km.
        fields.append(raster_to_target(str(bils[0]), tgrid, "average"))
        times.append(pd.Timestamp(d.name))
    da = xr.concat(fields, dim=pd.DatetimeIndex(times, name="time")).rename("tmp")
    return xr.merge([da])


def _fill_nan_nearest(da: xr.DataArray, name: str) -> xr.DataArray:
    """Nearest-neighbour fill for any residual NaN in a static field.

    Statics are fetched with a margin (see COV_PAD in download_covariates.py),
    but if a raster still does not cover every target cell, a single NaN in an
    input channel propagates through the network and makes the loss NaN. Fill
    from the closest valid cell and say how many were touched.
    """
    from scipy.ndimage import distance_transform_edt

    v = np.asarray(da.values, dtype="float32")
    mask = np.isnan(v)
    n = int(mask.sum())
    if n == 0:
        return da
    idx = distance_transform_edt(mask, return_distances=False,
                                 return_indices=True)
    da = da.copy(data=v[tuple(idx)])
    warnings.warn(
        f"[statics] {name}: filled {n} NaN cell(s) "
        f"({100 * n / v.size:.1f}%) by nearest neighbour — the source raster "
        f"does not fully cover the target grid; raise COV_PAD and re-fetch.",
        RuntimeWarning)
    return da


def _urban_fraction(lc_path: Path, tgrid: TargetGrid) -> xr.DataArray:
    """Fraction of each 10 km cell that is developed land (NLCD 21-24).

    The `landcover` channel is a MAJORITY class per cell, which discards the
    urban signal wherever a city does not dominate its whole 10 km cell — i.e.
    almost everywhere, since Austin's built-up core is a few cells across.
    A continuous developed-fraction channel keeps the urban heat island
    visible, which matters in a flat domain where terrain explains little.
    """
    import rasterio
    from rasterio.enums import Resampling

    from data.regrid import _affine

    with rasterio.open(lc_path) as src:
        codes = src.read(1)
        profile = src.profile
    urban = np.isin(codes, (21, 22, 23, 24)).astype("float32")

    tmp = lc_path.parent / "_urban_frac_src.tif"
    profile.update(dtype="float32", count=1, nodata=None, compress="deflate")
    profile.pop("photometric", None)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(urban, 1)

    import rioxarray  # noqa: F401

    da = xr.open_dataarray(str(tmp), engine="rasterio").squeeze()
    da = da.rio.reproject(tgrid.crs, transform=_affine(tgrid),
                          shape=tgrid.shape, resampling=Resampling.average)
    da = da.assign_coords(y=("y", tgrid.y), x=("x", tgrid.x))
    tmp.unlink(missing_ok=True)
    return da.rename("urban_frac")


def land_mask_from_regridded(truth: xr.Dataset, var: str = "tmp") -> xr.DataArray:
    """1 where the REGRIDDED truth is usable at every timestep, else 0.

    Derived from the target-grid truth, not the native grid. ERA5-Land's own
    land-sea mask is static, but the truth is interpolated onto the target grid
    and interpolation SMEARS NaN: a coastal cell with any ocean neighbour comes
    out NaN even though a nearest-resampled native mask calls it land. Deriving
    the mask from the regridded field makes mask, target and climatology
    consistent by construction rather than by coincidence.
    """
    finite = np.isfinite(truth[var].values).all(axis=0).astype("float32")
    mask = xr.DataArray(finite, dims=("y", "x"),
                        coords={"y": truth["y"], "x": truth["x"]},
                        name="land_mask")
    frac = float(finite.mean())
    print(f"[land_mask] {100 * frac:.1f}% of target cells have usable truth "
          f"({int((1 - frac) * finite.size)} cells excluded from loss)")
    return mask


def land_mask_from_truth(cfg, tgrid: TargetGrid) -> xr.DataArray:
    """1 where ERA5-Land reports land, 0 over ocean.

    ERA5-Land is a LAND reanalysis: every ocean cell is NaN. Two things follow.

    First, the mask must be built from the NATIVE grid BEFORE any interpolation
    — bilinear or cubic regridding of a field containing NaN smears the NaN
    across neighbouring cells, so a mask derived afterwards would be both wrong
    and larger than the true ocean.

    Second, the mask is regridded with NEAREST (a coastline is categorical, not
    something to average) and is used to exclude ocean from the loss and the
    metrics, not merely to inform the network. Without that, ocean NaNs
    propagate into the gradient and training dies on the first batch.
    """
    from data.download_era5_land import open_era5_land
    from data.regrid import field_to_target

    if cfg.get("truth_source") != "era5_land":
        warnings.warn(
            "land_mask is derived from ERA5-Land ocean NaNs; with "
            f"truth_source={cfg.get('truth_source')!r} it will be all-land.",
            RuntimeWarning)

    ds = open_era5_land(cfg)
    var = "t2m" if "t2m" in ds else cfg["sources"]["era5_land"]["variable"]
    # A cell counts as land only if truth exists at EVERY timestep. "finite at
    # any time" is too permissive: coastal cells that are NaN on some days pass
    # it, then produce a NaN climatology and a target the loss cannot use.
    # Requiring all-time coverage keeps mask, target and climatology consistent.
    finite = np.isfinite(ds[var].values).all(axis=0).astype("float32")
    native = xr.DataArray(finite, dims=("lat", "lon"),
                          coords={"lat": ds["lat"].values,
                                  "lon": ds["lon"].values}, name="land_mask")
    mask = field_to_target(native, tgrid, "nearest_s2d")
    mask = (mask > 0.5).astype("float32").rename("land_mask")
    frac = float(np.asarray(mask.values).mean())
    print(f"[land_mask] {100 * frac:.1f}% of target cells are land "
          f"({int((1 - frac) * mask.size)} ocean cells excluded from loss)")
    return mask


def load_statics(cfg, tgrid: TargetGrid,
                 truth: xr.Dataset | None = None) -> xr.Dataset:
    from data.regrid import coastal_distance, raster_to_target

    dem_p = Path(cfg["sources"]["dem"]["out_dir"]) / "dem.tif"
    lc_p = Path(cfg["sources"]["landcover"]["out_dir"]) / "landcover.tif"
    # 'average' not 'bilinear': the DEM is ~90 m and the target cells are
    # 10 km, so bilinear would point-sample one ridge or valley per cell.
    # 'mode' not 'nearest' for land cover, for the same reason on a
    # categorical field — the majority class beats whatever pixel the cell
    # center happens to land on.
    dem = _fill_nan_nearest(
        raster_to_target(str(dem_p), tgrid, "average"), "dem").rename("dem")
    lc = _fill_nan_nearest(
        raster_to_target(str(lc_p), tgrid, "mode"), "landcover").rename("landcover")
    coast_cfg = cfg["sources"].get("coastline") or {}
    cdist = coastal_distance(tgrid, coast_cfg.get("path"))
    urban = _fill_nan_nearest(_urban_fraction(lc_p, tgrid), "urban_frac") \
        .rename("urban_frac")
    fields = [dem, lc, cdist, urban]
    if "land_mask" in cfg["dataset"]["input_channels"]:
        fields.append(land_mask_from_regridded(truth) if truth is not None
                      else land_mask_from_truth(cfg, tgrid))
    return xr.merge(fields)


def doy_climatology(field: np.ndarray, times: pd.DatetimeIndex,
                    train_mask: np.ndarray, smooth_days: int = 15
                    ) -> np.ndarray:
    """Per-cell day-of-year climatology, fit on TRAIN days only.

    Returns (366, ny, nx). Feb 29 is included so leap days index cleanly.

    Fit on the training split alone — a climatology computed over all years
    would carry val/test information into the training target, which is exactly
    the leakage the temporal split exists to prevent.

    Smoothed with a WRAPPED centred rolling mean: with only ~3 training years,
    a raw per-DOY mean is 3 samples deep and very noisy, and day 365 must be
    adjacent to day 0 or the climatology has a discontinuity at New Year that
    the anomalies would inherit.
    """
    ny, nx = field.shape[1:]
    doy = times.dayofyear.values
    clim = np.full((366, ny, nx), np.nan, dtype="float32")
    for d in range(1, 367):
        sel = train_mask & (doy == d)
        if sel.any():
            clim[d - 1] = np.nanmean(field[sel], axis=0)

    # Fill any DOY with no training sample (Feb 29 in a non-leap train window)
    # from the nearest DOY that does have one.
    missing = np.isnan(clim).all(axis=(1, 2))
    if missing.any():
        idx = np.arange(366)
        good = idx[~missing]
        for d in idx[missing]:
            clim[d] = clim[good[np.abs(good - d).argmin()]]

    if smooth_days and smooth_days > 1:
        k = int(smooth_days)
        pad = k // 2
        wrapped = np.concatenate([clim[-pad:], clim, clim[:pad]], axis=0)
        kernel = np.ones(k, dtype="float32") / k
        clim = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="valid"), 0, wrapped
        ).astype("float32")
    return clim


def time_channels(times: pd.DatetimeIndex, names: list[str], shape) -> dict:
    """Time-varying, spatially-constant channels broadcast to the grid.

    Day-of-year is encoded as sin/cos so December and January are adjacent
    rather than maximally distant. The tree baselines put 35% of their feature
    importance on day-of-year while the networks had no time input at all —
    this closes that gap.
    """
    ny, nx = shape
    out = {}
    if not ({"doy_sin", "doy_cos"} & set(names)):
        return out
    ang = 2 * np.pi * times.dayofyear.values.astype("float32") / 365.25
    for name, vals in (("doy_sin", np.sin(ang)), ("doy_cos", np.cos(ang))):
        if name in names:
            out[name] = np.broadcast_to(
                vals.astype("float32")[:, None, None], (len(times), ny, nx))
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def assemble(coarse: xr.Dataset, truth: xr.Dataset, statics: xr.Dataset,
             cfg, tgrid: TargetGrid) -> xr.Dataset:
    # Align time across coarse & truth.
    coarse, truth = xr.align(coarse, truth, join="inner")
    times = pd.DatetimeIndex(coarse["time"].values)
    T = len(times)
    ny, nx = tgrid.shape

    in_names = cfg["dataset"]["input_channels"]
    tg_names = cfg["dataset"]["target_channels"]

    def broadcast_static(name):
        return np.broadcast_to(statics[name].values.astype("float32"),
                               (T, ny, nx))

    tchan = time_channels(times, in_names, (ny, nx))

    in_stack = []
    for name in in_names:
        if name in coarse:
            in_stack.append(coarse[name].values.astype("float32"))
        elif name in statics:
            in_stack.append(broadcast_static(name))
        elif name in tchan:
            in_stack.append(tchan[name])
        else:
            raise KeyError(f"input channel '{name}' not found in sources")
    inp = np.stack(in_stack, axis=1)                     # (T, C, y, x)
    tgt = np.stack([truth[n].values.astype("float32") for n in tg_names], axis=1)

    split = assign_splits(times, cfg["time"])

    # --- Anomalies -------------------------------------------------------- #
    # Subtract a smoothed day-of-year climatology from BOTH the coarse input
    # and the target, so the network predicts the fine-scale residual rather
    # than relearning the seasonal cycle. Climatologies are fit on train days
    # only and stored in the dataset so predictions convert back to degC.
    anom = cfg.get("anomaly", {}) or {}
    clim_vars = {}
    if anom.get("enabled", False):
        train = split == "train"
        if not train.any():
            raise ValueError("anomaly.enabled but the train split is empty")
        smooth = int(anom.get("smooth_days", 15))
        ci = in_names.index("coarse_tmp")
        ti = tg_names.index("tmp")
        doy0 = times.dayofyear.values - 1

        clim_in = doy_climatology(inp[:, ci], times, train, smooth)
        clim_out = doy_climatology(tgt[:, ti], times, train, smooth)
        inp[:, ci] -= clim_in[doy0]
        tgt[:, ti] -= clim_out[doy0]
        clim_vars = {"clim_coarse_tmp": clim_in, "clim_tmp": clim_out}
        print(f"[anomaly] subtracted {smooth}-day-smoothed DOY climatology "
              f"(fit on {int(train.sum())} train days); "
              f"coarse anomaly sd {inp[:, ci].std():.2f} degC, "
              f"target anomaly sd {np.nanstd(tgt[:, ti]):.2f} degC")

    return _to_dataset(inp, tgt, times, split, in_names, tg_names, tgrid,
                       clim_vars)


def _to_dataset(inp, tgt, times, split, in_names, tg_names, tgrid,
                clim_vars: dict | None = None):
    data_vars = dict(
        input=(("time", "channel_in", "y", "x"), inp),
        target=(("time", "channel_out", "y", "x"), tgt),
        split=(("time",), split),
    )
    coords = dict(time=times, channel_in=in_names, channel_out=tg_names,
                  y=tgrid.y, x=tgrid.x)
    # Climatologies travel WITH the dataset: every metric must be reported in
    # degC, which means adding the climatology back to the predicted anomaly.
    # Keeping them in a side file would let the two drift apart.
    for name, arr in (clim_vars or {}).items():
        data_vars[name] = (("doy", "y", "x"), arr)
    if clim_vars:
        coords["doy"] = np.arange(1, 367)
    return xr.Dataset(
        data_vars=data_vars, coords=coords,
        attrs=dict(crs=tgrid.crs, res_m=tgrid.res_m,
                   anomaly=int(bool(clim_vars))),
    )


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
# Channels that must NOT be z-scored. land_mask is a 0/1 indicator used to
# exclude ocean from the loss; standardizing it would turn it into two
# arbitrary real values and break that use.
NO_NORMALIZE = {"land_mask"}


def fit_normalizer(ds: xr.Dataset, cfg) -> Normalizer:
    nz = Normalizer()
    train = ds.sel(time=ds["split"] == "train")
    for i, name in enumerate(ds["channel_in"].values.tolist()):
        if name in NO_NORMALIZE:
            continue
        nz.fit(f"in::{name}", train["input"].isel(channel_in=i).values)
    for i, name in enumerate(ds["channel_out"].values.tolist()):
        nz.fit(f"out::{name}", train["target"].isel(channel_out=i).values)
    return nz


# --------------------------------------------------------------------------- #
# Synthetic mode
# --------------------------------------------------------------------------- #
def build_synthetic(n: int, size: int, cfg, block: int = 5
                    ) -> tuple[xr.Dataset, TargetGrid]:
    """Deterministic fake data posing a *real* downscaling task (not denoising).

    The coarse predictor is a BLOCK-AVERAGE of the truth (over ``block`` cells,
    default 5 == the real 50 km -> 10 km factor) re-expanded to the target
    grid: it retains the large-scale field but the
    sub-block, DEM-driven fine structure is withheld. That structure lives only
    in the full-resolution ``dem`` input, so:
      * bilinear (== the coarse channel) cannot recover it — by construction;
      * a model that uses the DEM can.
    This is what makes "beats bilinear" and rising resid_corr meaningful. If the
    model still can't beat bilinear here, that's a model/harness problem, not a
    rigged floor.
    """
    from scipy.ndimage import zoom

    rng = np.random.RandomState(0)
    ny = nx = size
    assert size % block == 0, f"synthetic size must be divisible by {block}"
    y = np.arange(ny, dtype="float64") * 10000.0
    x = np.arange(nx, dtype="float64") * 10000.0
    tgrid = TargetGrid(crs=cfg["grid"]["target_crs"],
                       res_m=cfg["grid"]["target_res_m"], x=x, y=y)
    in_names = cfg["dataset"]["input_channels"]
    tg_names = cfg["dataset"]["target_channels"]

    # Static DEM: broad terrain (survives block-averaging) + fine ridges
    # (period ~2*block cells, so block-averaging attenuates them -> withheld).
    yy, xx = np.meshgrid(np.linspace(0, 1, ny), np.linspace(0, 1, nx), indexing="ij")
    dem = (0.6 * np.sin(3 * np.pi * yy) * np.cos(3 * np.pi * xx)
           + 0.4 * np.sin(12 * np.pi * yy) * np.cos(11 * np.pi * xx)).astype("float32")
    landcover = np.floor((dem + 1) * 2).astype("float32")
    coastal = xx.astype("float32")
    # A couple of built-up blobs, on the same 0-1 fractional scale the real
    # NLCD-derived channel uses.
    urban = np.exp(-((yy - 0.3) ** 2 + (xx - 0.7) ** 2) / 0.01).astype("float32")
    # All land: the synthetic domain is inland, like Colorado and the Austin
    # crop. Ocean handling is exercised against the real store, not here.
    land_mask = np.ones((ny, nx), "float32")
    lat_g = np.linspace(0, 2, ny)[:, None]
    lon_g = np.linspace(0, 2, nx)[None, :]
    lapse = -6.0                                    # °C per DEM unit

    def block_upsample(field):
        small = field.reshape(ny // block, block, nx // block, block).mean((1, 3))
        return zoom(small, (block, block), order=1).astype("float32")

    # Built before the loop: doy_sin/doy_cos are per-DAY channels, so the
    # calendar has to exist while the samples are generated.
    times = pd.date_range("2019-01-01", periods=n, freq="D")
    doy = times.dayofyear.values

    inp = np.zeros((n, len(in_names), ny, nx), "float32")
    tgt = np.zeros((n, len(tg_names), ny, nx), "float32")
    for t in range(n):
        phase = t / 6.0
        weather = 1.5 * rng.randn()                 # large-scale daily offset (recoverable)
        large_t = 15 + 8 * np.sin(lat_g + phase) + 2 * np.cos(lon_g - 0.3 * phase)
        truth_tmp = (large_t + lapse * dem + weather).astype("float32")
        ang = 2 * np.pi * doy[t] / 365.25
        fields = {"coarse_tmp": block_upsample(truth_tmp),   # fine structure gone
                  "dem": dem, "landcover": landcover, "coastal_dist": coastal,
                  "urban_frac": urban, "land_mask": land_mask,
                  "doy_sin": np.full((ny, nx), np.sin(ang), "float32"),
                  "doy_cos": np.full((ny, nx), np.cos(ang), "float32"),
                  "tmp": truth_tmp}
        missing = [nm for nm in (*in_names, *tg_names) if nm not in fields]
        if missing:
            raise KeyError(
                f"synthetic mode cannot produce {missing}; either add them here "
                f"or point --synthetic at a config whose channels it covers")
        for c, name in enumerate(in_names):
            inp[t, c] = fields[name]
        for c, name in enumerate(tg_names):
            tgt[t, c] = fields[name]

    split = assign_splits(times, {
        "train": ["2019-01-01", times[int(n * 0.7)].strftime("%Y-%m-%d")],
        "val": [times[int(n * 0.7) + 1].strftime("%Y-%m-%d"),
                times[int(n * 0.85)].strftime("%Y-%m-%d")],
        "test": [times[int(n * 0.85) + 1].strftime("%Y-%m-%d"),
                 times[-1].strftime("%Y-%m-%d")],
    })
    return _to_dataset(inp, tgt, times, split, in_names, tg_names, tgrid), tgrid


# --------------------------------------------------------------------------- #
def write(ds: xr.Dataset, nz: Normalizer, out_zarr: str, chunks: dict) -> None:
    out = Path(out_zarr)
    ds = ds.chunk({"time": chunks.get("time", 32)})
    for v in ds.variables:                      # drop source encoding
        ds[v].encoding = {}
    ds.to_zarr(out, mode="w", consolidated=True)
    nz.save(out.parent / "norm_stats.json")
    print(f"[dataset] wrote {out} ({ds.sizes['time']} steps) "
          f"and {out.parent / 'norm_stats.json'}")


def main(synthetic: bool, n: int, size: int, norm_from: str | None = None,
         out_zarr: str | None = None) -> None:
    cfg = load_config("data")
    if synthetic:
        ds, _ = build_synthetic(n, size, cfg)
    else:
        tgrid = build_target_grid(cfg["domain"], cfg["grid"])
        coarse = load_coarse(cfg, tgrid)
        truth = load_truth(cfg, tgrid)
        statics = load_statics(cfg, tgrid, truth)
        ds = assemble(coarse, truth, statics, cfg, tgrid)

    if norm_from:
        # Zero-shot transfer: the model must see inputs standardized EXACTLY as
        # in training. Re-fitting on the new domain would rescale every channel
        # to local statistics — a 200 m Texas DEM would be normalized to the
        # same range as a 2100 m Colorado one, hiding the distribution shift
        # from the model and silently invalidating the transfer test.
        nz = Normalizer.load(norm_from)
        print(f"[dataset] reusing normalization from {norm_from} "
              f"(zero-shot: do NOT refit on the target domain)")
    else:
        nz = fit_normalizer(ds, cfg)
    write(ds, nz, out_zarr or cfg["dataset"]["out_zarr"],
          cfg["dataset"].get("chunks", {}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=64, help="synthetic timesteps")
    ap.add_argument("--size", type=int, default=100,
                    help="synthetic grid size (must divide by the block factor 5)")
    ap.add_argument("--out-zarr", default=None,
                    help="override dataset.out_zarr (keeps an existing store intact)")
    ap.add_argument("--norm-from", default=None,
                    help="reuse an existing norm_stats.json instead of fitting "
                         "(required for zero-shot transfer to a new domain)")
    args = ap.parse_args()
    main(args.synthetic, args.n, args.size, args.norm_from, args.out_zarr)
