"""Derived static covariates: terrain shape and land-cover composition.

Both families exist to carry information the coarse predictor cannot hold.
Inside a single 50 km POWER cell the interpolated temperature is essentially
flat, so every piece of sub-coarse-cell structure the network can produce has
to come from a channel that varies at 10 km. These are those channels.

RESOLUTION IS THE WHOLE POINT. Every derivative is computed at the source
raster's NATIVE resolution (~90 m for the DEM, ~208 m for NLCD) and then
area-averaged onto the 10 km target grid. Computing slope from the already
aggregated 10 km field would describe the large-scale gradient, which the
coarse predictor already implies; computing it natively and aggregating
describes SUB-GRID ruggedness, which is genuinely new information. The one
deliberate exception is ``tpi``, which asks a target-scale question ("is this
cell low relative to its neighbours?") and is therefore computed at 10 km.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from common.grid import TargetGrid

# rasterio is imported lazily inside the functions, NOT at module scope. The
# channel-name constants below are read by build_dataset, which training and
# evaluation import in turn — so a module-level rasterio import would make
# every training run depend on the full geospatial stack it never uses.

# Names this module knows how to build, so build_dataset can ask for a subset.
TERRAIN_CHANNELS = ("elev_std", "slope_mean", "tpi", "northness", "eastness")

# NLCD Level-1 groups (the leading digit of the class code). Fractions of these
# replace the majority-class ``landcover`` channel, which is both a categorical
# code fed as a continuous number and a statistic that discards everything but
# the single most common class in each cell.
NLCD_GROUPS = {
    "lc_water": (11, 12),
    "lc_developed": (21, 22, 23, 24),
    "lc_barren": (31,),
    "lc_forest": (41, 42, 43),
    "lc_shrub": (51, 52),
    "lc_herbaceous": (71, 72, 73, 74),
    "lc_cultivated": (81, 82),
    "lc_wetland": (90, 95),
}
LANDCOVER_CHANNELS = tuple(NLCD_GROUPS)


def _aggregate(src: np.ndarray, src_transform, src_crs, tgrid: TargetGrid,
               resampling=None) -> np.ndarray:
    """Area-average a native-resolution array onto the target grid."""
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    from data.regrid import _affine

    resampling = Resampling.average if resampling is None else resampling

    out = np.full(tgrid.shape, np.nan, dtype="float32")
    reproject(source=np.ascontiguousarray(src, dtype="float32"),
              destination=out,
              src_transform=src_transform, src_crs=src_crs,
              dst_transform=_affine(tgrid), dst_crs=tgrid.crs,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=resampling)
    return out


def _as_da(arr: np.ndarray, tgrid: TargetGrid, name: str) -> xr.DataArray:
    return xr.DataArray(arr, dims=("y", "x"),
                        coords={"y": tgrid.y, "x": tgrid.x}, name=name)


def terrain_features(dem_path, tgrid: TargetGrid, wanted=TERRAIN_CHANNELS,
                     dem_10km: np.ndarray | None = None) -> list[xr.DataArray]:
    """Terrain shape descriptors on the target grid.

    ``elev_std``   standard deviation of elevation within each 10 km cell —
                   sub-grid relief. Computed as sqrt(E[z^2] - E[z]^2) so no
                   per-cell grouping is needed, just two area-averages.
    ``slope_mean`` mean slope in degrees.
    ``northness``  slope-weighted mean of cos(aspect); ``eastness`` the sine.
                   Aspect is averaged as a VECTOR — a plain mean of angles
                   would treat 359 deg and 1 deg as opposite. Weighting by
                   slope keeps flat cells, where aspect is meaningless, near
                   zero instead of contributing random directions.
    ``tpi``        topographic position index at 10 km: cell elevation minus
                   the mean of its 5x5 neighbourhood. Negative = basin, which
                   is where cold air pools on calm nights.
    """
    wanted = [w for w in wanted if w in TERRAIN_CHANNELS]
    if not wanted:
        return []

    import rasterio

    out: dict[str, np.ndarray] = {}
    native = {"elev_std", "slope_mean", "northness", "eastness"} & set(wanted)
    # Only touch the full-resolution DEM if something actually needs it: at
    # ~630 MB that read dominates the cost, and `tpi` alone can be served from
    # an already-aggregated field.
    if native or (("tpi" in wanted) and dem_10km is None):
        with rasterio.open(str(dem_path)) as src:
            dem = src.read(1).astype("float32")
            transform, crs = src.transform, src.crs
    else:
        dem = None

    dem_mean = None
    if native:
        # The DEM is in degrees, so metres-per-pixel depends on latitude: a
        # degree of longitude shrinks by cos(lat). A single scalar spacing
        # would tilt every slope in the domain.
        dlat, dlon = abs(transform.e), abs(transform.a)
        rows = np.arange(dem.shape[0], dtype="float64")
        lat = transform.f - (rows + 0.5) * dlat
        dy_m = 110_540.0 * dlat
        dx_m = (111_320.0 * np.cos(np.deg2rad(lat)) * dlon).astype("float32")

        dzdy, dzdx = np.gradient(dem, edge_order=1)
        dzdy /= np.float32(dy_m)
        dzdx /= dx_m[:, None]
        slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype("float32")

        slope_agg = _aggregate(slope, transform, crs, tgrid)
        if "slope_mean" in wanted:
            out["slope_mean"] = slope_agg
        if {"northness", "eastness"} & set(wanted):
            aspect = np.arctan2(-dzdx, dzdy)
            for nm, comp in (("northness", np.cos(aspect)),
                             ("eastness", np.sin(aspect))):
                if nm in wanted:
                    num = _aggregate((slope * comp).astype("float32"),
                                     transform, crs, tgrid)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        out[nm] = np.where(np.nan_to_num(slope_agg) > 1e-6,
                                           num / slope_agg, 0.0)
            del aspect
        del dzdx, dzdy, slope

        if "elev_std" in wanted:
            dem_mean = _aggregate(dem, transform, crs, tgrid)
            sq = _aggregate(np.square(dem), transform, crs, tgrid)
            out["elev_std"] = np.sqrt(np.clip(sq - dem_mean ** 2, 0, None))

    if "tpi" in wanted:
        from scipy.ndimage import uniform_filter

        base = dem_10km if dem_10km is not None else (
            dem_mean if dem_mean is not None
            else _aggregate(dem, transform, crs, tgrid))
        base = np.asarray(base, dtype="float32")
        out["tpi"] = base - uniform_filter(np.nan_to_num(base), size=5,
                                           mode="nearest")
    del dem

    return [_as_da(out[n].astype("float32"), tgrid, n) for n in wanted]


def landcover_fractions(lc_path, tgrid: TargetGrid,
                        wanted=LANDCOVER_CHANNELS) -> list[xr.DataArray]:
    """Fraction of each 10 km cell falling in each NLCD Level-1 group.

    Replaces the majority-class channel. The majority is a poor summary here:
    in the South-Central domain one class (shrub/scrub) is the mode in 48% of
    land cells, so nearly half the grid carries an identical value, and the
    developed classes are the mode in under 2% of cells despite urban cover
    reaching 99% somewhere. A composition keeps the minority classes that the
    mode throws away.

    Pixels marked nodata are excluded from the denominator rather than counted
    as "not this class", so a cell that is half outside NLCD coverage still
    reports the composition of the half that is covered.
    """
    import rasterio

    wanted = [w for w in wanted if w in NLCD_GROUPS]
    if not wanted:
        return []

    with rasterio.open(str(lc_path)) as src:
        codes = src.read(1)
        transform, crs = src.transform, src.crs
        nodata = src.nodata
    valid = np.ones(codes.shape, dtype=bool) if nodata is None \
        else codes != nodata

    out = []
    for name in wanted:
        m = np.isin(codes, NLCD_GROUPS[name]).astype("float32")
        m[~valid] = np.nan
        out.append(_as_da(_aggregate(m, transform, crs, tgrid), tgrid, name))
    return out
