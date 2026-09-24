"""Regridding onto the target grid.

Two paths:
  * lat/lon gridded fields (ERA5, AORC, PRISM)  -> ``field_to_target``
      - conservative/area-weighted for aggregating hi-res truth down to 10 km
      - bilinear for the coarse predictor
  * projected rasters (DEM, land cover GeoTIFFs) -> ``raster_to_target``

Uses ``xesmf`` when available (proper conservative regridding); otherwise falls
back to scipy griddata (bilinear/nearest only — conservative is approximated by
nearest with a warning).
"""
from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from common.grid import TargetGrid

try:
    import xesmf  # noqa: F401
    _HAVE_XESMF = True
except Exception:  # noqa: BLE001
    _HAVE_XESMF = False


def _target_dataset(tgrid: TargetGrid) -> xr.Dataset:
    lat, lon = tgrid.latlon()
    return xr.Dataset(
        coords=dict(
            lat=(("y", "x"), lat),
            lon=(("y", "x"), lon),
            y=("y", tgrid.y),
            x=("x", tgrid.x),
        )
    )


def _regular_grid_interp(da: xr.DataArray, tgrid: TargetGrid,
                         method: str) -> xr.DataArray:
    """Interpolate a REGULAR lat/lon grid onto the target grid.

    Both POWER and ERA5-Land are regular lat/lon grids, so a tensor-product
    interpolator is the right tool: it is exact about the grid structure, far
    faster than scattered-point ``griddata``, and — crucially — it supports
    genuine **cubic** interpolation. ``griddata(method='cubic')`` builds a
    Clough-Tocher triangulation over scattered points, which is both slow and
    not the bicubic interpolation the downscaling literature means.

    Out-of-domain target cells are filled by nearest-neighbour rather than
    left NaN; the target grid is snapped outward in Albers, so its corners can
    fall marginally outside the source bbox even after padding.
    """
    from scipy.interpolate import RegularGridInterpolator

    lat = np.asarray(da["lat"].values)
    lon = np.asarray(da["lon"].values)
    # RegularGridInterpolator needs strictly ascending axes.
    flip_lat, flip_lon = lat[0] > lat[-1], lon[0] > lon[-1]
    if flip_lat:
        lat = lat[::-1]
    if flip_lon:
        lon = lon[::-1]

    lat_t, lon_t = tgrid.latlon()
    pts = np.column_stack([lat_t.ravel(), lon_t.ravel()])
    # Clamp to the source bbox so the interpolator never sees out-of-range
    # coordinates (equivalent to nearest-neighbour extrapolation at the edge).
    pts[:, 0] = np.clip(pts[:, 0], lat[0], lat[-1])
    pts[:, 1] = np.clip(pts[:, 1], lon[0], lon[-1])

    def _one(arr2d):
        a = np.asarray(arr2d, dtype="float64")
        if flip_lat:
            a = a[::-1, :]
        if flip_lon:
            a = a[:, ::-1]
        # Cubic cannot handle NaN; fall back per-field when any is present.
        m = method
        if m == "cubic" and np.isnan(a).any():
            m = "linear"
        f = RegularGridInterpolator((lat, lon), a, method=m,
                                    bounds_error=False, fill_value=None)
        return f(pts).reshape(tgrid.shape)

    dims_extra = [d for d in da.dims if d not in ("lat", "lon", "y", "x")]
    if dims_extra:
        stacked = da.stack(_s=dims_extra)
        outs = [_one(stacked.isel(_s=i).values)
                for i in range(stacked.sizes["_s"])]
        data = np.stack(outs).reshape(
            [da.sizes[d] for d in dims_extra] + list(tgrid.shape))
        coords = {d: da[d].values for d in dims_extra}
    else:
        data = _one(da.values)
        coords = {}
    coords.update(y=tgrid.y, x=tgrid.x)
    return xr.DataArray(data, dims=dims_extra + ["y", "x"], coords=coords,
                        name=da.name, attrs=da.attrs)


def field_to_target(da: xr.DataArray, tgrid: TargetGrid,
                    method: str = "bilinear") -> xr.DataArray:
    """Regrid a lat/lon DataArray onto the target grid.

    method: 'bilinear' | 'cubic' | 'conservative' | 'nearest_s2d'
    """
    # Regular lat/lon source: use the tensor-product interpolator, which is the
    # only path that offers true cubic. xesmf has no cubic method at all.
    if method in ("bilinear", "cubic", "linear") and \
            da["lat"].ndim == 1 and da["lon"].ndim == 1:
        return _regular_grid_interp(
            da, tgrid, "cubic" if method == "cubic" else "linear")

    # Aggregating a FINE source onto a coarse target is the case that matters
    # for truth products (AORC is ~1 km, the target is 10 km, so one target
    # cell covers ~100 source pixels). The scipy fallback below approximates
    # "conservative" with NEAREST, which takes a single pixel instead of the
    # cell mean — a different physical quantity, and one that injects the
    # sub-grid variance the aggregation is supposed to remove. Do a real
    # block mean instead, which needs no xesmf.
    if method == "conservative" and not _HAVE_XESMF:
        return _block_mean_to_target(da, tgrid)

    tgt = _target_dataset(tgrid)
    if _HAVE_XESMF:
        import xesmf

        regridder = xesmf.Regridder(da, tgt, method, periodic=False,
                                    ignore_degenerate=True)
        out = regridder(da, keep_attrs=True)
        return out.assign_coords(y=("y", tgrid.y), x=("x", tgrid.x))

    warnings.warn(
        "xesmf not available; falling back to scipy griddata. "
        "Conservative regridding is approximated — install xesmf for "
        "area-weighted aggregation.",
        RuntimeWarning,
    )
    return _griddata_fallback(da, tgrid, method)


def _block_mean_to_target(da: xr.DataArray, tgrid: TargetGrid) -> xr.DataArray:
    """Area-weighted aggregation of a fine lat/lon field onto the target grid.

    Every source pixel is assigned to the target cell its CENTRE falls in, and
    each target cell takes the mean of its members. With ~100 source pixels per
    target cell this is within a fraction of a percent of true conservative
    regridding, and unlike the nearest-neighbour fallback it is the same
    quantity the coarse product reports: a cell mean.

    Source pixels outside the target grid are dropped; target cells that
    receive no pixel come back NaN rather than silently borrowing a neighbour.
    """
    from pyproj import Transformer

    lat = np.asarray(da["lat"].values)
    lon = np.asarray(da["lon"].values)
    if lat.ndim == 1:
        lon2, lat2 = np.meshgrid(lon, lat)
    else:
        lon2, lat2 = lon, lat

    tf = Transformer.from_crs("EPSG:4326", tgrid.crs, always_xy=True)
    xs, ys = tf.transform(lon2.ravel(), lat2.ravel())

    # Target cell edges from its centres (uniform spacing by construction).
    dx = float(tgrid.x[1] - tgrid.x[0])
    dy = float(tgrid.y[1] - tgrid.y[0])
    ny, nx = tgrid.shape
    j = np.floor((xs - (tgrid.x[0] - dx / 2)) / dx).astype(np.int64)
    i = np.floor((ys - (tgrid.y[0] - dy / 2)) / dy).astype(np.int64)
    ok = (i >= 0) & (i < ny) & (j >= 0) & (j < nx)
    flat = np.where(ok, i * nx + j, 0)

    vals = np.asarray(da.values)
    stack = vals.reshape(-1, lat2.size) if vals.ndim == 3 else vals.reshape(1, -1)
    out = np.full((stack.shape[0], ny * nx), np.nan, dtype="float32")
    for t in range(stack.shape[0]):
        v = stack[t]
        good = ok & np.isfinite(v)
        tot = np.bincount(flat[good], weights=v[good], minlength=ny * nx)
        cnt = np.bincount(flat[good], minlength=ny * nx)
        with np.errstate(invalid="ignore"):
            m = cnt > 0
            out[t, m] = (tot[m] / cnt[m]).astype("float32")
    out = out.reshape((-1, ny, nx))

    if vals.ndim == 3:
        return xr.DataArray(out, dims=("time", "y", "x"),
                            coords={"time": da["time"], "y": tgrid.y, "x": tgrid.x},
                            name=da.name, attrs=da.attrs)
    return xr.DataArray(out[0], dims=("y", "x"),
                        coords={"y": tgrid.y, "x": tgrid.x},
                        name=da.name, attrs=da.attrs)


def _griddata_fallback(da: xr.DataArray, tgrid: TargetGrid,
                       method: str) -> xr.DataArray:
    from scipy.interpolate import griddata

    lat_t, lon_t = tgrid.latlon()
    # Source cell centers as points.
    lat_s = np.asarray(da["lat"].values)
    lon_s = np.asarray(da["lon"].values)
    if lat_s.ndim == 1:
        lon_s, lat_s = np.meshgrid(lon_s, lat_s)
    pts = np.column_stack([lon_s.ravel(), lat_s.ravel()])
    interp = "nearest" if method in ("nearest_s2d", "conservative") else "linear"

    def _one(arr2d):
        vals = np.asarray(arr2d).ravel()
        out = griddata(pts, vals, (lon_t, lat_t), method=interp)
        if np.isnan(out).any():  # fill edges with nearest
            fil(out, pts, vals, lon_t, lat_t)
        return out

    dims_extra = [d for d in da.dims if d not in ("lat", "lon", "y", "x")]
    if dims_extra:
        stacked = da.stack(_s=dims_extra)
        outs = [_one(stacked.isel(_s=i).values) for i in range(stacked.sizes["_s"])]
        data = np.stack(outs).reshape(
            [da.sizes[d] for d in dims_extra] + list(tgrid.shape))
        coords = {d: da[d].values for d in dims_extra}
    else:
        data = _one(da.values)
        coords = {}
    coords.update(y=tgrid.y, x=tgrid.x)
    return xr.DataArray(data, dims=dims_extra + ["y", "x"], coords=coords,
                        name=da.name, attrs=da.attrs)


def fil(out, pts, vals, lon_t, lat_t):
    from scipy.interpolate import griddata

    mask = np.isnan(out)
    fill = griddata(pts, vals, (lon_t[mask], lat_t[mask]), method="nearest")
    out[mask] = fill


def raster_to_target(path: str, tgrid: TargetGrid,
                     resampling: str = "bilinear") -> xr.DataArray:
    """Reproject a GeoTIFF (DEM / land cover) onto the target grid."""
    import rioxarray  # noqa: F401
    from rasterio.enums import Resampling

    da = xr.open_dataarray(path, engine="rasterio").squeeze()
    da = da.rio.reproject(
        tgrid.crs,
        transform=_affine(tgrid),
        shape=tgrid.shape,
        resampling=getattr(Resampling, resampling),
    )
    return da.assign_coords(y=("y", tgrid.y), x=("x", tgrid.x))


def _affine(tgrid: TargetGrid):
    from rasterio.transform import from_origin

    res = tgrid.res_m
    x0 = tgrid.x[0] - res / 2
    y0 = tgrid.y[0] + res / 2
    return from_origin(x0, y0, res, res)


def coastal_distance(tgrid: TargetGrid, coastline_path: str | None = None,
                     clip_pad_deg: float = 8.0) -> xr.DataArray:
    """Distance (km) from each target cell to the nearest coastline.

    Matters more than it looks. For an inland high-relief domain (Colorado)
    this field is near-constant and contributes nothing — it measured exactly
    0.000 feature importance there. For a domain within a few hundred km of a
    coast (Austin is 200-350 km from the Gulf) marine-air intrusion is a
    leading control on sub-50 km temperature, and a constant-zero channel
    throws that signal away.

    The global coastline is clipped to the domain plus ``clip_pad_deg`` before
    projection: distances are computed against every retained segment, so
    keeping the whole planet's coastline would be both slow and pointless.
    Distances use shapely's vectorized path (shapely >= 2), not a per-cell
    Python loop.

    Returns a zero field with a warning if no coastline is supplied, so inland
    domains still run — but the warning is the point: check it before assuming
    the channel carries information.
    """
    zeros = xr.DataArray(
        np.zeros(tgrid.shape, dtype="float32"),
        dims=("y", "x"),
        coords=dict(y=tgrid.y, x=tgrid.x),
        name="coastal_dist",
    )
    if coastline_path is None:
        warnings.warn(
            "No coastline provided; coastal_dist = 0 (a DEAD input channel). "
            "Set sources.coastline.path in the data config for any domain "
            "within a few hundred km of a coast.", RuntimeWarning)
        return zeros
    try:
        import geopandas as gpd
        import shapely
    except Exception:  # noqa: BLE001
        warnings.warn("geopandas/shapely missing; coastal_dist = 0.",
                      RuntimeWarning)
        return zeros

    lat, lon = tgrid.latlon()
    bbox = (float(lon.min()) - clip_pad_deg, float(lat.min()) - clip_pad_deg,
            float(lon.max()) + clip_pad_deg, float(lat.max()) + clip_pad_deg)
    coast = gpd.read_file(coastline_path, bbox=bbox)
    if coast.empty:
        warnings.warn(
            f"No coastline within {clip_pad_deg} deg of the domain; "
            f"coastal_dist = 0.", RuntimeWarning)
        return zeros

    coast = coast.to_crs(tgrid.crs)
    line = coast.geometry.union_all()
    xx, yy = tgrid.meshgrid()
    pts = shapely.points(xx.ravel(), yy.ravel())
    dist = shapely.distance(pts, line).reshape(tgrid.shape) / 1000.0
    zeros.values = dist.astype("float32")
    print(f"[coastal] distance to coast: {dist.min():.0f}–{dist.max():.0f} km "
          f"across the domain")
    return zeros
