"""Target-grid construction and bbox subsetting."""
from __future__ import annotations

import numpy as np
import xarray as xr

from common.config import load_config
from common.grid import build_target_grid, subset_bbox

GRID = {"target_crs": "EPSG:5070", "target_res_m": 10000}


def test_grid_spacing_and_orientation():
    dom = {"lon_min": -107.0, "lon_max": -103.0, "lat_min": 37.5, "lat_max": 41.5}
    g = build_target_grid(dom, GRID)
    assert np.allclose(np.diff(g.x), 10000)
    # North-up: y descends, which is what the raster writers assume.
    assert np.all(np.diff(g.y) < 0)
    assert g.shape == (g.y.size, g.x.size)


def test_grid_contains_the_requested_domain():
    """Snapping is outward, so every requested corner must fall inside."""
    dom = {"lon_min": -104.0, "lon_max": -94.0, "lat_min": 27.0, "lat_max": 35.0}
    g = build_target_grid(dom, GRID)
    lat, lon = g.latlon()
    assert lon.min() <= dom["lon_min"] and lon.max() >= dom["lon_max"]
    assert lat.min() <= dom["lat_min"] and lat.max() >= dom["lat_max"]


def test_shipped_colorado_domain_is_49x40():
    """The odd grid the padding logic in models/model.py exists for."""
    cfg = load_config("configs/data.yaml")
    assert build_target_grid(cfg["domain"], cfg["grid"]).shape == (49, 40)


def _da(lats, lons):
    return xr.DataArray(
        np.arange(len(lats) * len(lons), dtype="float32").reshape(len(lats), len(lons)),
        coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))


def test_subset_bbox_ascending_latitude():
    da = _da(np.arange(20.0, 45.0), np.arange(-120.0, -90.0))
    dom = {"lon_min": -105.0, "lon_max": -100.0, "lat_min": 30.0, "lat_max": 35.0}
    out = subset_bbox(da, dom)
    assert out["lat"].min() >= 30.0 and out["lat"].max() <= 35.0
    assert out["lon"].min() >= -105.0 and out["lon"].max() <= -100.0


def test_subset_bbox_descending_latitude_keeps_the_same_cells():
    """ERA5 ships north-to-south; the slice direction has to follow."""
    asc = _da(np.arange(20.0, 45.0), np.arange(-120.0, -90.0))
    dom = {"lon_min": -105.0, "lon_max": -100.0, "lat_min": 30.0, "lat_max": 35.0}
    desc = asc.isel(lat=slice(None, None, -1))
    a = np.sort(subset_bbox(asc, dom)["lat"].values)
    d = np.sort(subset_bbox(desc, dom)["lat"].values)
    assert np.array_equal(a, d)


def test_subset_bbox_handles_0_360_longitudes():
    da = _da(np.arange(20.0, 45.0), np.arange(230.0, 280.0))
    dom = {"lon_min": -105.0, "lon_max": -100.0, "lat_min": 30.0, "lat_max": 35.0}
    out = subset_bbox(da, dom)
    assert out.sizes["lon"] > 0
    assert out["lon"].min() >= 255.0 and out["lon"].max() <= 260.0
