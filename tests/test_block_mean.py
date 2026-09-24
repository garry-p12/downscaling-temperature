"""Block-mean aggregation for fine -> coarse regridding.

This replaces a fallback that silently approximated 'conservative' with
NEAREST. For a 1 km truth product on a 10 km grid that took one pixel out of
~100 instead of the cell mean — a different physical quantity, and the reason
README listed AORC-as-truth as blocked.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from common.grid import build_target_grid
from data.regrid import _block_mean_to_target, field_to_target


@pytest.fixture
def tgrid():
    from common import load_config
    cfg = load_config("configs/data_southcentral.yaml")
    return build_target_grid(cfg["domain"], cfg["grid"])


def _fine(tgrid, fn, n=400):
    """A fine lat/lon field covering the target grid, valued by fn(lat, lon)."""
    LAT, LON = tgrid.latlon()
    lat = np.linspace(LAT.min() - .1, LAT.max() + .1, n)
    lon = np.linspace(LON.min() - .1, LON.max() + .1, n)
    lo, la = np.meshgrid(lon, lat)
    return xr.DataArray(fn(la, lo), dims=("lat", "lon"),
                        coords={"lat": lat, "lon": lon}, name="t")


def test_constant_field_survives_exactly(tgrid):
    """The mean of a constant is that constant — the weakest possible check,
    and the one that would have caught the nearest-neighbour fallback only if
    the field were NOT constant. Kept as a floor."""
    out = _block_mean_to_target(_fine(tgrid, lambda a, o: np.full_like(a, 7.5)), tgrid)
    v = out.values[np.isfinite(out.values)]
    assert np.allclose(v, 7.5, atol=1e-4)


def test_it_averages_rather_than_samples(tgrid):
    """The point of the fix. Give the fine field a checkerboard that averages
    to a known constant within any cell; a sampling scheme returns the extremes,
    an averaging one returns the mean."""
    def checker(la, lo):
        i = (np.arange(la.shape[0])[:, None] // 1) % 2
        j = (np.arange(la.shape[1])[None, :] // 1) % 2
        return np.where((i + j) % 2 == 0, 0.0, 10.0)
    out = _block_mean_to_target(_fine(tgrid, checker, n=600), tgrid).values
    v = out[np.isfinite(out)]
    assert v.min() > 1.0, "looks like point sampling: some cell returned ~0"
    assert v.max() < 9.0, "looks like point sampling: some cell returned ~10"
    assert abs(float(np.mean(v)) - 5.0) < 0.6


def test_a_linear_ramp_is_reproduced(tgrid):
    """A smooth field should come back close to its value at the cell centre."""
    out = _block_mean_to_target(_fine(tgrid, lambda a, o: a), tgrid)
    LAT, _ = tgrid.latlon()
    d = out.values - LAT
    assert np.nanmax(np.abs(d)) < 0.05


def test_empty_cells_are_nan_not_borrowed(tgrid):
    """A cell with no source pixel must be NaN. Quietly borrowing a neighbour
    is how the NLCD nodata bug produced a city in the desert."""
    LAT, LON = tgrid.latlon()
    lat = np.linspace(LAT.min(), LAT.min() + 0.3, 40)      # covers a sliver only
    lon = np.linspace(LON.min(), LON.min() + 0.3, 40)
    lo, la = np.meshgrid(lon, lat)
    da = xr.DataArray(np.ones_like(la), dims=("lat", "lon"),
                      coords={"lat": lat, "lon": lon}, name="t")
    out = _block_mean_to_target(da, tgrid).values
    assert np.isnan(out).any(), "no cell left empty — coverage check is not working"
    assert np.allclose(out[np.isfinite(out)], 1.0)


def test_time_dimension_is_preserved(tgrid):
    base = _fine(tgrid, lambda a, o: a, n=200)
    stack = xr.concat([base + k for k in range(3)], dim="time")
    stack = stack.assign_coords(time=np.arange(3))
    out = _block_mean_to_target(stack, tgrid)
    assert out.dims == ("time", "y", "x") and out.sizes["time"] == 3
    a, b = out.isel(time=0).values, out.isel(time=1).values
    m = np.isfinite(a) & np.isfinite(b)
    assert np.allclose(b[m] - a[m], 1.0, atol=1e-4)


def test_field_to_target_routes_conservative_here(tgrid):
    """The routing is the fix: field_to_target(..., 'conservative') must no
    longer fall through to the nearest-neighbour approximation."""
    da = _fine(tgrid, lambda a, o: a, n=200)
    direct = _block_mean_to_target(da, tgrid).values
    routed = field_to_target(da, tgrid, "conservative").values
    m = np.isfinite(direct) & np.isfinite(routed)
    assert m.sum() > 0 and np.allclose(direct[m], routed[m], atol=1e-5)
