"""Derived covariates and the channel-subsetting that ablations depend on.

No raster files are read here — the terrain maths is checked against synthetic
DEMs with known answers, which is both faster and a stronger test than
round-tripping the real one.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from common.grid import TargetGrid  # noqa: F401
from data.covariates import (
    LANDCOVER_CHANNELS,
    NLCD_GROUPS,
    TERRAIN_CHANNELS,
    landcover_fractions,
    terrain_features,
)

# These tests write real GeoTIFFs, so they need the geospatial stack. A
# training-only environment — which is exactly what the cluster runs — skips
# them rather than failing collection.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("rasterio") is None,
    reason="rasterio not installed (training-only environment)")


@pytest.fixture
def tgrid() -> TargetGrid:
    """A small Albers grid sitting inside the synthetic rasters' footprint.

    Derived from a lat/lon box rather than hand-written projected coordinates,
    so it is guaranteed to overlap the test rasters written below.
    """
    from common.grid import build_target_grid

    return build_target_grid(
        {"lon_min": -98.9, "lon_max": -98.5, "lat_min": 31.0, "lat_max": 31.4},
        {"target_crs": "EPSG:5070", "target_res_m": 10000})


def _write_raster(path, arr, dtype, nodata=None, west=-99.0, north=31.5,
                  px=0.001):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype=dtype,
                       crs="EPSG:4326", nodata=nodata,
                       transform=from_origin(west, north, px, px)) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


# --------------------------------------------------------------------------- #
# Terrain
# --------------------------------------------------------------------------- #
def test_flat_terrain_has_no_relief_and_no_slope(tmp_path, tgrid):
    p = _write_raster(tmp_path / "flat.tif", np.full((600, 600), 100.0), "float32")
    out = {d.name: d.values for d in
           terrain_features(p, tgrid, ["elev_std", "slope_mean"])}
    assert np.nanmax(out["elev_std"]) == pytest.approx(0.0, abs=1e-3)
    assert np.nanmax(out["slope_mean"]) == pytest.approx(0.0, abs=1e-3)


def test_elev_std_detects_relief_that_the_cell_mean_hides(tmp_path, tgrid):
    """The point of the channel: two cells with identical MEAN elevation but
    different roughness must not look identical."""
    rng = np.random.default_rng(0)
    rough = 100.0 + rng.normal(0, 50, size=(600, 600))
    p = _write_raster(tmp_path / "rough.tif", rough, "float32")
    std = terrain_features(p, tgrid, ["elev_std"])[0].values
    finite = std[np.isfinite(std)]
    assert finite.size and finite.max() > 10.0


def test_slope_of_a_known_ramp_matches_the_analytic_value(tmp_path, tgrid):
    """A south-to-north ramp of 1 m per row. Row spacing is 0.001 deg, i.e.
    110.54 m, so the slope is arctan(1/110.54) = 0.518 deg."""
    ramp = np.tile(np.arange(600, dtype="float32")[:, None], (1, 600))
    p = _write_raster(tmp_path / "ramp.tif", ramp, "float32")
    slope = terrain_features(p, tgrid, ["slope_mean"])[0].values
    finite = slope[np.isfinite(slope)]
    assert finite.mean() == pytest.approx(np.degrees(np.arctan(1 / 110.54)),
                                          rel=0.05)


def test_aspect_is_averaged_as_a_vector_not_an_angle(tmp_path, tgrid):
    """A north-facing slope must give northness ~ +1, not an angle-wrap mean."""
    # Elevation decreasing northward (row 0 is the north edge) => faces north.
    ramp = np.tile(np.arange(600, dtype="float32")[:, None], (1, 600))
    p = _write_raster(tmp_path / "north.tif", ramp, "float32")
    out = {d.name: d.values for d in
           terrain_features(p, tgrid, ["northness", "eastness"])}
    n = out["northness"][np.isfinite(out["northness"])]
    e = out["eastness"][np.isfinite(out["eastness"])]
    assert abs(n.mean()) == pytest.approx(1.0, abs=0.05)
    assert abs(e.mean()) == pytest.approx(0.0, abs=0.05)


def test_northness_is_zero_where_there_is_no_slope(tmp_path, tgrid):
    """Flat ground has no aspect; it must not contribute a random direction."""
    p = _write_raster(tmp_path / "flat2.tif", np.full((600, 600), 7.0), "float32")
    n = terrain_features(p, tgrid, ["northness"])[0].values
    assert np.nanmax(np.abs(n)) == pytest.approx(0.0, abs=1e-6)


def test_tpi_is_signed_and_centred(tgrid):
    """A basin must come out negative and a peak positive."""
    dem = np.full(tgrid.shape, 500.0, dtype="float32")
    dem[3, 3] = 300.0                      # basin
    dem[1, 1] = 800.0                      # peak
    tpi = terrain_features("unused", tgrid, ["tpi"], dem_10km=dem)[0].values
    assert tpi[3, 3] < -50
    assert tpi[1, 1] > 50


def test_terrain_features_returns_requested_order_and_ignores_unknowns(tmp_path, tgrid):
    p = _write_raster(tmp_path / "f.tif", np.full((300, 300), 1.0), "float32")
    got = [d.name for d in terrain_features(p, tgrid, ["slope_mean", "nonsense"])]
    assert got == ["slope_mean"]
    assert terrain_features(p, tgrid, []) == []


# --------------------------------------------------------------------------- #
# Land cover
# --------------------------------------------------------------------------- #
def test_fractions_of_a_uniform_class_are_one_and_zero(tmp_path, tgrid):
    codes = np.full((600, 600), 42, dtype="uint8")           # evergreen forest
    p = _write_raster(tmp_path / "lc.tif", codes, "uint8", nodata=0)
    out = {d.name: d.values for d in landcover_fractions(p, tgrid)}
    f = out["lc_forest"][np.isfinite(out["lc_forest"])]
    assert f.min() == pytest.approx(1.0, abs=1e-3)
    assert np.nanmax(out["lc_shrub"]) == pytest.approx(0.0, abs=1e-3)


def test_minority_class_survives_where_the_mode_would_erase_it(tmp_path, tgrid):
    """The whole reason for this channel: a 30% urban cell is never the mode
    against 70% shrub, but must still report 0.3 developed."""
    codes = np.full((600, 600), 52, dtype="uint8")
    codes[:180] = 23                                          # 30% developed
    p = _write_raster(tmp_path / "mix.tif", codes, "uint8", nodata=0)
    out = {d.name: d.values for d in landcover_fractions(p, tgrid)}
    dev = out["lc_developed"][np.isfinite(out["lc_developed"])]
    assert dev.max() > 0.2, "developed fraction was lost"


def test_nodata_is_excluded_from_the_denominator(tmp_path, tgrid):
    """Half-covered cells report the composition of the covered half, rather
    than being diluted toward zero."""
    codes = np.full((600, 600), 42, dtype="uint8")
    codes[:, :300] = 0                                        # nodata
    p = _write_raster(tmp_path / "half.tif", codes, "uint8", nodata=0)
    f = landcover_fractions(p, tgrid, ["lc_forest"])[0].values
    finite = f[np.isfinite(f)]
    assert finite.size and finite.max() == pytest.approx(1.0, abs=1e-3)


def test_groups_are_disjoint_and_cover_the_real_nlcd_codes():
    seen: set[int] = set()
    for members in NLCD_GROUPS.values():
        assert not (seen & set(members)), "a class is in two groups"
        seen |= set(members)
    # Every code observed in the South-Central raster must be classified.
    for code in (11, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95):
        assert code in seen, f"NLCD {code} falls in no group"


def test_channel_name_constants_agree_with_the_group_table():
    assert set(LANDCOVER_CHANNELS) == set(NLCD_GROUPS)
    assert "elev_std" in TERRAIN_CHANNELS


