"""Fetch static covariates: DEM (elevation) and land cover.

These are time-invariant, so they are pulled once for the domain. Both have
working no-credential defaults, verified live:

  * DEM       — Copernicus GLO-90 (90 m) from the anonymous AWS Open Data
                bucket ``s3://copernicus-dem-90m``, one COG per 1x1 deg tile.
                90 m is deliberate: the target cells are 10 km, so a 30 m DEM
                would be ~9x the bytes for structure that gets averaged away.
  * Landcover — NLCD 2021 CONUS via the MRLC GeoServer WMS. WMS returns a
                *paletted* raster (indices 1..21 + colormap), not class codes,
                so the palette RGB is matched back to the official NLCD legend
                and real class codes are written out.

Both accept a local GeoTIFF override, and the DEM can still come from
``py3dep`` (USGS 3DEP) if you have it installed and want 30 m CONUS data.

Coastal distance is derived analytically in regrid.py from the target grid and
a coastline mask, so it is not downloaded here.

Usage:
    python -m data.download_covariates                        # auto-fetch both
    python -m data.download_covariates --dem-source py3dep    # USGS 3DEP 30 m
    python -m data.download_covariates --dem my_dem.tif --landcover my_nlcd.tif
"""
from __future__ import annotations

import argparse
import math
import shutil
import warnings
from pathlib import Path

import numpy as np

from common import load_config

DEM_BUCKET = "copernicus-dem-90m"
DEM_TILE = "Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM"
MRLC_WMS = "https://www.mrlc.gov/geoserver/mrlc_display/wms"
MRLC_LAYER = "mrlc_display:NLCD_2021_Land_Cover_L48"
WMS_MAX_PX = 2048          # GeoServer's default per-request pixel cap

# Statics are fetched with a margin around the lat/lon domain. The target grid
# is built in EPSG:5070 and snapped OUTWARD to 10 km, and Albers rotates
# relative to lat/lon (~6 deg meridian convergence at 105W), so the projected
# grid's corners sit well outside the lat/lon bbox. Fetching exactly the bbox
# leaves ~18% of the target cells as NaN, which then poisons the input tensor.
COV_PAD = 0.75             # degrees

# Official NLCD legend: class code -> canonical RGB. The WMS renders these
# colors (within a couple of counts of rounding), so nearest-color matching
# recovers the class code from the palette index.
NLCD_LEGEND = {
    11: (70, 107, 159),    # open water
    12: (209, 222, 248),   # perennial ice/snow
    21: (222, 197, 197),   # developed, open space
    22: (217, 146, 130),   # developed, low intensity
    23: (235, 0, 0),       # developed, medium intensity
    24: (171, 0, 0),       # developed, high intensity
    31: (179, 172, 159),   # barren land
    41: (104, 171, 95),    # deciduous forest
    42: (28, 95, 44),      # evergreen forest
    43: (181, 197, 143),   # mixed forest
    52: (204, 184, 121),   # shrub/scrub
    71: (223, 223, 194),   # herbaceous
    81: (220, 217, 57),    # hay/pasture
    82: (171, 108, 40),    # cultivated crops
    90: (184, 217, 235),   # woody wetlands
    95: (108, 159, 184),   # emergent herbaceous wetlands
}


# --------------------------------------------------------------------------- #
# DEM
# --------------------------------------------------------------------------- #
def _padded(domain: dict, pad: float = COV_PAD) -> dict:
    return dict(lon_min=domain["lon_min"] - pad, lon_max=domain["lon_max"] + pad,
                lat_min=domain["lat_min"] - pad, lat_max=domain["lat_max"] + pad)


def _dem_tile_keys(domain: dict, pad: float = 0.1) -> list[str]:
    """1x1 deg Copernicus tile keys covering the domain (tiles name their SW corner)."""
    lat0 = math.floor(domain["lat_min"] - pad)
    lat1 = math.floor(domain["lat_max"] + pad)
    lon0 = math.floor(domain["lon_min"] - pad)
    lon1 = math.floor(domain["lon_max"] + pad)
    keys = []
    for lat in range(lat0, lat1 + 1):
        for lon in range(lon0, lon1 + 1):
            name = DEM_TILE.format(ns="N" if lat >= 0 else "S", lat=abs(lat),
                                   ew="E" if lon >= 0 else "W", lon=abs(lon))
            keys.append(f"{name}/{name}.tif")
    return keys


def fetch_dem_copernicus(domain: dict, out_dir: Path) -> Path:
    """Download + mosaic Copernicus GLO-90 tiles, clipped to the domain."""
    import rioxarray  # noqa: F401
    import s3fs
    import xarray as xr
    from rioxarray.merge import merge_arrays

    cache = out_dir / "tiles"
    cache.mkdir(parents=True, exist_ok=True)
    fs = s3fs.S3FileSystem(anon=True)

    domain = _padded(domain)
    keys = _dem_tile_keys(domain)
    print(f"[dem] {len(keys)} Copernicus GLO-90 tile(s) for the domain")
    local: list[Path] = []
    for key in keys:
        dest = cache / Path(key).name
        if not dest.exists():
            remote = f"{DEM_BUCKET}/{key}"
            if not fs.exists(remote):
                # Ocean-only 1-deg cells simply have no tile.
                warnings.warn(f"[dem] no tile {Path(key).name}; skipping",
                              RuntimeWarning)
                continue
            print(f"[dem] fetching {dest.name}")
            fs.get(remote, str(dest))
        local.append(dest)
    if not local:
        raise SystemExit("[dem] no Copernicus tiles found for this domain")

    arrays = [xr.open_dataarray(str(p), engine="rasterio").squeeze()
              for p in local]
    mosaic = merge_arrays(arrays) if len(arrays) > 1 else arrays[0]
    mosaic = mosaic.rio.clip_box(domain["lon_min"], domain["lat_min"],
                                 domain["lon_max"], domain["lat_max"])
    dest = out_dir / "dem.tif"
    mosaic.rio.to_raster(dest)
    print(f"[dem] wrote {dest} {tuple(mosaic.shape)} "
          f"(elevation in metres, EGM2008)")
    return dest


def fetch_dem_py3dep(domain: dict, out_dir: Path) -> Path:
    import py3dep

    bbox = (domain["lon_min"], domain["lat_min"],
            domain["lon_max"], domain["lat_max"])
    print("[dem] fetching 3DEP elevation via py3dep ...")
    dem = py3dep.get_map("DEM", bbox, resolution=30, geo_crs="EPSG:4326")
    dest = out_dir / "dem.tif"
    dem.rio.to_raster(dest)
    print(f"[dem] wrote {dest}")
    return dest


def fetch_dem(domain: dict, out_dir: Path, local: str | None,
              source: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "dem.tif"
    if local:
        if Path(local).resolve() == dest.resolve():
            print(f"[dem] {dest} already in place, keeping it")
            return dest
        shutil.copy(local, dest)
        print(f"[dem] copied {local} -> {dest}")
        return dest
    if source == "py3dep":
        return fetch_dem_py3dep(domain, out_dir)
    return fetch_dem_copernicus(domain, out_dir)


# --------------------------------------------------------------------------- #
# Land cover
# --------------------------------------------------------------------------- #
def _palette_to_nlcd(colormap: dict) -> dict[int, int]:
    """Map each palette index to the nearest official NLCD class by RGB."""
    codes = np.array(list(NLCD_LEGEND.keys()))
    ref = np.array([NLCD_LEGEND[c] for c in codes], dtype="int32")
    out = {}
    for idx, rgba in colormap.items():
        rgb = np.array(rgba[:3], dtype="int32")
        d = np.abs(ref - rgb).sum(axis=1)
        j = int(d.argmin())
        if d[j] > 30:          # not a legend color (background/nodata entry)
            continue
        out[int(idx)] = int(codes[j])
    return out


def _wms_tile(box: dict, out_path: Path, retries: int = 3) -> Path | None:
    """One WMS GetMap tile, palette converted to real NLCD class codes."""
    import time
    import urllib.request

    import rasterio

    span_lon = box["lon_max"] - box["lon_min"]
    span_lat = box["lat_max"] - box["lat_min"]
    scale = WMS_MAX_PX / max(span_lon, span_lat)
    width = min(WMS_MAX_PX, max(64, int(round(span_lon * scale))))
    height = min(WMS_MAX_PX, max(64, int(round(span_lat * scale))))
    url = (f"{MRLC_WMS}?service=WMS&version=1.1.1&request=GetMap"
           f"&layers={MRLC_LAYER}&srs=EPSG:4326"
           f"&bbox={box['lon_min']},{box['lat_min']},"
           f"{box['lon_max']},{box['lat_max']}"
           f"&width={width}&height={height}&format=image/geotiff")

    raw = out_path.with_suffix(".palette.tif")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:
                raw.write_bytes(resp.read())
            break
        except Exception as e:  # noqa: BLE001, PERF203
            print(f"[landcover] tile {out_path.name} attempt {attempt}/{retries}"
                  f" failed: {type(e).__name__}")
            if attempt == retries:
                return None
            time.sleep(5 * attempt)

    with rasterio.open(raw) as src:
        idx = src.read(1)
        profile = src.profile
        lut = _palette_to_nlcd(src.colormap(1))
    codes = np.zeros_like(idx, dtype="uint8")
    for palette_idx, nlcd in lut.items():
        codes[idx == palette_idx] = nlcd
    profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    profile.pop("photometric", None)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(codes, 1)
    raw.unlink(missing_ok=True)
    return out_path


def fetch_landcover_mrlc(domain: dict, out_dir: Path,
                         max_span_deg: float = 4.0) -> Path:
    """Pull NLCD 2021 from the MRLC WMS and convert palette -> class codes.

    Tiled: the service drops the connection on large requests (an 11.5 x 9.5
    deg box returns ``RemoteDisconnected``), and a single 2048 px image over a
    domain that size would be ~620 m/px anyway — coarse enough to lose the
    urban fraction this channel exists to measure. Tiling keeps each request
    small enough to succeed AND raises the effective resolution.
    """
    import rioxarray  # noqa: F401
    import xarray as xr
    from rioxarray.merge import merge_arrays

    dom = _padded(domain)
    tile_dir = out_dir / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    def _edges(lo, hi):
        n = max(1, int(np.ceil((hi - lo) / max_span_deg)))
        step = (hi - lo) / n
        return [(lo + i * step, lo + (i + 1) * step) for i in range(n)]

    boxes = [dict(lon_min=a, lon_max=b, lat_min=c, lat_max=d)
             for a, b in _edges(dom["lon_min"], dom["lon_max"])
             for c, d in _edges(dom["lat_min"], dom["lat_max"])]
    print(f"[landcover] {len(boxes)} WMS tile(s) for the domain")

    paths = []
    for i, box in enumerate(boxes):
        p = tile_dir / f"nlcd_t{i:02d}.tif"
        if not p.exists() and _wms_tile(box, p) is None:
            raise SystemExit(f"[landcover] tile {i} failed after retries")
        paths.append(p)

    arrays = []
    for p in paths:
        a = xr.open_dataarray(str(p), engine="rasterio").squeeze()
        # Clear nodata BEFORE merging. The tiles are written with nodata=0
        # (0 == "outside the NLCD legend"), and merge_arrays treats nodata as
        # "nothing here" — with every tile declaring 0 as nodata the mosaic
        # comes back 100% empty. Merging with nodata unset keeps real class
        # codes; genuine 0s are still 0 afterwards.
        a.rio.write_nodata(None, inplace=True)
        arrays.append(a)
    mosaic = merge_arrays(arrays) if len(arrays) > 1 else arrays[0]
    dest = out_dir / "landcover.tif"
    mosaic.rio.to_raster(dest, dtype="uint8", compress="deflate")

    codes = mosaic.values
    unmapped = int((codes == 0).sum())
    if unmapped:
        print(f"[landcover] {unmapped} px ({100 * unmapped / codes.size:.2f}%) "
              f"outside the legend -> 0 (water/nodata)")
    present = sorted(int(v) for v in np.unique(codes) if v)
    res_m = (dom["lon_max"] - dom["lon_min"]) / mosaic.shape[-1] * 111_000
    print(f"[landcover] wrote {dest} {tuple(mosaic.shape)} (~{res_m:.0f} m/px); "
          f"NLCD classes present: {present}")
    return dest


def fetch_landcover(domain: dict, out_dir: Path, local: str | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "landcover.tif"
    if local:
        shutil.copy(local, dest)
        print(f"[landcover] copied {local} -> {dest}")
        return dest
    return fetch_landcover_mrlc(domain, out_dir)


# --------------------------------------------------------------------------- #
def main(dem: str | None, landcover: str | None, dem_source: str) -> None:
    cfg = load_config("data")
    fetch_dem(cfg["domain"], Path(cfg["sources"]["dem"]["out_dir"]), dem,
              dem_source)
    fetch_landcover(cfg["domain"],
                    Path(cfg["sources"]["landcover"]["out_dir"]), landcover)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", default=None, help="local DEM GeoTIFF (skips fetch)")
    ap.add_argument("--landcover", default=None,
                    help="local NLCD GeoTIFF (skips fetch)")
    ap.add_argument("--dem-source", default="copernicus",
                    choices=["copernicus", "py3dep"],
                    help="auto-fetch backend when --dem is not given")
    args = ap.parse_args()
    main(args.dem, args.landcover, args.dem_source)
