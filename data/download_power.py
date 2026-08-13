"""Download the coarse predictor (NASA POWER daily T2M) — no credentials.

POWER serves MERRA-2-derived daily fields at the MERRA-2 native spacing —
0.5 deg lat x 0.625 deg lon, ~55 x ~50 km at mid-latitudes — which is the
coarse side of this pipeline's 50 km -> 10 km task. T2M arrives in degC with
``cell_methods = "time: mean"``, i.e. the daily mean of the diurnal cycle, the
same statistic AORC-hourly-averaged truth provides. The regional endpoint is
used:

    https://power.larc.nasa.gov/api/temporal/daily/regional
        ?parameters=T2M&community=RE&format=netcdf&time-standard=UTC
        &latitude-min=..&latitude-max=..&longitude-min=..&longitude-max=..
        &start=YYYYMMDD&end=YYYYMMDD

Service limits handled here:
  * one parameter per request  -> we only need T2M;
  * bounding box max ~4.5 deg per side (and it rejects very small boxes)
    -> the padded domain is tiled into <= ``MAX_SPAN`` deg boxes, each grown to
       at least ``MIN_SPAN`` deg;
  * long windows time out      -> requests are chunked one calendar year each.

Verified against a live 2 deg x 2 deg / 15-day response: HTTP 200, netCDF4 with
``T2M(time, lat, lon)`` in degC, ``lat``/``lon`` coords already named, missing
values encoded as NaN.

Files land as one netCDF per (tile, year) under ``sources.power.out_dir``;
``open_power`` stitches them back into a single (time, lat, lon) Dataset.

Usage:
    python -m data.download_power --years 2019 2020 2021 2022 2023
"""
from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import numpy as np

from common import load_config

BASE = "https://power.larc.nasa.gov/api/temporal/daily/regional"
MAX_SPAN = 4.0      # deg; service cap is 4.5 -> stay under it
MIN_SPAN = 2.0      # deg; service rejects boxes below ~2 deg per side
PAD = 1.0           # deg; bracket the target domain so interpolation is interior
# netCDF responses encode missing as NaN, but POWER's other formats (and older
# versions) use the -999 sentinel; screen for it defensively on read.
FILL = -999.0


# --------------------------------------------------------------------------- #
# Tiling
# --------------------------------------------------------------------------- #
def _edges(lo: float, hi: float) -> list[tuple[float, float]]:
    """Split [lo, hi] into <= MAX_SPAN chunks, each at least MIN_SPAN wide."""
    span = hi - lo
    n = max(1, int(np.ceil(span / MAX_SPAN)))
    step = span / n
    out = []
    for i in range(n):
        a, b = lo + i * step, lo + (i + 1) * step
        if b - a < MIN_SPAN:                     # grow around the center
            c = 0.5 * (a + b)
            a, b = c - MIN_SPAN / 2, c + MIN_SPAN / 2
        out.append((round(a, 4), round(b, 4)))
    return out


def tiles(domain: dict, pad: float = PAD) -> list[dict]:
    lat_lo, lat_hi = domain["lat_min"] - pad, domain["lat_max"] + pad
    lon_lo, lon_hi = domain["lon_min"] - pad, domain["lon_max"] + pad
    return [
        dict(lat_min=a, lat_max=b, lon_min=c, lon_max=d)
        for a, b in _edges(lat_lo, lat_hi)
        for c, d in _edges(lon_lo, lon_hi)
    ]


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
def _url(box: dict, year: int, parameter: str, community: str) -> str:
    q = {
        "parameters": parameter,
        "community": community,
        "format": "netcdf",
        "time-standard": "UTC",
        "latitude-min": box["lat_min"],
        "latitude-max": box["lat_max"],
        "longitude-min": box["lon_min"],
        "longitude-max": box["lon_max"],
        "start": f"{year}0101",
        "end": f"{year}1231",
    }
    return BASE + "?" + "&".join(f"{k}={v}" for k, v in q.items())


def fetch(box: dict, year: int, parameter: str, community: str, dest: Path,
          retries: int = 3, backoff: float = 5.0) -> Path | None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[power] {dest.name} exists, skipping")
        return dest
    url = _url(box, year, parameter, community)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=600) as resp:
                payload = resp.read()
            dest.write_bytes(payload)
            print(f"[power] {dest.name} ok ({len(payload)/1e6:.1f} MB)")
            return dest
        except (urllib.error.URLError, TimeoutError) as e:  # noqa: PERF203
            print(f"[power] {dest.name} attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(backoff * attempt)
    print(f"[power] GIVING UP on {dest.name}\n         url: {url}")
    return None


# --------------------------------------------------------------------------- #
# Read back
# --------------------------------------------------------------------------- #
def open_power(cfg, parameter: str | None = None):
    """Stitch the per-(tile, year) netCDFs into one (time, lat, lon) Dataset.

    Tiles are mosaicked per year with ``combine_first`` (each tile fills the
    cells the previous ones left empty; MIN_SPAN overlaps resolve first-wins),
    then concatenated along time.

    NOT ``xr.merge(..., compat='override')``: every tile carries the SAME
    variable name, so merge resolves the conflict by keeping one tile's array
    and aligning it to the union index — leaving ~70% NaN with no error.
    """
    import functools

    import xarray as xr

    src = cfg["sources"]["power"]
    parameter = parameter or src["parameter"]
    files = sorted(Path(src["out_dir"]).glob(f"power_{parameter}_*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No POWER files in {src['out_dir']}. Run: "
            f"python -m data.download_power --years ...")

    # Filenames end in ..._<year>_t<NN>; the domain name may contain '_'.
    by_year: dict[str, list] = {}
    for f in files:
        by_year.setdefault(f.stem.split("_")[-2], []).append(f)

    years = []
    for year in sorted(by_year):
        parts = [xr.open_dataset(f) for f in by_year[year]]
        mosaic = functools.reduce(lambda a, b: a.combine_first(b), parts)
        years.append(mosaic)
    ds = xr.concat(years, dim="time") if len(years) > 1 else years[0]

    rename = {a: b for a, b in (("latitude", "lat"), ("longitude", "lon"))
              if a in ds.coords}
    if rename:
        ds = ds.rename(rename)
    ds = ds.sortby("lat").sortby("lon").sortby("time")
    ds = ds.where(ds > FILL / 2)            # -999 sentinel -> NaN (NaN stays NaN)

    # A gap here means the tiles did not actually tile the domain. Say so —
    # a mostly-NaN coarse channel is otherwise invisible until training.
    frac = float(np.isnan(ds[parameter].values).mean())
    if frac > 0:
        warnings.warn(
            f"[power] {100 * frac:.1f}% of the stitched {parameter} field is "
            f"NaN — tile coverage is incomplete for this domain.",
            RuntimeWarning)
    return ds


# --------------------------------------------------------------------------- #
def main(years: list[int]) -> None:
    cfg = load_config("data")
    src = cfg["sources"]["power"]
    out_dir = Path(src["out_dir"])
    name = cfg["domain"]["name"]
    parameter = src["parameter"]
    community = src.get("community", "RE")

    boxes = tiles(cfg["domain"], src.get("pad_deg", PAD))
    print(f"[power] {len(boxes)} tile(s) x {len(years)} year(s) "
          f"= {len(boxes) * len(years)} requests")
    missing = 0
    for year in years:
        for i, box in enumerate(boxes):
            dest = out_dir / f"power_{parameter}_{name}_{year}_t{i:02d}.nc"
            if fetch(box, year, parameter, community, dest) is None:
                missing += 1
    if missing:
        raise SystemExit(f"[power] {missing} request(s) failed — re-run to retry")
    print("[power] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", required=True)
    args = ap.parse_args()
    main(args.years)
