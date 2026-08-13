"""Download AORC (Analysis Of Record for Calibration) from the public S3 Zarr.

AORC is the ground truth: hourly, ~1 km, CONUS. The bucket holds ONE Zarr store
PER YEAR (``s3://noaa-nws-aorc-v1-1-1km/<year>.zarr``) — the root is not itself
a Zarr group — so each year is opened separately.

Each year is subset to the domain bbox, then **aggregated to a daily mean in
flight** and written as float32. That matters: the raw hourly window for a
4x4 deg domain is ~16 GB/year, the daily means ~0.3 GB/year, and the pipeline
only ever consumes the daily mean (which is also the statistic NASA POWER
reports as T2M). Pass --keep-hourly to skip the aggregation.

Cost warning: even writing daily means, the *reads* stream every hourly chunk
that intersects the domain — expect tens of GB of S3 traffic and hours of wall
clock for a multi-year pull. For a cheaper truth, set ``truth_source: prism``
in configs/data.yaml and use data/download_prism.py (4 km daily, ~GB).

Usage:
    python -m data.download_aorc --start 2019-01-01 --end 2023-12-31
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from common import load_config, subset_bbox


def open_aorc_year(bucket: str, year: int):
    """Open one yearly AORC Zarr store from anonymous S3."""
    import s3fs
    import xarray as xr

    fs = s3fs.S3FileSystem(anon=True)
    uri = f"{bucket.removeprefix('s3://').rstrip('/')}/{year}.zarr"
    return xr.open_zarr(s3fs.S3Map(uri, s3=fs), consolidated=True)


def _normalize(ds, variables: list[str]):
    """Rename latitude/longitude -> lat/lon and keep only what we need."""
    rename = {a: b for a, b in (("latitude", "lat"), ("longitude", "lon"))
              if a in ds.coords}
    if rename:
        ds = ds.rename(rename)
    return ds[variables]


def _covers(path: Path, lo, hi, keep_hourly: bool) -> bool:
    """True if an existing store already spans the requested window.

    A plain ``path.exists()`` check is a trap: a store written for a short test
    window looks identical to a full-year one, and the year would be silently
    skipped, leaving holes in the training set. Compare the actual time span.
    """
    import xarray as xr

    try:
        existing = xr.open_zarr(path, consolidated=True)
    except Exception as e:  # noqa: BLE001 - unreadable/partial store -> redo it
        print(f"[aorc] {path.name} unreadable ({e}); re-downloading")
        return False
    t = pd.DatetimeIndex(existing["time"].values)
    if len(t) == 0:
        return False
    # Daily means are stamped at 00:00, so the last day covers up to +1 day.
    tail = t.max() + (pd.Timedelta(0) if keep_hourly else pd.Timedelta("23h"))
    return t.min() <= lo and tail >= hi


def _strip_encoding(ds):
    """Drop encoding inherited from the source store before writing.

    The AORC stores are Zarr v2 and carry a ``numcodecs.Zstd`` compressor plus
    the source chunk grid in ``.encoding``. Passing that through to a Zarr v3
    writer raises ``TypeError: Expected a BytesBytesCodec``, and the stale
    chunk shapes fight our own rechunking. Clearing lets zarr pick defaults.
    """
    for var in list(ds.variables):
        ds[var].encoding = {}
    return ds


def main(start: str, end: str, keep_hourly: bool) -> None:
    cfg = load_config("data")
    src = cfg["sources"]["aorc"]
    out_dir = Path(src["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cfg["domain"]["name"]

    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    for year in range(t0.year, t1.year + 1):
        tag = "hourly" if keep_hourly else "daily"
        out_path = out_dir / f"aorc_{name}_{year}_{tag}.zarr"
        lo = max(t0, pd.Timestamp(f"{year}-01-01"))
        hi = min(t1, pd.Timestamp(f"{year}-12-31 23:00"))
        if out_path.exists():
            if _covers(out_path, lo, hi, keep_hourly):
                print(f"[aorc] {out_path.name} covers the window, skipping")
                continue
            print(f"[aorc] {out_path.name} exists but does NOT cover "
                  f"{lo.date()}..{hi.date()} — re-downloading")

        ds = _normalize(open_aorc_year(src["s3_uri"], year), src["variables"])
        ds = subset_bbox(ds, cfg["domain"])
        ds = ds.sel(time=slice(lo, hi))       # clip at the year boundaries
        if ds.sizes.get("time", 0) == 0:
            print(f"[aorc] {year}: no timesteps in window, skipping")
            continue

        if not keep_hourly:
            ds = ds.resample(time="1D").mean()
        ds = ds.astype("float32")

        ny, nx = ds.sizes["lat"], ds.sizes["lon"]
        print(f"[aorc] {year}: {ds.sizes['time']} steps x {ny}x{nx} "
              f"-> {out_path} (streaming from S3, slow)")
        # Write to a .partial store and rename on success, so an interrupted
        # or failed year never leaves a half-written store that the `exists`
        # check above would then happily skip forever.
        tmp_path = out_path.with_name(out_path.name + ".partial")
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        ds = _strip_encoding(ds)
        # Rechunk spatially too: the bbox subset slices mid-chunk, leaving
        # ragged dask chunks that Zarr rejects ("uniform chunk sizes except
        # for final chunk"). One chunk per field is ~1 MB at 480x480 float32.
        ds.chunk({"time": 32, "lat": -1, "lon": -1}).to_zarr(
            tmp_path, mode="w", consolidated=True)
        if out_path.exists():          # stale/short store being replaced
            shutil.rmtree(out_path)
        tmp_path.rename(out_path)
        print(f"[aorc] {year} done")
    print("[aorc] all years done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--keep-hourly", action="store_true",
                    help="write raw hourly instead of the daily mean (~50x bigger)")
    args = ap.parse_args()
    main(args.start, args.end, args.keep_hourly)
