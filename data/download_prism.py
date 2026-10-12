"""Download PRISM daily grids via the NACSE web service.

PRISM is an alternative truth source (daily ``tmean``), used as an independent
judge: it is produced by a different group, from different station data, by a
different method than either ERA5-Land or AORC.

THE ENDPOINT MOVED. The old /data/public/<res>/<var>/<YYYYMMDD> path now
returns a 200 with an HTML notice instead of a zip, so the previous code failed
with BadZipFile on the first day rather than with anything that named the
cause. The current form carries a region segment:

    https://services.nacse.org/prism/data/get/<region>/<res>/<var>/<YYYYMMDD>

Both 4km and 800m are on the public service as of this writing — 2.3 MB and
41.5 MB per day respectively for CONUS. 800m matters at a 5 km target: 4 km
source into 5 km cells is only ~1.6 source cells per target cell, which is
barely an aggregation, while 800m gives ~39 and is a real one.

``tmean`` is (tmin + tmax) / 2, NOT the mean of the diurnal cycle that POWER
T2M and AORC daily means report, so PRISM carries a known systematic offset
against both. Use it for structure and for ranking, never for absolute bias.

Usage:
    python -m data.download_prism --start 2019-01-01 --end 2019-01-31
    python -m data.download_prism --start 2019-01-01 --end 2023-12-31 --res 800m
"""
from __future__ import annotations

import argparse
import io
import time
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

from common import load_config

BASE = "https://services.nacse.org/prism/data/get"
REGION = "us"


def _daterange(start: str, end: str):
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def download_day(var: str, res: str, day: date, out_dir: Path) -> Path | None:
    """One day's zipped BIL grid. Returns the marker path, or None on failure."""
    stamp = day.strftime("%Y%m%d")
    dest_dir = out_dir / var
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / f"{stamp}.done"
    if marker.exists():
        return marker

    url = f"{BASE}/{REGION}/{res}/{var}/{stamp}"
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            payload = resp.read()
    except Exception as e:  # noqa: BLE001 - keep going on transient failures
        print(f"[prism] {var} {stamp} FAILED: {e}")
        return None

    if not payload[:2] == b"PK":
        head = payload[:200].decode("utf-8", "replace").strip().replace("\n", " ")
        print(f"[prism] {var} {stamp} did not return a zip ({len(payload)} B). "
              f"The service said: {head}")
        return None

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(dest_dir / stamp)
    marker.write_text("ok")
    print(f"[prism] {var} {stamp} ok")
    return marker


def main(start: str, end: str, sleep: float = 1.0, res: str | None = None) -> None:
    cfg = load_config("data")
    src = cfg["sources"]["prism"]
    out_dir = Path(src["out_dir"]) / (res or src["resolution"])
    res = res or src["resolution"]
    print(f"[prism] {res} {src['variables']} -> {out_dir}")
    for day in _daterange(start, end):
        for var in src["variables"]:
            download_day(var, res, day, out_dir)
            time.sleep(sleep)  # be polite to the free service


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--res", default=None,
                    help="4km | 800m; overrides sources.prism.resolution")
    args = ap.parse_args()
    main(args.start, args.end, args.sleep, args.res)
