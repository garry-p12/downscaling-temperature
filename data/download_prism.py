"""Download PRISM daily grids via the NACSE web service.

PRISM is an alternative truth source (daily 4 km ``tmean``), used as a
cross-check against ERA5-Land. The web service
returns a zipped BIL grid per variable per day:
    https://services.nacse.org/prism/data/public/<res>/<var>/<YYYYMMDD>

Note: the free service rate-limits to ~2 downloads/day per IP for the same
file; for bulk pulls use the FTP mirror. This script is fine for a prototype
region/window.

Usage:
    python -m data.download_prism --start 2019-01-01 --end 2019-01-31
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

BASE = "https://services.nacse.org/prism/data/public"


def _daterange(start: str, end: str):
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def download_day(var: str, res: str, day: date, out_dir: Path) -> Path | None:
    stamp = day.strftime("%Y%m%d")
    dest_dir = out_dir / var
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / f"{stamp}.done"
    if marker.exists():
        return marker

    url = f"{BASE}/{res}/{var}/{stamp}"
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            payload = resp.read()
    except Exception as e:  # noqa: BLE001 - keep going on transient failures
        print(f"[prism] {var} {stamp} FAILED: {e}")
        return None

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(dest_dir / stamp)
    marker.write_text("ok")
    print(f"[prism] {var} {stamp} ok")
    return marker


def main(start: str, end: str, sleep: float = 1.0) -> None:
    cfg = load_config("data")
    src = cfg["sources"]["prism"]
    out_dir = Path(src["out_dir"])
    res = src["resolution"]
    for day in _daterange(start, end):
        for var in src["variables"]:
            download_day(var, res, day, out_dir)
            time.sleep(sleep)  # be polite to the free service


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()
    main(args.start, args.end, args.sleep)
