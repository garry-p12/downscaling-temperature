"""Download ERA5-Land 2 m temperature (0.1 deg) as the high-resolution truth.

ERA5-Land is ECMWF's land-surface reanalysis at 0.1 deg — about 11 km in
latitude and 8.7 km in longitude at 39N, so it lands almost exactly on this
project's 10 km target grid. That is its advantage over AORC: the truth needs
only reprojection, not aggregation across a 10x resolution gap.

The caveat, stated plainly because it changes how results should be read:
ERA5-Land is PRODUCED by downscaling 0.25 deg ERA5 atmospheric forcing with a
land-surface model, including a lapse-rate correction against 0.1 deg
orography. A network given a DEM can learn much of that deterministic
correction, so RMSE against ERA5-Land will be markedly lower than against
AORC WITHOUT the method being better — the target is simply more predictable.
Numbers from the two truths are not comparable. AORC is the observation-informed
reference; ERA5-Land is the resolution-matched one. Report both.

Requires a free Copernicus CDS account and ``~/.cdsapirc``:
    url: https://cds.climate.copernicus.eu/api
    key: <your-key>
Accept the ERA5-Land licence on the dataset page before the first request.

Hourly data is requested and averaged to a daily mean in flight — the same
statistic NASA POWER reports as T2M, so predictor and target stay aligned.

Usage:
    python -m data.download_era5_land --years 2019 2020 2021 2022 2023
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import load_config

DATASET = "reanalysis-era5-land"


def _area(domain: dict, pad: float) -> list[float]:
    # CDS "area" is [North, West, South, East].
    return [domain["lat_max"] + pad, domain["lon_min"] - pad,
            domain["lat_min"] - pad, domain["lon_max"] + pad]


def request_month(client, variable: str, year: int, month: int,
                  area: list[float], out_path: Path) -> bool:
    """One month per request — a full year of hourly ERA5-Land times out."""
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[era5land] {out_path.name} exists, skipping")
        return True
    request = {
        "variable": [variable],
        "year": str(year),
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[era5land] requesting {year}-{month:02d} -> {out_path.name}")
    try:
        client.retrieve(DATASET, request, str(out_path))
        return True
    except Exception as e:  # noqa: BLE001 - keep going; re-run retries
        print(f"[era5land] {year}-{month:02d} FAILED: {e}")
        return False


def open_era5_land(cfg):
    """Open all downloaded months as one daily-mean (time, lat, lon) Dataset."""
    import xarray as xr

    src = cfg["sources"]["era5_land"]
    files = sorted(Path(src["out_dir"]).glob("era5land_*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No ERA5-Land files in {src['out_dir']}. Run: "
            f"python -m data.download_era5_land --years ...")

    ds = xr.open_mfdataset(files, combine="by_coords")
    rename = {a: b for a, b in (("latitude", "lat"), ("longitude", "lon"),
                                ("valid_time", "time"))
              if a in ds.coords or a in ds.dims}
    if rename:
        ds = ds.rename(rename)
    # Hourly -> daily mean, matching POWER's T2M definition.
    ds = ds.resample(time="1D").mean()
    return ds.sortby("lat").sortby("lon").sortby("time")


def main(years: list[int], months: list[int] | None = None) -> None:
    import cdsapi

    cfg = load_config("data")
    src = cfg["sources"]["era5_land"]
    out_dir = Path(src["out_dir"])
    name = cfg["domain"]["name"]
    area = _area(cfg["domain"], src.get("pad_deg", 0.5))
    client = cdsapi.Client()

    failed = 0
    # Month filter exists so a long pull can be SPLIT across processes on
    # disjoint months. CDS accepts concurrent requests, but two processes must
    # never target the same file — cdsapi writes the destination directly, so
    # overlapping runs would corrupt it.
    wanted = months or list(range(1, 13))
    for year in years:
        for month in wanted:
            out = out_dir / f"era5land_{name}_{year}{month:02d}.nc"
            if not request_month(client, src["variable"], year, month, area, out):
                failed += 1
    if failed:
        raise SystemExit(f"[era5land] {failed} request(s) failed — re-run to retry")
    print("[era5land] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--months", type=int, nargs="+", default=None,
                    help="subset of months (for parallel splits)")
    args = ap.parse_args()
    main(args.years, args.months)
