"""Independent cross-check against NOAA ISD station observations.

ERA5-Land is a reanalysis, not observations. Every metric in this project is
measured against it, so a model can score well while still being wrong about
what actually happened on the ground. This script scores the downscaled field
against real thermometers.

Data: NOAA Integrated Surface Database (ISD-Lite), public, no credentials.
    https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/<year>/<USAF>-<WBAN>-<year>.gz
Default station is Austin-Bergstrom (KAUS, 722540-13904), which sits inside the
spatial holdout — so this is an observational check on the one region no model
was trained on.

ISD-Lite is hourly, temperature in tenths of degC, -9999 = missing. Daily means
are computed only from days with >= min_hours valid readings, matching the
daily-mean definition used for POWER and ERA5-Land.

Read the output as a three-way comparison, not a scoreboard: interpolated
POWER, the model, and ERA5-Land itself are all scored against the station. If
ERA5-Land is as far from the station as the model is, the truth product is the
limiting factor and no amount of modelling fixes it.

Usage:
    DOWNSCALE_CONFIG_data=configs/data_southcentral.yaml \\
        python -m evaluation.station_check --archs deepsd edsr --year 2023
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from common import Normalizer, load_config
from models.model import load_checkpoint
from training.run_all import ckpt_path

ISD_URL = ("https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/"
           "{year}/{usaf}-{wban}-{year}.gz")

STATIONS = {
    # name: (USAF, WBAN, lat, lon)   — all inside the Austin holdout box
    "KAUS_Austin_Bergstrom": ("722540", "13904", 30.183, -97.680),
    "KATT_Austin_Executive": ("722544", "13958", 30.320, -97.760),
}


def fetch_isd(usaf: str, wban: str, year: int, cache: Path) -> pd.DataFrame:
    """Daily mean temperature (degC) from ISD-Lite hourly records."""
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{usaf}-{wban}-{year}.gz"
    if not dest.exists():
        url = ISD_URL.format(year=year, usaf=usaf, wban=wban)
        print(f"[station] fetching {url}")
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())

    with gzip.open(dest, "rt") as fh:
        raw = fh.read()
    # Fixed-width: year, month, day, hour, airtemp(tenths degC), ...
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+", header=None,
                     usecols=[0, 1, 2, 3, 4],
                     names=["y", "m", "d", "h", "t"])
    df = df[df["t"] != -9999]
    df["t"] = df["t"] / 10.0
    df["date"] = pd.to_datetime(dict(year=df.y, month=df.m, day=df.d))
    return df


def daily_mean(df: pd.DataFrame, min_hours: int = 20) -> pd.Series:
    """Daily means from days with enough hourly coverage.

    A day with only daytime readings biases warm; requiring near-complete
    coverage keeps the station comparable to a true 24-hour mean.
    """
    g = df.groupby("date")["t"]
    return g.mean()[g.count() >= min_hours]


def nearest_cell(ds: xr.Dataset, lat: float, lon: float) -> tuple[int, int]:
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", ds.attrs.get("crs", "EPSG:5070"),
                              always_xy=True)
    x, y = tf.transform(lon, lat)
    j = int(np.abs(ds["x"].values - x).argmin())
    i = int(np.abs(ds["y"].values - y).argmin())
    return i, j


@torch.no_grad()
def model_series(arch: str, test: xr.Dataset, nz: Normalizer, i: int, j: int,
                 device) -> np.ndarray | None:
    ckpt = ckpt_path(arch)
    if not ckpt.exists():
        return None
    model, _ = load_checkpoint(ckpt, device, load_config("model"))

    names = test["channel_in"].values.tolist()
    inp = test["input"].values
    out = np.empty(inp.shape[0], "float32")
    for k in range(inp.shape[0]):
        x = inp[k].copy()
        for c, nm in enumerate(names):
            if nm != "land_mask":
                x[c] = nz.transform(f"in::{nm}", x[c])
        np.nan_to_num(x, copy=False)
        p = model(torch.from_numpy(x).unsqueeze(0).to(device), {"temp"})
        out[k] = p["temp"][0, 0, i, j].cpu().numpy()
    return out


def _score(pred: np.ndarray, obs: np.ndarray) -> dict:
    d = pred - obs
    return {"rmse": float(np.sqrt(np.nanmean(d ** 2))),
            "mae": float(np.nanmean(np.abs(d))),
            "bias": float(np.nanmean(d)),
            "corr": float(np.corrcoef(pred[np.isfinite(d)],
                                      obs[np.isfinite(d)])[0, 1]),
            "n_days": int(np.isfinite(d).sum())}


def main(archs: list[str], year: int, station: str, device_name: str) -> None:
    cfg = load_config("data")
    zarr = cfg["dataset"]["out_zarr"]
    nz = Normalizer.load(str(Path(zarr).parent / "norm_stats.json"))
    device = torch.device(device_name)

    usaf, wban, lat, lon = STATIONS[station]
    obs = daily_mean(fetch_isd(usaf, wban, year,
                               Path("data_store_sc/raw/isd")))

    full = xr.open_zarr(zarr, consolidated=True)
    test = full.sel(time=full["split"] == "test")
    i, j = nearest_cell(full, lat, lon)
    times = pd.DatetimeIndex(test["time"].values)
    doy = times.dayofyear.values - 1
    names = test["channel_in"].values.tolist()

    clim_t = full["clim_tmp"].values[doy, i, j]
    clim_c = full["clim_coarse_tmp"].values[doy, i, j]
    era5 = test["target"].values[:, 0, i, j] + clim_t
    power = test["input"].values[:, names.index("coarse_tmp"), i, j] + clim_c

    aligned = pd.Series(obs).reindex(times)
    o = aligned.values
    print(f"[station] {station} at ({lat}, {lon}) -> grid cell ({i}, {j})")
    print(f"[station] {int(np.isfinite(o).sum())}/{len(times)} days with "
          f">=20 valid hourly obs in {year}")

    results = {"station": station, "year": year,
               "grid_cell": [i, j], "methods": {}}
    results["methods"]["ERA5-Land (our 'truth')"] = _score(era5, o)
    results["methods"]["interpolated_POWER"] = _score(power, o)

    for arch in archs:
        s = model_series(arch, test, nz, i, j, device)
        if s is None:
            print(f"[station] {arch}: no checkpoint, skipping")
            continue
        results["methods"][arch] = _score(nz.inverse("out::tmp", s) + clim_t, o)

    # Per-station filename: a single fixed path meant the second
    # station silently overwrote the first.
    out = Path(f"outputs/station_check_{station}_{year}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print(f"\n{'method':<26}{'RMSE':>8}{'MAE':>8}{'bias':>8}{'corr':>8}{'days':>7}")
    print("-" * 65)
    for k, m in sorted(results["methods"].items(), key=lambda kv: kv[1]["rmse"]):
        print(f"{k:<26}{m['rmse']:>8.3f}{m['mae']:>8.3f}{m['bias']:>+8.3f}"
              f"{m['corr']:>8.4f}{m['n_days']:>7}")
    print("\nScored against STATION OBSERVATIONS, not ERA5-Land. If ERA5-Land's")
    print("own row is no better than the models', the reanalysis is the")
    print("accuracy ceiling and further modelling cannot help at this site.")
    print(f"[station] wrote {out}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["deepsd", "edsr"])
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--station", default="KAUS_Austin_Bergstrom",
                    choices=list(STATIONS))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    main(args.archs, args.year, args.station, args.device)
