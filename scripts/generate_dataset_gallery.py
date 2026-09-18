"""Inspect raw + built datasets and write preview figures for PPT docs."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SC = ROOT / "data_store_sc"
OUT = ROOT / "docs" / "dataset_gallery"
OUT.mkdir(parents=True, exist_ok=True)

POWER = SC / "raw/power/power_T2M_southcentral_2019_t00.nc"
ERA5 = SC / "raw/era5_land/era5land_southcentral_201906.nc"
DEM = SC / "raw/dem/dem.tif"
LC = SC / "raw/landcover/landcover.tif"
ZARR = SC / "super" / "dataset.zarr"   # the 21-channel superset, not the V0 store
NORM = SC / "norm_stats.json"
ISD = SC / "raw/isd/722540-13904-2023.gz"  # KAUS


def save_fig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close()


def inspect_power() -> str:
    ds = xr.open_dataset(POWER)
    lines = [str(ds), "", f"T2M shape: {ds['T2M'].shape}", f"units: {ds['T2M'].attrs.get('units')}"]
  # map mid-year
    da = ds["T2M"].isel(time=180)
    fig, ax = plt.subplots(figsize=(6, 4))
    da.plot(ax=ax, cmap="RdYlBu_r")
    ax.set_title(f"POWER T2M — {da.time.values} (tile t00, 2019)")
    save_fig("01_power_t2m_map.png")
    ds.close()
    return "\n".join(lines)


def inspect_era5() -> str:
    ds = xr.open_dataset(ERA5)
    var = "t2m" if "t2m" in ds else "2m_temperature"
    tname = "valid_time" if "valid_time" in ds.dims else "time"
    daily = ds[var].resample({tname: "1D"}).mean()
    if float(daily.max()) > 200:
        daily = daily - 273.15
    lines = [str(ds), "", f"{var} hourly shape: {ds[var].shape}",
             f"daily mean shape: {daily.shape}"]
    da = daily.isel({tname: 15})
    fig, ax = plt.subplots(figsize=(6, 4))
    da.plot(ax=ax, cmap="RdYlBu_r")
    ax.set_title(f"ERA5-Land daily mean {var} — {da[tname].values}")
    save_fig("02_era5land_t2m_map.png")
    ds.close()
    return "\n".join(lines)


def inspect_raster(path: Path, title: str, cmap: str, fname: str) -> str:
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = json.dumps({k: str(v) for k, v in src.profile.items()}, indent=2)
        lines = [f"File: {path.name}", profile,
                 f"shape: {data.shape}", f"dtype: {data.dtype}",
                 f"min/max (valid): {np.nanmin(data[data > -1e4]):.1f} / {np.nanmax(data):.1f}"]
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data, cmap=cmap, origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(fname)
    return "\n".join(lines)


def inspect_zarr() -> str:
    ds = xr.open_zarr(ZARR, consolidated=True)
    lines = [str(ds), "",
             f"input channels: {list(ds.channel_in.values)}",
             f"target channels: {list(ds.channel_out.values)}",
             f"time steps: {ds.sizes['time']}",
             f"grid: {ds.sizes['y']} x {ds.sizes['x']}"]
    # one day, coarse_tmp and tmp
    t = 180
    names = list(ds.channel_in.values)
    ci = names.index("coarse_tmp")
    coarse = ds["input"].isel(time=t, channel_in=ci).values
    truth = ds["target"].isel(time=t, channel_out=0).values
    dem_i = names.index("dem")
    dem = ds["input"].isel(time=t, channel_in=dem_i).values
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, arr, title in zip(axes, [coarse, truth, dem],
                              ["coarse_tmp (anomaly)", "tmp target (anomaly)", "dem"]):
        im = ax.imshow(arr, cmap="RdYlBu_r" if "dem" not in title else "terrain")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Built Zarr — day index {t} ({ds.time.values[t]})")
    save_fig("05_zarr_channels.png")
    return "\n".join(lines)


def inspect_isd() -> str:
    with gzip.open(ISD, "rt") as f:
        rows = [f.readline().strip().split(",") for _ in range(24 * 31)]  # Jan
    # ISD-Lite: year,month,day,hour,temp(C*10),...
    temps = [int(r[4]) / 10.0 for r in rows if len(r) > 4 and r[4] not in ("-9999", "")]
    hours = list(range(len(temps)))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(hours, temps, lw=0.8)
    ax.set_title("ISD KAUS — hourly temp (Jan 2023, °C)")
    ax.set_xlabel("hour index (Jan)")
    ax.set_ylabel("°C")
    save_fig("06_isd_timeseries.png")
    with gzip.open(ISD, "rt") as f:
        head = [f.readline() for _ in range(8)]
    lines = ["File: KAUS ISD-Lite 2023 (gzipped CSV)",
             "Columns: year, month, day, hour, temp, dewp, slp, wdir, wspd, ...",
             "temp in tenths °C; -9999 = missing",
             ""] + [ln.rstrip() for ln in head]
    return "\n".join(lines)


def inspect_stitched_power() -> str:
    import os
    os.environ.setdefault("DOWNSCALE_CONFIG_data", "configs/data_southcentral.yaml")
    from common import load_config
    from data.download_power import open_power

    ds = open_power(load_config("data"))
    da = ds["T2M"].isel(time=180)
    fig, ax = plt.subplots(figsize=(7, 5))
    da.plot(ax=ax, cmap="RdYlBu_r")
    ax.set_title(f"POWER T2M stitched (all tiles) — {da.time.values}")
    save_fig("01b_power_stitched_map.png")
    lines = [str(ds), "", f"stitched T2M shape: {ds['T2M'].shape}"]
    return "\n".join(lines)


def main() -> None:
    meta = {}
    meta["power"] = inspect_power()
    meta["power_stitched"] = inspect_stitched_power()
    meta["era5"] = inspect_era5()
    meta["dem"] = inspect_raster(DEM, "GLO-90 DEM (m)", "terrain", "03_dem_map.png")
    meta["landcover"] = inspect_raster(LC, "NLCD land-cover class", "tab20", "04_nlcd_map.png")
    meta["zarr"] = inspect_zarr()
    meta["isd"] = inspect_isd()
    meta["norm_stats"] = NORM.read_text()
    (OUT / "introspection.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote figures + introspection to {OUT}")


if __name__ == "__main__":
    main()
