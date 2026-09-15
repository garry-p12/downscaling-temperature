"""Inspect NASA POWER daily T2M netCDF files (same idea as xarr.py).

Examples:
    # Print overview of one tile file (edit DEFAULT_NC or pass --file)
    python power_vis.py

    # List all POWER .nc files under the store
    python power_vis.py --list

    # Inspect a specific file
    python power_vis.py --file data_store_sc/raw/power/power_T2M_southcentral_2019_t00.nc

    # Show one timestep as numbers + optional map
    python power_vis.py --file ... --time 0
    python power_vis.py --file ... --time 100 --plot

    # Stitch all tiles/years via download_power.open_power (needs config)
    DOWNSCALE_CONFIG_data=configs/data_southcentral.yaml python power_vis.py --stitch
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

# Default: first POWER file found under the south-central store, or set explicitly.
DEFAULT_DIR = Path("data_store_sc/raw/power")
DEFAULT_NC: str | None = None  # e.g. "data_store_sc/raw/power/power_T2M_southcentral_2019_t00.nc"
PARAM = "T2M"
FILL = -999.0


def find_power_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.glob("power_*.nc"))


def resolve_file(path: str | None, search_dir: Path) -> Path:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return p
    if DEFAULT_NC:
        p = Path(DEFAULT_NC)
        if not p.exists():
            raise FileNotFoundError(f"DEFAULT_NC not found: {p}")
        return p
    files = find_power_files(search_dir)
    if not files:
        raise FileNotFoundError(
            f"No power_*.nc files in {search_dir}. "
            "Run: python -m data.download_power --years 2019"
        )
    return files[0]


def normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename = {a: b for a, b in (("latitude", "lat"), ("longitude", "lon")) if a in ds.coords}
    if rename:
        ds = ds.rename(rename)
    return ds.sortby("lat").sortby("lon").sortby("time")


def open_one(path: Path) -> xr.Dataset:
    ds = normalize_coords(xr.open_dataset(path))
    if PARAM in ds:
        ds[PARAM] = ds[PARAM].where(ds[PARAM] > FILL / 2)
    return ds


def print_overview(ds: xr.Dataset, label: str) -> None:
    print(f"\n=== {label} ===")
    print(ds)
    print("\n--- data variables ---")
    print(ds.data_vars)
    if PARAM not in ds:
        print(f"\n(warning: expected variable {PARAM!r} not found)")
        return
    da = ds[PARAM]
    print(f"\n--- {PARAM} ---")
    print(da)
    vals = da.values
    finite = np.isfinite(vals)
    if finite.any():
        print(
            f"  range: {vals[finite].min():.2f} .. {vals[finite].max():.2f} degC"
            f"  |  NaN fraction: {1 - finite.mean():.3f}"
        )
    if "time" in da.dims:
        print(f"  time: {da.sizes['time']} steps")
        print(f"        {da.time.values[0]} .. {da.time.values[-1]}")


def print_timestep(ds: xr.Dataset, time_idx: int) -> None:
    if PARAM not in ds:
        return
    da = ds[PARAM].isel(time=time_idx)
    print(f"\n--- {PARAM} at time index {time_idx} ({da.time.values}) ---")
    print(da)
    arr = da.values
    finite = np.isfinite(arr)
    if finite.any():
        print(f"  min/mean/max: {arr[finite].min():.2f} / "
              f"{arr[finite].mean():.2f} / {arr[finite].max():.2f} degC")


def maybe_plot(ds: xr.Dataset, time_idx: int) -> None:
    if PARAM not in ds:
        return
    import matplotlib.pyplot as plt

    da = ds[PARAM].isel(time=time_idx)
    fig, ax = plt.subplots(figsize=(8, 5))
    da.plot(ax=ax, cmap="RdYlBu_r")
    ax.set_title(f"NASA POWER {PARAM} — {da.time.values}")
    plt.tight_layout()
    plt.show()


def list_files(search_dir: Path) -> None:
    files = find_power_files(search_dir)
    if not files:
        print(f"No power_*.nc files in {search_dir}")
        return
    print(f"Found {len(files)} file(s) in {search_dir}:\n")
    for f in files:
        mb = f.stat().st_size / 1e6
        print(f"  {f}  ({mb:.1f} MB)")


def open_stitched() -> xr.Dataset:
    from common import load_config
    from data.download_power import open_power

    cfg = load_config("data")
    return open_power(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect NASA POWER netCDF files")
    ap.add_argument("--file", "-f", help="Path to one power_*.nc file")
    ap.add_argument("--dir", default=str(DEFAULT_DIR),
                    help=f"Directory to search when --file is omitted (default: {DEFAULT_DIR})")
    ap.add_argument("--list", "-l", action="store_true", help="List POWER .nc files and exit")
    ap.add_argument("--stitch", action="store_true",
                    help="Open stitched dataset via download_power.open_power (uses config)")
    ap.add_argument("--time", "-t", type=int, default=None,
                    help="Print (and optionally plot) one time index")
    ap.add_argument("--plot", "-p", action="store_true", help="Show a map (requires --time)")
    args = ap.parse_args()

    search_dir = Path(args.dir)

    if args.list:
        list_files(search_dir)
        return

    if args.stitch:
        ds = open_stitched()
        print_overview(ds, "stitched POWER (all tiles/years)")
    else:
        path = resolve_file(args.file, search_dir)
        print(f"Opening: {path}")
        ds = open_one(path)
        print_overview(ds, path.name)

    if args.time is not None:
        print_timestep(ds, args.time)
        if args.plot:
            maybe_plot(ds, args.time)


if __name__ == "__main__":
    main()
