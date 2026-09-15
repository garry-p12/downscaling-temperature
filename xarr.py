"""Compare ERA5-Land monthly NetCDFs — why we stitch them together.

Each CDS download is ONE calendar month of hourly data. Training needs a
continuous daily time series (2019–2023), so open_era5_land() concatenates
all months, then resamples hourly -> daily mean.

Usage:
    python xarr.py
    python xarr.py --dir data_store_sc/raw/era5_land --plot
"""
from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

# Three months that illustrate different lengths (31d, 28d, 30d hours).
DEFAULT_FILES = [
    "era5land_southcentral_201901.nc",  # Jan — 31 days
    "era5land_southcentral_201902.nc",  # Feb — 28 days (2019)
    "era5land_southcentral_201906.nc",  # Jun — 30 days
]


def _time_dim(ds: xr.Dataset) -> str:
    for name in ("valid_time", "time"):
        if name in ds.dims:
            return name
    raise KeyError(f"no time dim in {list(ds.dims)}")


def _temp_var(ds: xr.Dataset) -> str:
    for name in ("t2m", "2m_temperature"):
        if name in ds:
            return name
    raise KeyError(f"no temperature var in {list(ds.data_vars)}")


def summarize(path: Path) -> dict:
    ds = xr.open_dataset(path)
    tdim = _time_dim(ds)
    var = _temp_var(ds)
    da = ds[var]
    t = ds[tdim]
    units = da.attrs.get("units", "?")
    lat = ds["latitude"] if "latitude" in ds else ds["lat"]
    lon = ds["longitude"] if "longitude" in ds else ds["lon"]
    info = {
        "file": path.name,
        "time_dim": tdim,
        "variable": var,
        "units": units,
        "n_timesteps": int(ds.sizes[tdim]),
        "n_days_approx": int(ds.sizes[tdim]) // 24,
        "spatial": (int(lat.size), int(lon.size)),
        "lat_range": (float(lat.min()), float(lat.max())),
        "lon_range": (float(lon.min()), float(lon.max())),
        "time_start": str(t.values[0]),
        "time_end": str(t.values[-1]),
        "size_mb": path.stat().st_size / 1e6,
    }
    vals = da.values
    finite = vals[vals > -1e4]
    if finite.size:
        k = float(finite.max()) > 200
        info["temp_range"] = (
            float(finite.min()) - (273.15 if k else 0),
            float(finite.max()) - (273.15 if k else 0),
        )
    ds.close()
    return info


def print_comparison(rows: list[dict]) -> None:
    print("\n=== ERA5-Land monthly files (each is ONE month, hourly) ===\n")
    hdr = f"{'file':<36} {'hours':>6} {'~days':>6} {'grid':>12} {'units':>6}  time span"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        grid = f"{r['spatial'][0]}x{r['spatial'][1]}"
        span = f"{r['time_start'][:10]} .. {r['time_end'][:10]}"
        print(f"{r['file']:<36} {r['n_timesteps']:>6} {r['n_days_approx']:>6} "
              f"{grid:>12} {r['units']:>6}  {span}")

    print("\n--- What differs between files? ---")
    hours = [r["n_timesteps"] for r in rows]
    grids = [r["spatial"] for r in rows]
    lats = [r["lat_range"] for r in rows]
    lons = [r["lon_range"] for r in rows]
    print(f"  Hourly steps per file: {hours}  (varies with month length)")
    print(f"  Spatial grid (lat x lon): {grids[0]} — same across months")
    print(f"  Lat range:  {lats[0]}  (identical)")
    print(f"  Lon range:  {lons[0]}  (identical)")

    print("\n--- Why stitch? ---")
    print("  * Each .nc = 1 month only — you cannot train on 2019–2023 from one file.")
    print("  * Spatial extent is the same; stitching concatenates along TIME.")
    print("  * open_era5_land() uses xr.open_mfdataset(..., combine='by_coords')")
    print("    then resample(time='1D').mean() for daily means (match POWER T2M).")


def show_one_file(path: Path) -> None:
    ds = xr.open_dataset(path)
    print(f"\n=== Full xarray view: {path.name} ===\n")
    print(ds)
    var = _temp_var(ds)
    print(f"\n--- {var} ---")
    print(ds[var])
    ds.close()


def show_stitched(era5_dir: Path) -> None:
    files = sorted(era5_dir.glob("era5land_*.nc"))
    if len(files) < 2:
        print("\n[stitch] need >=2 files in dir to demo concatenation")
        return
    print(f"\n=== After stitching all {len(files)} files ===\n")
    ds = xr.open_mfdataset(files, combine="by_coords")
    tdim = _time_dim(ds)
    var = _temp_var(ds)
    print(f"Combined hourly: {var} {tuple(ds[var].shape)}  ({tdim})")
    print(f"Time: {ds[tdim].values[0]} .. {ds[tdim].values[-1]}")
    daily = ds[var].resample({tdim: "1D"}).mean()
    print(f"After daily mean:  {var} {tuple(daily.shape)}  ({len(daily[tdim])} days)")
    print(f"\n  {len(files)} monthly files -> 1 continuous series -> daily aggregation")
    ds.close()


def maybe_plot(rows: list[dict], era5_dir: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: hours per compared month
    names = [r["file"].replace("era5land_southcentral_", "") for r in rows]
    hours = [r["n_timesteps"] for r in rows]
    axes[0].bar(names, hours, color=["#4c78a8", "#72b7b2", "#e45756"])
    axes[0].set_ylabel("hourly timesteps")
    axes[0].set_title("Each file = one month (different lengths)")
    axes[0].tick_params(axis="x", rotation=20)

    # Right: hours per file across full archive
    all_files = sorted(era5_dir.glob("era5land_*.nc"))
    counts = []
    labels = []
    for f in all_files:
        with xr.open_dataset(f) as ds:
            counts.append(int(ds.sizes[_time_dim(ds)]))
        labels.append(f.name.replace("era5land_southcentral_", ""))
    axes[1].bar(range(len(counts)), counts, color="#4c78a8", width=1.0)
    axes[1].set_xticks(range(0, len(labels), 6))
    axes[1].set_xticklabels([labels[i] for i in range(0, len(labels), 6)],
                            rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("hourly timesteps")
    axes[1].set_title(f"All {len(all_files)} monthly files (stitch along time)")

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[plot] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare ERA5-Land monthly NetCDFs")
    ap.add_argument("--dir", default="data_store_sc/raw/era5_land",
                    help="directory with era5land_*.nc files")
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES,
                    help="2-3 filenames to compare")
    ap.add_argument("--plot", action="store_true",
                    help="save bar chart to docs/dataset_gallery/era5_monthly_stitch.png")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="print full xarray repr for the first file")
    args = ap.parse_args()

    era5_dir = Path(args.dir)
    paths = [era5_dir / f for f in args.files]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing:\n  " + "\n  ".join(str(p) for p in missing))

    rows = [summarize(p) for p in paths]
    print_comparison(rows)
    if args.verbose:
        show_one_file(paths[0])
    show_stitched(era5_dir)
    if args.plot:
        maybe_plot(rows, era5_dir, Path("docs/dataset_gallery/era5_monthly_stitch.png"))


if __name__ == "__main__":
    main()
