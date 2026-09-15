"""Compare NASA POWER tiled NetCDFs — why we stitch them together.

Unlike ERA5-Land (one file per month, same spatial grid), POWER downloads are
split two ways:
  * SPATIALLY — API box limit ~4.5 deg/side -> 9 tiles (t00..t08)
  * TEMPORALLY — one calendar year per request -> 5 years

open_power() mosaics tiles per year with combine_first, then concatenates years.

Usage:
    python power_xarr.py
    python power_xarr.py --dir data_store_sc/raw/power --plot
    python power_xarr.py -v
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Same year, different tiles (spatial split) + same tile, different year (time split).
DEFAULT_FILES = [
    "power_T2M_southcentral_2019_t00.nc",  # SW corner tile
    "power_T2M_southcentral_2019_t04.nc",  # central tile (different lat/lon box)
    "power_T2M_southcentral_2020_t00.nc",  # same tile as t00, leap year
]

_TILE_RE = re.compile(r"_t(\d+)\.nc$")


def _parse_tile(path: Path) -> tuple[int | None, int | None]:
    m = _TILE_RE.search(path.name)
    tile = int(m.group(1)) if m else None
    parts = path.stem.split("_")
    year = int(parts[-2]) if parts[-2].isdigit() else None
    return tile, year


def summarize(path: Path) -> dict:
    ds = xr.open_dataset(path)
    lat = ds["latitude"] if "latitude" in ds.coords else ds["lat"]
    lon = ds["longitude"] if "longitude" in ds.coords else ds["lon"]
    da = ds["T2M"]
    tile, year = _parse_tile(path)
    info = {
        "file": path.name,
        "tile": tile,
        "year": year,
        "variable": "T2M",
        "units": da.attrs.get("units", "?"),
        "n_days": int(ds.sizes["time"]),
        "spatial": (int(lat.size), int(lon.size)),
        "lat_range": (float(lat.min()), float(lat.max())),
        "lon_range": (float(lon.min()), float(lon.max())),
        "lat_step": float(lat[1] - lat[0]) if lat.size > 1 else None,
        "lon_step": float(lon[1] - lon[0]) if lon.size > 1 else None,
        "time_start": str(ds["time"].values[0])[:10],
        "time_end": str(ds["time"].values[-1])[:10],
        "size_mb": path.stat().st_size / 1e6,
    }
    finite = da.values[np.isfinite(da.values) & (da.values > -900)]
    if finite.size:
        info["temp_range_c"] = (float(finite.min()), float(finite.max()))
    ds.close()
    return info


def print_comparison(rows: list[dict]) -> None:
    print("\n=== NASA POWER files (each = one tile × one year) ===\n")
    hdr = (f"{'file':<38} {'tile':>4} {'year':>5} {'days':>5} {'grid':>8} "
           f"{'units':>5}  lat range        lon range")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        grid = f"{r['spatial'][0]}x{r['spatial'][1]}"
        lat = f"{r['lat_range'][0]:.1f}..{r['lat_range'][1]:.1f}"
        lon = f"{r['lon_range'][0]:.1f}..{r['lon_range'][1]:.1f}"
        print(f"{r['file']:<38} {r['tile']:>4} {r['year']:>5} {r['n_days']:>5} "
              f"{grid:>8} {r['units']:>5}  {lat:>15}  {lon:>15}")

    print("\n--- What differs between files? ---")
    r0, r1, r2 = rows
    print(f"  t00 vs t04 (same year):  grid {r0['spatial']} vs {r1['spatial']}; "
          f"different lat/lon boxes (spatial tiles)")
    print(f"  t00 2019 vs 2020:        grid {r0['spatial']} vs {r2['spatial']}; "
          f"days {r0['n_days']} vs {r2['n_days']} (leap year); same tile extent")
    print(f"  Native spacing:          lat ~{r0['lat_step']} deg, "
          f"lon ~{r1['lon_step'] or r0['lon_step']} deg (~50 km)")

    print("\n--- Why stitch? ---")
    print("  * Each .nc covers only a PATCH of the domain for ONE year.")
    print("  * Spatial tiles (t00..t08) are mosaicked with combine_first per year.")
    print("  * Years are concatenated along time -> one (1826, 21, 20) field.")
    print("  * open_power() in data/download_power.py does both steps.")
    print("  * NOT xr.merge — same variable name would keep one tile and leave ~70% NaN.")


def show_one_file(path: Path) -> None:
    ds = xr.open_dataset(path)
    print(f"\n=== Full xarray view: {path.name} ===\n")
    print(ds)
    print("\n--- T2M ---")
    print(ds["T2M"])
    ds.close()


def show_stitched(power_dir: Path) -> None:
    files = sorted(power_dir.glob("power_*.nc"))
    if len(files) < 2:
        print("\n[stitch] need >=2 files in dir to demo stitching")
        return

    import os
    os.environ.setdefault("DOWNSCALE_CONFIG_data", "configs/data_southcentral.yaml")
    from common import load_config
    from data.download_power import open_power

    print(f"\n=== After stitching all {len(files)} files ===\n")
    ds = open_power(load_config("data"))
    print(f"Stitched T2M: {tuple(ds['T2M'].shape)}  (time, lat, lon)")
    print(f"Time: {ds['time'].values[0]} .. {ds['time'].values[-1]}")
    print(f"Lat:  {float(ds['lat'].min()):.1f} .. {float(ds['lat'].max()):.1f}  "
          f"({len(ds['lat'])} points)")
    print(f"Lon:  {float(ds['lon'].min()):.1f} .. {float(ds['lon'].max()):.1f}  "
          f"({len(ds['lon'])} points)")
    nan_frac = float((~ds["T2M"].notnull()).mean())
    print(f"NaN fraction: {100 * nan_frac:.2f}%")
    print(f"\n  {len(files)} tile-year files -> 1 continuous coarse field (daily degC)")


def maybe_plot(rows: list[dict], power_dir: Path, out: Path) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: grid size per tile for 2019
    tiles_2019 = sorted(power_dir.glob("power_*_2019_t*.nc"))
    labels, lat_n, lon_n = [], [], []
    for f in tiles_2019:
        with xr.open_dataset(f) as ds:
            labels.append(f.name.split("_")[-1].replace(".nc", ""))
            lat_n.append(int(ds.sizes["lat"]))
            lon_n.append(int(ds.sizes["lon"]))
    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w / 2, lat_n, w, label="lat points", color="#4c78a8")
    axes[0].bar(x + w / 2, lon_n, w, label="lon points", color="#e45756")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel("grid points")
    axes[0].set_title("2019: each tile = different spatial patch (7x6..7x7)")
    axes[0].legend(fontsize=8)

    # Right: schematic lat/lon footprint of compared tiles (2019)
    ax = axes[1]
    colors = ["#4c78a8", "#e45756", "#72b7b2"]
    for r, c in zip(rows[:2], colors):  # t00 and t04 only (same year)
        lat0, lat1 = r["lat_range"]
        lon0, lon1 = r["lon_range"]
        rect = mpatches.Rectangle(
            (lon0, lat0), lon1 - lon0, lat1 - lat0,
            linewidth=2, edgecolor=c, facecolor=c, alpha=0.25,
            label=f"t{r['tile']:02d} ({r['spatial'][0]}x{r['spatial'][1]})",
        )
        ax.add_patch(rect)
    ax.set_xlim(min(r["lon_range"][0] for r in rows[:2]) - 0.5,
                max(r["lon_range"][1] for r in rows[:2]) + 0.5)
    ax.set_ylim(min(r["lat_range"][0] for r in rows[:2]) - 0.5,
                max(r["lat_range"][1] for r in rows[:2]) + 0.5)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Spatial tiles cover different boxes (must mosaic)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[plot] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare NASA POWER tiled NetCDFs")
    ap.add_argument("--dir", default="data_store_sc/raw/power",
                    help="directory with power_*.nc files")
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES,
                    help="2-3 filenames to compare")
    ap.add_argument("--plot", action="store_true",
                    help="save chart to docs/dataset_gallery/power_tile_stitch.png")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="print full xarray repr for the first file")
    args = ap.parse_args()

    power_dir = Path(args.dir)
    paths = [power_dir / f for f in args.files]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing:\n  " + "\n  ".join(str(p) for p in missing))

    rows = [summarize(p) for p in paths]
    print_comparison(rows)
    if args.verbose:
        show_one_file(paths[0])
    show_stitched(power_dir)
    if args.plot:
        maybe_plot(rows, power_dir,
                   Path("docs/dataset_gallery/power_tile_stitch.png"))


if __name__ == "__main__":
    main()
