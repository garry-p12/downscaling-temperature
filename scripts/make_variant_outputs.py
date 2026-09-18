"""Geo-referenced maps and per-cell CSVs for every covariate variant.

One command regenerates everything a reader needs to inspect the training store
without opening Zarr:

    outputs/training_store_cells.csv        all static channels, one row per cell
    outputs/variant_stores/<v>_cells.csv    one file per variant, its exact channels
    outputs/variant_stores/README.txt       what the columns mean
    image_outputs/slides/proto_terrain.png      terrain covariates, geo-referenced
    image_outputs/slides/proto_landcover.png    land-cover composition
    image_outputs/slides/variant_<v>.png        one map sheet per variant

WHY THE CHANNEL LISTS ARE PARSED, NOT TYPED
    The variant definitions are read out of slurm/covariate_ablation_body.sh,
    which is the script that actually trained them. Retyping them here would let
    the documentation drift from the experiment, and the whole ablation rests on
    each arm's channel set being exactly what was claimed.

WHY LAT/LON NEEDS CARE
    The target grid is EPSG:5070 Albers. Lines of constant latitude are CURVED
    across it: latitude varies ~0.3 deg along a single pixel row. So lat/lon
    cannot be reconstructed from row/col by linear interpolation, and the maps
    carry a real graticule rather than evenly spaced ticks.

WHY TWO PROVENANCE FLAGS ARE EMITTED
    Two source datasets do not cover the whole target grid, and the build fills
    the gap by nearest neighbour (data/build_dataset.py _fill_nan_nearest). That
    fill is invisible in the values themselves - it looks like ordinary data -
    so every row carries a flag saying whether its land cover and its truth are
    measured or extrapolated:

    nlcd_extrapolated   NLCD is CONUS-only but the domain reaches 26.7 N, ~470 km
                        into Mexico. ~14% of land cells have NO real NLCD pixel.
                        Worse than a plain fill: landcover_fractions excludes
                        nodata from the denominator, so a donor cell with 9% real
                        coverage still emits a composition summing to 1.0, which
                        nearest neighbour then broadcasts ~150 km. That is where
                        the 71%-"developed" cells in Coahuila desert and the
                        >50%-wetland cells in the Chihuahuan Desert come from.
    era5_extrapolated   ERA5-Land was downloaded to -104.5 W, but the Albers grid
                        bulges west to -104.79 W, so a strip of cells has no
                        reanalysis truth.

    Neither affects the held-out region (zero flagged cells fall inside it), and
    every variant saw bit-identical data, so the ablation ranking is unaffected.
    The maps GREY OUT nlcd_extrapolated cells on land-cover panels; the CSVs keep
    the real stored numbers and flag them, so nothing is silently altered.

Usage:
    python scripts/make_variant_outputs.py            # everything
    python scripts/make_variant_outputs.py --day 2023-07-20
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common import load_config                       # noqa: E402
from common.grid import build_target_grid            # noqa: E402
from training.dataset import holdout_bounds          # noqa: E402

FIGDIR = REPO / "image_outputs" / "slides"
CSVDIR = REPO / "outputs" / "variant_stores"
CACHE = REPO / "outputs" / ".provenance_cache.npz"

# Channels that vary in time. A per-cell CSV can only show one slice of a
# (time, channel, y, x) store, so these are emitted for one named day.
TIME_VARYING = {"coarse_tmp", "doy_sin", "doy_cos"}
# ...and of those, these are also spatially CONSTANT, so mapping them is pointless.
SPATIALLY_CONSTANT = {"doy_sin", "doy_cos"}
# NLCD class codes. The number is a LABEL, not a quantity, so it gets a
# qualitative colormap; a light-to-dark ramp would imply 95 > 11 is meaningful.
CATEGORICAL = {"landcover"}
# Channels derived from NLCD, hence greyed out where NLCD does not reach.
NLCD_DERIVED = {"landcover", "urban_frac"}

UNITS = {
    "coarse_tmp": "degC anomaly", "dem": "m", "coastal_dist": "km",
    "elev_std": "m", "slope_mean": "degrees", "tpi": "m",
    "northness": "cos(aspect)", "eastness": "sin(aspect)",
    "urban_frac": "fraction", "land_mask": "0 / 1", "landcover": "NLCD class",
}

SURF, INK, INK2, INK3 = "#fcfdfe", "#0f1419", "#55606e", "#7d8996"
ORANGE, GRAT, NODATA = "#eb6834", "#5b6b7a", "#c9c6bf"
LAT_LEVELS = np.arange(28, 35, 2)        # one definition for lines AND labels;
LON_LEVELS = np.arange(-104, -93, 2)     # separate ones silently disagreed once.


def variant_channels() -> dict[str, list[str]]:
    """Parse the variant channel lists from the script that ran the ablation."""
    src = (REPO / "slurm" / "covariate_ablation_body.sh").read_text()
    raw = dict(re.findall(r'^(BASE|V1|V2|V3|V4)="((?:[^"\\]|\\\n)*)"', src, re.M))
    if len(raw) != 5:
        raise SystemExit(
            f"expected BASE,V1..V4 in slurm/covariate_ablation_body.sh, "
            f"parsed {sorted(raw)}. The ablation definition changed shape - fix "
            f"this parser rather than shipping CSVs that do not match training.")

    def expand(key: str) -> list[str]:
        return raw[key].replace("\\\n", " ").replace("$BASE", raw["BASE"]).split()

    return {"v0_base": expand("BASE"), "v1_terrain": expand("V1"),
            "v2_landcov": expand("V2"), "v3_nocoast": expand("V3"),
            "v4_elevstd": expand("V4")}


def provenance_masks(tg, cfg) -> tuple[np.ndarray, np.ndarray]:
    """(nlcd_valid_frac, era5_outside) on the target grid.

    nlcd_valid_frac is the fraction of each 10 km cell covered by a real NLCD
    pixel, computed by area-averaging the raster's valid mask with the same
    reprojection the covariates use - so 0.0 means the cell's land cover is
    pure nearest-neighbour extrapolation. Cached: it is a raster warp.
    """
    if CACHE.exists():
        z = np.load(CACHE)
        if z["shape"].tolist() == list(tg.shape):
            return z["nlcd"], z["era5"]

    # rasterio imported lazily: the training stack must not depend on it
    # (a module-scope import of it once killed every cluster run).
    import rasterio

    from data.covariates import _aggregate
    lc_path = Path(cfg["sources"]["landcover"]["out_dir"]) / "landcover.tif"
    with rasterio.open(str(lc_path)) as src:
        codes, transform, crs, nodata = src.read(1), src.transform, src.crs, src.nodata
    valid = np.ones(codes.shape, "float32") if nodata is None \
        else (codes != nodata).astype("float32")
    nlcd = np.nan_to_num(_aggregate(valid, transform, crs, tg), nan=0.0)

    # ERA5-Land extent, read from the raw downloads rather than the config, so a
    # config edit cannot make this flag lie.
    LAT, LON = tg.latlon()
    era5_dir = Path(cfg["sources"]["era5_land"]["out_dir"])
    files = sorted(glob.glob(str(era5_dir / "*.nc")))
    if files:
        d = xr.open_dataset(files[0])
        la = d["latitude"].values if "latitude" in d.coords else d["lat"].values
        lo = d["longitude"].values if "longitude" in d.coords else d["lon"].values
        era5 = ((LAT < la.min()) | (LAT > la.max())
                | (LON < lo.min()) | (LON > lo.max()))
    else:
        print("  [warn] no ERA5-Land files found; era5_extrapolated set to 0")
        era5 = np.zeros(tg.shape, bool)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, nlcd=nlcd, era5=era5, shape=np.array(tg.shape))
    return nlcd, era5


def main(day: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import (LinearSegmentedColormap,
                                   ListedColormap, TwoSlopeNorm)

    seq = LinearSegmentedColormap.from_list(
        "s", ["#e8f1fd", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
    div = LinearSegmentedColormap.from_list(
        "d", ["#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a2", "#e34948"])
    for c in (seq, div):
        c.set_bad(NODATA)
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                         "savefig.facecolor": SURF, "text.color": INK,
                         "font.size": 11})

    cfg = load_config("configs/data_southcentral.yaml")
    tg = build_target_grid(cfg["domain"], cfg["grid"])
    LAT, LON = tg.latlon()
    ny, nx = tg.shape

    # The ablation trained on the 21-channel SUPERSET store, which is what these
    # extracts must describe - not whatever out_zarr the domain config points at.
    zarr_path = REPO / "data_store_sc/super/dataset.zarr"
    store = xr.open_zarr(zarr_path, consolidated=True)
    names = store["channel_in"].values.tolist()
    times = pd.DatetimeIndex(store["time"].values)
    want = pd.Timestamp(day)
    if not (times.min() <= want <= times.max()):
        raise SystemExit(f"--day {day} is outside the store's range "
                         f"{times.min().date()}..{times.max().date()}")
    k = int(np.argmin(np.abs(times - want)))
    day = times[k].strftime("%Y-%m-%d")
    split = str(store["split"].values[k])

    X = store["input"].isel(time=k).values
    Y = store["target"].isel(time=k).values[0]
    doy = times[k].dayofyear - 1
    clim_t = store["clim_tmp"].values[doy]
    clim_c = store["clim_coarse_tmp"].values[doy]
    land = X[names.index("land_mask")] > 0.5
    i0, i1, j0, j1 = holdout_bounds(store, cfg["holdout"])
    hold = np.zeros((ny, nx), bool)
    hold[i0:i1, j0:j1] = True
    ch = {n: X[names.index(n)] for n in names}
    lc_names = [n for n in names if n.startswith("lc_")]
    nlcd_affected = NLCD_DERIVED | set(lc_names)

    print(f"store {zarr_path.relative_to(REPO)}")
    print(f"reference day {day} (index {k}, '{split}' split)")

    nlcd_frac, era5_out = provenance_masks(tg, cfg)
    # Two thresholds, because zero-coverage is not the only failure mode.
    # nlcd_fill  : no real NLCD pixel at all -> the value is pure fill.
    # nlcd_unrel : under half the cell is covered, so the composition is
    #              extrapolated from a minority sliver and is not trustworthy
    #              even though it is technically "measured". Verified that this
    #              cut flags the 36 fabricated Coahuila cells while leaving
    #              Lubbock, Abilene, Midland and Odessa (valid_frac = 1.0)
    #              correctly unflagged.
    nlcd_fill = nlcd_frac <= 0.0
    nlcd_unrel = nlcd_frac < 0.5
    nlcd_bad = nlcd_unrel            # what the maps grey out
    nland = land.sum()
    print(f"held-out region rows {i0}:{i1} cols {j0}:{j1} = "
          f"{(i1 - i0) * 10}x{(j1 - j0) * 10} km, {int((hold & land).sum())} "
          f"land cells ({100 * (hold & land).sum() / nland:.1f}% of land)")
    print(f"nlcd_extrapolated (no real pixel): {int((nlcd_fill & land).sum())} land "
          f"cells ({100 * (nlcd_fill & land).sum() / nland:.1f}%), "
          f"{int((nlcd_fill & land & hold).sum())} inside the held-out region")
    print(f"nlcd_unreliable (<50% covered):    {int((nlcd_unrel & land).sum())} land "
          f"cells ({100 * (nlcd_unrel & land).sum() / nland:.1f}%), "
          f"{int((nlcd_unrel & land & hold).sum())} inside the held-out region")
    print(f"era5_extrapolated: {int((era5_out & land).sum())} land cells, "
          f"{int((era5_out & land & hold).sum())} inside the held-out region")

    # ---------------------------------------------------------------- maps --
    hold_lat = (LAT[i0:i1, j0:j1].min(), LAT[i0:i1, j0:j1].max())
    hold_lon = (LON[i0:i1, j0:j1].min(), LON[i0:i1, j0:j1].max())
    KEY_GEO = ("Orange outline = region held back from training and used for "
               "scoring. Dotted lines = latitude / longitude, labelled on the "
               "left and bottom edges.")
    KEY_NLCD = "Grey = no land-cover survey coverage there. " + KEY_GEO

    def graticule(ax):
        ax.contour(LON, levels=LON_LEVELS, colors=GRAT, linewidths=.5,
                   alpha=.45, linestyles=":")
        ax.contour(LAT, levels=LAT_LEVELS, colors=GRAT, linewidths=.5,
                   alpha=.45, linestyles=":")

        def edge(profile, levels, n):
            asc = profile[-1] > profile[0]
            xp = profile if asc else profile[::-1]
            fp = np.arange(n) if asc else np.arange(n)[::-1]
            t = [(np.interp(L, xp, fp), L) for L in levels
                 if profile.min() <= L <= profile.max()]
            return [a for a, _ in t], [b for _, b in t]

        yt, yl = edge(LAT[:, 0], LAT_LEVELS, ny)
        ax.set_yticks(yt)
        ax.set_yticklabels([f"{v:.0f}N" for v in yl], fontsize=7.5, color=INK3)
        xt, xl = edge(LON[-1, :], LON_LEVELS, nx)
        ax.set_xticks(xt)
        ax.set_xticklabels([f"{abs(v):.0f}W" for v in xl], fontsize=7.5, color=INK3)
        ax.tick_params(length=2.5, color=GRAT, pad=1.5)

    def sheet(keys, title, sub, out, ncol, diverging=()):
        keys = [k_ for k_ in keys if k_ not in SPATIALLY_CONSTANT]
        ncol = min(ncol, max(1, len(keys)))
        nrow = int(np.ceil(len(keys) / ncol))
        if nrow > 1:                                 # balance the last row rather
            ncol = int(np.ceil(len(keys) / nrow))    # than leaving 3 blanks of 4
            nrow = int(np.ceil(len(keys) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.8 * nrow),
                                 squeeze=False)
        for ax, key in zip(np.ravel(axes), keys):
            raw = ch[key]
            lo, hi = float(np.nanmin(raw)), float(np.nanmax(raw))
            shown = raw.copy()
            if key in nlcd_affected:          # never render a fill as measurement
                shown = np.where(nlcd_bad, np.nan, shown)
            v = np.ma.masked_invalid(shown)
            if key in CATEGORICAL:
                # Discrete scale: one band per class code present, so the bar
                # cannot read as a continuous ramp and the ticks cannot collide.
                # The codes are labels - 95 is not "8.6x" 11 - so the spacing
                # between them must carry no meaning.
                codes = np.unique(raw[np.isfinite(raw)]).astype(int)
                idx = np.full(v.shape, np.nan)
                for t, c in enumerate(codes):
                    idx[np.asarray(shown) == c] = t
                qual = ListedColormap(
                    [plt.get_cmap("tab20")(t % 20) for t in range(len(codes))])
                qual.set_bad(NODATA)
                im = ax.imshow(np.ma.masked_invalid(idx), cmap=qual,
                               vmin=-.5, vmax=len(codes) - .5)
                cb = fig.colorbar(im, ax=ax, fraction=.042, pad=.02,
                                  ticks=range(len(codes)))
                cb.ax.set_yticklabels([str(c) for c in codes])
                note = f"{len(codes)} classes, categorical - codes are labels"
            elif key in diverging:
                lim = float(np.nanpercentile(np.abs(raw), 99))
                im = ax.imshow(v, cmap=div, norm=TwoSlopeNorm(0, -lim, lim))
                cb = fig.colorbar(im, ax=ax, fraction=.042, pad=.02)
                note = f"true {lo:+.3g} to {hi:+.3g}, white = 0"
            else:
                im = ax.imshow(v, cmap=seq, vmin=np.nanpercentile(raw, 1),
                               vmax=np.nanpercentile(raw, 99))
                cb = fig.colorbar(im, ax=ax, fraction=.042, pad=.02)
                note = f"true {lo:.3g} to {hi:.3g}"
            cb.set_label(UNITS.get(key, "fraction"), fontsize=7.5, color=INK3)
            cb.ax.tick_params(labelsize=7, color=GRAT, labelcolor=INK3)
            graticule(ax)
            ax.add_patch(plt.Rectangle((j0 - .5, i0 - .5), j1 - j0, i1 - i0,
                                       fill=False, ec=ORANGE, lw=1.6, zorder=6))
            # A light stroke so the label reads over dark fill as well as light.
            ax.text((j0 + j1) / 2, i0 - 1.6, "HELD-OUT REGION", ha="center",
                    va="bottom", fontsize=6.8, color=ORANGE,
                    fontweight="bold", zorder=7,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground=SURF)])
            label = f"{key}  ({day})" if key in TIME_VARYING else key
            ax.set_title(label, fontsize=11.5, color=INK,
                         fontweight="semibold", pad=13)
            ax.text(.5, 1.006, note, transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=7, color=INK3)
        for ax in np.ravel(axes)[len(keys):]:
            ax.axis("off")
        fig.suptitle(title, x=.02, y=1.0, ha="left", va="top", fontsize=17,
                     fontweight="semibold")
        fig.text(.02, .975, sub, ha="left", va="top", fontsize=9.4, color=INK2,
                 wrap=True)
        # h_pad reserves room for the per-panel "true range" note, which is a
        # manual text artist and so invisible to tight_layout's bbox maths.
        fig.tight_layout(rect=(0, 0, 1, .945 if nrow > 2 else .915), h_pad=3.2)
        fig.savefig(out, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"  {Path(out).name}  ({nrow}x{ncol}, {len(keys)} panels)")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    print("\nmaps:")
    DIVERGING = ("tpi", "northness", "eastness")
    sheet(["dem", "elev_std", "slope_mean", "tpi", "northness", "eastness"],
          "Terrain covariates - South-Central domain",
          KEY_GEO,
          FIGDIR / "proto_terrain.png", 3, diverging=DIVERGING)
    sheet(lc_names, "Land-cover composition - South-Central domain",
          KEY_NLCD, FIGDIR / "proto_landcover.png", 4)

    VARIANTS = variant_channels()
    for vname, chans in VARIANTS.items():
        skipped = [c for c in chans if c in SPATIALLY_CONSTANT]
        key = KEY_NLCD if set(chans) & nlcd_affected else KEY_GEO
        if skipped:
            key = f"{', '.join(skipped)} not shown (same value everywhere). {key}"
        sheet(chans, f"{vname} - the exact input stack", key,
              FIGDIR / f"variant_{vname}.png", 4, diverging=DIVERGING)

    # ---------------------------------------------------------------- CSVs --
    CSVDIR.mkdir(parents=True, exist_ok=True)
    print("\ncsv:")
    FLAGS = ["in_heldout_region", "is_land", "nlcd_extrapolated",
             "nlcd_unreliable", "nlcd_valid_frac", "era5_extrapolated"]

    def flagvals(i, j):
        return [int(hold[i, j]), int(land[i, j]), int(nlcd_fill[i, j]),
                int(nlcd_unrel[i, j]), f"{nlcd_frac[i, j]:.4f}",
                int(era5_out[i, j])]

    static = [n for n in names if n not in TIME_VARYING]
    out = REPO / "outputs" / "training_store_cells.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "lat", "lon", "x_albers_m", "y_albers_m"]
                   + FLAGS + ["clim_tmp_annual_mean_degC"] + static)
        cl = store["clim_tmp"].values.mean(axis=0)
        for i in range(ny):
            for j in range(nx):
                w.writerow([i, j, f"{LAT[i, j]:.5f}", f"{LON[i, j]:.5f}",
                            f"{tg.x[j]:.1f}", f"{tg.y[i]:.1f}"]
                           + flagvals(i, j) + [f"{cl[i, j]:.3f}"]
                           + [f"{ch[n][i, j]:.5f}" for n in static])
    print(f"  {out.relative_to(REPO)}  ({ny * nx} rows, {len(static) + 12} cols)")

    for vname, chans in VARIANTS.items():
        path = CSVDIR / f"{vname}_cells.csv"
        hdr = (["row", "col", "lat", "lon"] + FLAGS
               + [f"{c}__{day}" if c in TIME_VARYING else c for c in chans]
               + [f"target_tmp_anomaly__{day}", f"target_tmp_degC__{day}",
                  f"coarse_tmp_degC__{day}"])
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            for i in range(ny):
                for j in range(nx):
                    w.writerow([i, j, f"{LAT[i, j]:.5f}", f"{LON[i, j]:.5f}"]
                               + flagvals(i, j)
                               + [f"{ch[c][i, j]:.5f}" for c in chans]
                               + [f"{Y[i, j]:.5f}",
                                  f"{Y[i, j] + clim_t[i, j]:.3f}",
                                  f"{ch['coarse_tmp'][i, j] + clim_c[i, j]:.3f}"])
        print(f"  {path.relative_to(REPO)}  ({len(chans)} channels, {len(hdr)} cols)")

    # -------------------------------------------------------------- README --
    from data.build_dataset import NO_NORMALIZE
    v2 = VARIANTS["v2_landcov"]
    v2lc = [c for c in v2 if c.startswith("lc_")]
    v2sum, allsum = sum(ch[c] for c in v2lc), sum(ch[c] for c in lc_names)
    v2_raw = [c for c in v2 if c in NO_NORMALIZE]
    real = land & ~nlcd_bad
    uf_r = float(np.corrcoef(ch["urban_frac"][land], ch["lc_developed"][land])[0, 1])
    uf_rr = float(np.corrcoef(ch["urban_frac"][real], ch["lc_developed"][real])[0, 1])
    uf_gap = float(np.abs(ch["urban_frac"] - ch["lc_developed"]).max())
    pct = lambda m: 100 * (m & land).sum() / nland   # noqa: E731

    L = [
        "Per-variant training-store extracts - South-Central domain",
        "=" * 58, "",
        "One CSV per covariate variant. Each row is ONE 10 km grid cell;",
        f"{ny * nx} rows = {ny} x {nx} cells. Channel lists are parsed from",
        "slurm/covariate_ablation_body.sh, so they match exactly what trained.",
        f"Also shipped: outputs/training_store_cells.csv - the same {ny * nx} rows",
        f"with ALL {len(static)} static channels plus the Albers x/y metres and the",
        "annual-mean climatology, i.e. the superset the variants select from.",
        "", "Geometry", "-" * 58,
        "  lat, lon          cell CENTRE, degrees",
        "  row, col          index into the (y, x) grid; row 0 is the NORTH edge",
        "  x_albers_m/y_..   EPSG:5070 cell-centre metres (superset file only)",
        "  The grid is EPSG:5070 Albers at 10 km. Latitude varies ~0.3 deg along a",
        "  single row, so lat/lon CANNOT be reconstructed from row/col linearly.",
        "", "Flags", "-" * 58,
        "  in_heldout_region 1 = the region withheld from training AND evaluated.",
        f"                    rows {i0}:{i1}, cols {j0}:{j1} = "
        f"{(i1 - i0) * 10} x {(j1 - j0) * 10} km, {int((hold & land).sum())} land "
        f"cells ({pct(hold):.1f}% of land),",
        f"                    {hold_lat[0]:.2f}-{hold_lat[1]:.2f} N, "
        f"{abs(hold_lon[1]):.2f}-{abs(hold_lon[0]):.2f} W.",
        "                    NOT the Austin box alone. configs/data_southcentral.yaml",
        "                    gives the Austin box as lon -98.6..-97.0, lat 29.7..31.0",
        "                    (140 x 150 km, 210 land cells) and buffer_deg: 1.0.",
        "                    holdout_bounds() applies that buffer, and the SAME bounds",
        "                    serve both the loss mask and the evaluation crop - so the",
        "                    evaluated area is 6.2x the Austin box and also contains",
        "                    San Antonio, Waco, Killeen and College Station.",
        "                    These cells receive ZERO training gradient.",
        "  is_land           1 = ERA5-Land has truth here. 0 cells are excluded from",
        "                    the loss and from every metric.",
        "  nlcd_extrapolated 1 = NO real NLCD pixel in this cell, so landcover,",
        "                    urban_frac and every lc_* value is nearest-neighbour",
        f"                    extrapolation. {int((nlcd_fill & land).sum())} land cells "
        f"({pct(nlcd_fill):.1f}% of land),",
        f"                    {int((nlcd_fill & land & hold).sum())} of them inside the "
        f"held-out region.",
        "  nlcd_unreliable   1 = under HALF the cell has real NLCD, so its land-cover",
        "                    composition comes from a minority sliver and should not",
        "                    be read as measurement even where it is not pure fill.",
        f"                    {int((nlcd_unrel & land).sum())} land cells "
        f"({pct(nlcd_unrel):.1f}% of land), "
        f"{int((nlcd_unrel & land & hold).sum())} inside the held-out region.",
        "                    This is the flag to filter on. It was checked against",
        "                    ground truth: it catches the 36 fabricated Coahuila",
        "                    cells while leaving Lubbock, Abilene, Midland and",
        "                    Odessa (valid_frac = 1.0) correctly unflagged, so it",
        "                    does not discard real West Texas urban cells.",
        "  nlcd_valid_frac   fraction of the cell covered by real NLCD pixels, from",
        "                    area-averaging the raster's valid mask with the same",
        "                    reprojection the covariates use. 0.0 <=> extrapolated;",
        "                    < 0.5 <=> unreliable.",
        "  era5_extrapolated 1 = the cell centre falls outside the ERA5-Land download",
        "                    extent, so its TRUTH is extrapolated too.",
        f"                    {int((era5_out & land).sum())} land cells, "
        f"{int((era5_out & land & hold).sum())} inside the held-out region.",
        "",
        "KNOWN DATA CAVEAT - read before using the land-cover columns",
        "-" * 58,
        "NLCD 2021 is CONUS-only, but this domain reaches 26.67 N, roughly 470 km",
        "into Mexico. data/build_dataset.py _fill_nan_nearest fills the gap by",
        "nearest neighbour, and the fill is invisible in the values themselves.",
        "",
        "It is worse than a plain nearest-neighbour fill. landcover_fractions()",
        "excludes nodata from the DENOMINATOR (data/covariates.py:170), so a donor",
        "cell with only ~9% real NLCD coverage still emits a composition summing to",
        "1.0 - and nearest neighbour then broadcasts that sliver up to ~150 km.",
        "Consequences present in these files and on the land-cover slides:",
        "  - 35 cells at 27.6-28.7 N, 101.9-99.6 W carry landcover = 23",
        "    ('Developed, Medium Intensity') and lc_developed up to 0.711. That is",
        "    rural Coahuila desert. There is no city there.",
        "  - 309 cells west of 99.5 W and south of 30 N carry lc_wetland > 0.5",
        "    ('Emergent Herbaceous Wetlands') in the Chihuahuan Desert.",
        "  - urban_frac and lc_developed are the same quantity, but urban_frac counts",
        "    nodata as 'not developed' (0/N, never NaN) so it was never filled and is",
        "    hard-zero there, while lc_developed reaches 0.71. They therefore disagree",
        f"    by up to {uf_gap:.3f}. Measured corr over all land = {uf_r:.4f}; over",
        f"    cells with real NLCD only = {uf_rr:.4f}. The r = 0.996 quoted in",
        "    configs/data_sc_super.yaml and README.md was measured on the NLCD-covered",
        "    subset, not on the grid, and it is the stated reason urban_frac was",
        "    dropped from V2 - so that justification is weaker than it reads.",
        "",
        "What this does NOT affect:",
        f"  - The held-out region: {int((nlcd_bad & land & hold).sum())} flagged cells "
        f"fall inside it, so the reported",
        "    metrics are computed on measured land cover and real reanalysis truth.",
        "  - The ablation ranking: one superset store was built once and every variant",
        "    selects from it, so all arms saw bit-identical data.",
        "  - dem and coastal_dist, which come from global sources.",
        "",
        "Values - RAW PHYSICAL UNITS, not normalized",
        "-" * 58,
        "The Zarr store holds unnormalized values. For MOST channels z-scoring happens",
        "at LOAD time in training/dataset.py __getitem__, using norm_stats.json fit on",
        "the TRAIN split only. The exception is data/build_dataset.py NO_NORMALIZE:",
        f"  {sorted(NO_NORMALIZE)}",
        f"which are NEVER z-scored and reach the network on their natural scale -",
        f"that is {len(v2_raw)} of V2's {len(v2)} channels ({', '.join(v2_raw)}).",
        "z-scoring a near-absent class would amplify rounding noise into signal.",
        "",
        "    coarse_tmp     degC ANOMALY (day-of-year climatology removed)",
        "    dem            metres above sea level",
        "    coastal_dist   km to the nearest coastline",
        "    elev_std       metres, sub-grid elevation sd (roughness)",
        "    slope_mean     degrees, area-averaged from ~90 m",
        "    tpi            metres, cell minus its neighbourhood mean",
        "    northness      cos(aspect), -1 south-facing .. +1 north-facing",
        "    eastness       sin(aspect), -1 west-facing .. +1 east-facing",
        "                   aspect is averaged as a VECTOR and slope-weighted, so a",
        "                   flat cell tends to 0 rather than to a random compass",
        "                   direction; consequently no cell reaches exactly +/-1.",
        "    urban_frac     fraction 0-1 of the cell that is NLCD developed, with",
        "                   nodata counted as NOT developed (see caveat above)",
        "    lc_*           fraction 0-1 per NLCD Level-1 group, nodata EXCLUDED from",
        f"                   the denominator. Across all {len(lc_names)} groups they "
        f"sum to {float(np.nanmin(allsum)):.3f}-{float(np.nanmax(allsum)):.3f}.",
        "                   BUT v2_landcov_cells.csv drops lc_barren, so in THAT file",
        f"                   the lc_ columns sum to {float(np.nanmin(v2sum)):.3f}-"
        f"{float(np.nanmax(v2sum)):.3f}, not to 1.",
        "    land_mask      0/1, 1 = ERA5-Land has truth (same as is_land)",
        "    doy_sin/cos    dimensionless -1..1, day-of-year phase. Spatially",
        "                   CONSTANT, so they are omitted from the map sheets.",
        "    landcover      NLCD CLASS CODE, CATEGORICAL. The integer is a label, not",
        "                   a quantity: 95 is not '8.6x' 11, and the majority class",
        "                   discards everything else in a 10 km cell. Replacing it",
        "                   with the lc_* fractions is what V2 tests.",
        "",
        f"  Columns ending __{day} are time-varying, given for that one day, which",
        f"  falls in the '{split}' split. Everything else is static.",
        "  target_tmp_anomaly is the ERA5-Land truth in anomaly space - what the",
        "  network is actually trained to predict. target_tmp_degC and coarse_tmp_degC",
        "  add that day's day-of-year climatology back to give absolute degC.",
        "",
        "Empty cells",
        "-" * 58,
        f"  target_tmp_anomaly__{day} and target_tmp_degC__{day} are EMPTY (read as",
        "  NaN by pandas) on the 790 cells where is_land = 0. ERA5-Land has no truth",
        "  over the Gulf of Mexico, so there is nothing to report there; those cells",
        "  are excluded from the loss and from every metric. Every other column is",
        "  fully populated on all rows. If you open these files in Excel the blanks",
        "  will show as empty cells rather than zeros - do not fill them with 0.",
        "",
        "What V2 actually showed",
        "-" * 58,
        "V2 is the only variant that improved anything, but it is not a clean win and",
        "it is not one change. It BOTH swaps the majority class for the 7-group",
        "composition AND drops landcover, urban_frac and lc_barren, so those effects",
        "are not separated. Against the V0 baseline it gains SSIM +0.0084 while RMSE",
        "gets slightly WORSE, 0.7817 -> 0.7834 degC. It was adopted as the project",
        "default on the SSIM gain alone; see configs/data_sc_v2.yaml and README.md",
        "section 5.2 for the pre-registered decision rule and the between-seed spread",
        "any claimed improvement has to clear.",
        "",
        "Variants", "-" * 58,
    ]
    L += [f"  {v:12s} {len(c):2d} ch : {' '.join(c)}" for v, c in VARIANTS.items()]
    L += ["",
          "Generated by scripts/make_variant_outputs.py from",
          f"{zarr_path.relative_to(REPO)} ({len(names)}-channel superset).",
          "Re-run that script to regenerate every file in this directory.", ""]
    readme = CSVDIR / "README.txt"
    readme.write_text("\n".join(L), encoding="ascii", errors="backslashreplace")
    print(f"  {readme.relative_to(REPO)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default="2023-07-20",
                    help="reference day for the time-varying channels "
                         "(default: the day used in the published field figures)")
    main(ap.parse_args().day)
