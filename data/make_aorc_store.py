"""Build an AORC-truth store by swapping the TARGET of an existing store.

A full rebuild would re-read the 630 MB DEM and the NLCD mosaic to regenerate
static channels that are already correct — the AORC experiment changes only
which product the model is trained TOWARD, not what it is shown. So this copies
the source store's inputs verbatim and replaces the target, its climatology and
the land mask.

WHY THE LAND MASK IS AN INTERSECTION
    AORC is a CONUS analysis and ERA5-Land is not: ~74% of this domain's cells
    south of 29 N are finite in AORC against 99%+ for ERA5-Land. Training on
    each product's own coverage would confound "different truth" with
    "different cells". The mask here is the INTERSECTION, so the two arms see
    an identical set of scored cells and the only difference is the target.

WHY THE CLIMATOLOGY IS REFIT
    The anomaly framing subtracts a day-of-year climatology fit on train years.
    That climatology belongs to the product, so AORC needs its own; reusing
    ERA5-Land's would push the product difference into the anomaly and make the
    comparison meaningless.

Usage:
    DOWNSCALE_CONFIG_data=configs/data_sc_aorc.yaml \\
        python -m data.make_aorc_store
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common import load_config                     # noqa: E402
from common.grid import build_target_grid          # noqa: E402
from common.normalize import Normalizer            # noqa: E402
from data.build_dataset import doy_climatology     # noqa: E402


def main(src_zarr: str, out_zarr: str, year_glob: str = "20*"):
    from data.regrid import field_to_target

    cfg = load_config("data")
    tgrid = build_target_grid(cfg["domain"], cfg["grid"])
    src = xr.open_zarr(src_zarr, consolidated=True)
    names = src["channel_in"].values.tolist()
    times = pd.DatetimeIndex(src["time"].values)
    split = src["split"].values

    # --- AORC daily, every year, block-meaned onto the target grid
    adir = Path(cfg["sources"]["aorc"]["out_dir"])
    var = cfg["sources"]["aorc"]["variables"][0]
    files = sorted(adir.glob(f"aorc_{cfg['domain']['name']}_{year_glob}_daily.zarr"))
    if not files:
        raise SystemExit(f"no AORC daily stores in {adir}")
    print(f"[aorc] {len(files)} yearly store(s)")
    parts = []
    for f in files:
        d = xr.open_zarr(f, consolidated=False)[var]
        if "latitude" in d.dims:
            d = d.rename({"latitude": "lat", "longitude": "lon"})
        if float(d.max()) > 200:
            d = d - 273.15
        g = field_to_target(d, tgrid, "conservative")
        print(f"  {f.name}: {g.sizes['time']} days, "
              f"{100 * np.isfinite(g.values).mean():.1f}% cells finite")
        parts.append(g)
    aorc = xr.concat(parts, dim="time").sortby("time")

    at = pd.DatetimeIndex(aorc["time"].values).normalize()
    idx = pd.Series(np.arange(len(at)), index=at).reindex(times.normalize())
    if idx.isna().any():
        raise SystemExit(f"{int(idx.isna().sum())} store days missing from AORC")
    A = aorc.values[idx.values.astype(int)]                    # (T, y, x) degC

    # --- masks. Intersection, so both arms score identical cells.
    lm_i = names.index("land_mask")
    era_land = src["input"].values[:, lm_i][0] > 0.5
    aorc_ok = np.isfinite(A).all(axis=0)
    land = era_land & aorc_ok
    print(f"\n[mask] ERA5-Land cells {int(era_land.sum())}, AORC {int(aorc_ok.sum())}, "
          f"intersection {int(land.sum())} "
          f"({100 * land.sum() / era_land.sum():.1f}% of the ERA5-Land mask)")

    # --- anomaly: AORC needs its OWN day-of-year climatology, train years only
    A = np.where(land[None], A, np.nan).astype("float32")
    train = split == "train"
    smooth = int((cfg.get("anomaly") or {}).get("smooth_days", 15))
    clim = doy_climatology(A, times, train, smooth)
    doy0 = times.dayofyear.values - 1
    tgt = (A - clim[doy0])[:, None]                            # (T, 1, y, x)
    print(f"[anomaly] AORC target anomaly sd {np.nanstd(tgt):.3f} degC "
          f"(ERA5-Land was {np.nanstd(src['target'].values):.3f})")

    # --- inputs copied verbatim, except land_mask which follows the new mask
    inp = src["input"].values.copy()
    inp[:, lm_i] = land[None].astype("float32")

    out = xr.Dataset(
        {"input": (("time", "channel_in", "y", "x"), inp),
         "target": (("time", "channel_out", "y", "x"), tgt),
         "split": ("time", split),
         "clim_tmp": (("doy", "y", "x"), clim),
         "clim_coarse_tmp": (("doy", "y", "x"), src["clim_coarse_tmp"].values)},
        coords={"time": src["time"].values, "channel_in": names,
                "channel_out": ["tmp"], "y": tgrid.y, "x": tgrid.x,
                "doy": np.arange(clim.shape[0])},
        attrs=dict(src.attrs) | {"truth_source": "aorc"})

    Path(out_zarr).parent.mkdir(parents=True, exist_ok=True)
    out.chunk({"time": 64}).to_zarr(out_zarr, mode="w", consolidated=True)

    # Normalizer: target stats change with the product; inputs are unchanged, so
    # reuse the source's entries and refit only out::tmp.
    nz = Normalizer.load(str(Path(src_zarr).parent / "norm_stats.json"))
    nz.fit("out::tmp", tgt[train][np.isfinite(tgt[train])])
    nz.save(Path(out_zarr).parent / "norm_stats.json")
    print(f"\n[aorc] wrote {out_zarr} and norm_stats.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data_store_sc/super/dataset.zarr")
    ap.add_argument("--out", default="data_store_sc/aorc/dataset.zarr")
    a = ap.parse_args()
    main(a.src, a.out)
