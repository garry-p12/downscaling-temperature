"""Pre-training sanity checks on the assembled dataset.

Run before any training. Every check here corresponds to a failure that has
actually happened in this project and would otherwise have been invisible
until the results looked odd:

  * a channel silently all-zero (coastal_dist before the coastline was wired)
  * a covariate mosaic that merged to 100% nodata (NLCD tiles)
  * NaN reaching the network and killing the loss (ERA5-Land ocean)
  * a target grid one row off the prediction (odd-size patch embed)
  * a holdout region too small to give a stable metric

Exits non-zero if any HARD check fails, so it can gate a sweep.

Usage:
    DOWNSCALE_CONFIG_data=configs/data_southcentral.yaml \\
        python -m data.sanity_check
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import xarray as xr

from common import load_config

OK, WARN, FAIL = "  ok  ", " WARN ", " FAIL "


def main(zarr: str | None = None) -> int:
    cfg = load_config("data")
    path = zarr or cfg["dataset"]["out_zarr"]
    ds = xr.open_zarr(path, consolidated=True)
    hard_fail = False

    def report(status, msg):
        nonlocal hard_fail
        if status is FAIL:
            hard_fail = True
        print(f"[{status}] {msg}")

    print(f"\n=== dataset: {path}")
    print(f"dims {dict(ds.sizes)}\n")

    # --- splits ---------------------------------------------------------- #
    split = ds["split"].values
    counts = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    times = pd.DatetimeIndex(ds["time"].values)
    print(f"--- splits: {counts}")
    for name, n in counts.items():
        report(OK if n > 0 else FAIL, f"{name}: {n} days")
    # Contiguous blocks, never interleaved — otherwise the temporal split leaks.
    for name in ("train", "val", "test"):
        sel = np.where(split == name)[0]
        if len(sel):
            contiguous = (sel.max() - sel.min() + 1) == len(sel)
            report(OK if contiguous else FAIL,
                   f"{name} is a contiguous block "
                   f"({times[sel.min()].date()}..{times[sel.max()].date()})")

    # --- channels -------------------------------------------------------- #
    print("\n--- input channels")
    inp = ds["input"].values
    for i, name in enumerate(ds["channel_in"].values.tolist()):
        v = inp[:, i]
        nan = float(np.isnan(v).mean())
        lo, hi, sd = float(np.nanmin(v)), float(np.nanmax(v)), float(np.nanstd(v))
        line = f"{name:13s} [{lo:9.3f}, {hi:9.3f}] sd {sd:8.3f} nan {nan:.4f}"
        if nan > 0:
            report(FAIL, line + "  <- NaN reaches the network")
        elif sd == 0:
            report(FAIL, line + "  <- CONSTANT: dead channel, carries no signal")
        else:
            report(OK, line)

    print("\n--- target")
    tgt = ds["target"].values
    for i, name in enumerate(ds["channel_out"].values.tolist()):
        v = tgt[:, i]
        nan = float(np.isnan(v).mean())
        report(OK, f"{name:13s} [{np.nanmin(v):9.3f}, {np.nanmax(v):9.3f}] "
                   f"sd {np.nanstd(v):8.3f} nan {nan:.4f}")

    # --- land mask vs target NaN ----------------------------------------- #
    names = ds["channel_in"].values.tolist()
    if "land_mask" in names:
        mask = inp[0, names.index("land_mask")] > 0.5
        tnan = np.isnan(tgt[:, 0]).any(axis=0)
        agree = float((mask != tnan).mean())
        report(OK if agree > 0.97 else WARN,
               f"land_mask agrees with target-NaN pattern on {100*agree:.1f}% "
               f"of cells ({int(mask.sum())} land / {int((~mask).sum())} ocean)")

    # --- anomalies ------------------------------------------------------- #
    print("\n--- anomalies / climatology")
    if ds.attrs.get("anomaly"):
        lm_all = (inp[0, names.index("land_mask")] > 0.5) \
            if "land_mask" in names else np.ones(tgt.shape[-2:], bool)
        for v in ("clim_tmp", "clim_coarse_tmp"):
            if v in ds:
                c = ds[v].values
                # Finiteness is only required on LAND — ocean has no truth, and
                # those cells are masked out of the loss and every metric.
                ok = bool(np.isfinite(c[:, lm_all]).all())
                report(OK if ok else FAIL,
                       f"{v}: {c.shape} range [{np.nanmin(c):.2f}, "
                       f"{np.nanmax(c):.2f}] degC, finite-on-land={ok}")
        # A correct anomaly field is near-zero-mean on the TRAIN split.
        tr = split == "train"
        m = float(np.nanmean(tgt[tr, 0]))
        report(OK if abs(m) < 0.5 else WARN,
               f"train target anomaly mean {m:+.3f} degC (should be ~0)")
    else:
        report(WARN, "anomaly mode OFF — targets are absolute temperature")

    # --- holdout --------------------------------------------------------- #
    print("\n--- spatial holdout")
    holdout = cfg.get("holdout")
    if holdout:
        from training.dataset import holdout_bounds

        i0, i1, j0, j1 = holdout_bounds(ds, holdout)
        cells = (i1 - i0) * (j1 - j0)
        total = ds.sizes["y"] * ds.sizes["x"]
        report(OK, f"'{holdout['name']}' rows {i0}:{i1} cols {j0}:{j1} "
                   f"= {i1-i0}x{j1-j0} = {cells} cells "
                   f"({100*cells/total:.1f}% of domain)")
        # Small regions give noisy metrics; say so rather than let it surprise.
        report(OK if cells >= 400 else WARN,
               f"holdout has {cells} cells — "
               f"{'enough for stable metrics' if cells >= 400 else 'SMALL: report bootstrap CIs, not point estimates'}")
        if "land_mask" in names:
            land = int((inp[0, names.index('land_mask')][i0:i1, j0:j1] > 0.5).sum())
            report(OK if land > 0 else FAIL,
                   f"holdout land cells: {land}/{cells}")
    else:
        report(WARN, "no holdout configured — evaluation is temporal only")

    # --- baseline floor --------------------------------------------------- #
    print("\n--- baseline floor (test split, land only, absolute degC)")
    te = split == "test"
    ci = names.index("coarse_tmp")
    coarse, truth = inp[te, ci], tgt[te, 0]
    if ds.attrs.get("anomaly"):
        doy = pd.DatetimeIndex(ds["time"].values[te]).dayofyear.values - 1
        coarse = coarse + ds["clim_coarse_tmp"].values[doy]
        truth = truth + ds["clim_tmp"].values[doy]
    if "land_mask" in names:
        lm = inp[0, names.index("land_mask")] > 0.5
        coarse, truth = np.where(lm, coarse, np.nan), np.where(lm, truth, np.nan)
    rmse = float(np.sqrt(np.nanmean((coarse - truth) ** 2)))
    report(OK, f"interpolated-POWER RMSE = {rmse:.3f} degC  "
               f"(bias {np.nanmean(coarse - truth):+.3f}) — the floor to beat")

    print()
    if hard_fail:
        print("SANITY CHECK FAILED — do not train on this dataset.\n")
        return 1
    print("sanity checks passed.\n")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default=None)
    args = ap.parse_args()
    sys.exit(main(args.zarr))
