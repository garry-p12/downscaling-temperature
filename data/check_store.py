"""Exit 0 only if a store is complete and usable.

WHY THIS EXISTS
    A build guard that tests `[ -d store/dataset.zarr ]` passes for a directory
    a killed build left half-written. That happened here: a store whose build
    died before consolidating metadata was reported as "already built", and the
    next job would have trained on it. Existence is not validity.

Usage:
    python -m data.check_store data_store_sc/ix_aorc/dataset.zarr --channels 34 --days 1826
"""
from __future__ import annotations

import argparse
import os
import sys

if os.environ.get("DOWNSCALE_MAX_WORKERS"):
    import dask
    import zarr
    zarr.config.set({"threading.max_workers": int(os.environ["DOWNSCALE_MAX_WORKERS"])})
    dask.config.set(scheduler="synchronous")


def main(path, channels, days, require):
    import numpy as np
    import xarray as xr
    try:
        d = xr.open_zarr(path, consolidated=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"[check] {path}: unreadable — {type(e).__name__}: {str(e)[:120]}")
        return 1
    names = d["channel_in"].values.tolist()
    bad = []
    if channels and len(names) != channels:
        bad.append(f"{len(names)} channels, expected {channels}")
    if days and d.sizes["time"] != days:
        bad.append(f"{d.sizes['time']} days, expected {days}")
    for r in require:
        if r not in names:
            bad.append(f"missing channel {r}")
    # A store can carry the right shape and still be empty where it matters.
    tgt = d["target"][-1, 0].values
    if not np.isfinite(tgt).any():
        bad.append("last day's target is entirely non-finite")
    if bad:
        print(f"[check] {path}: INVALID — " + "; ".join(bad))
        return 1
    print(f"[check] {path}: ok ({len(names)} channels, {d.sizes['time']} days)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--channels", type=int, default=0)
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--require", nargs="*", default=[])
    a = ap.parse_args()
    sys.exit(main(a.path, a.channels, a.days, a.require))
