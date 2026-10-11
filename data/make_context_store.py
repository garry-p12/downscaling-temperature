"""Add temporal-context channels: the coarse field on neighbouring days.

WHY
    The model sees one day at a time. Temperature anomalies persist and advect,
    so yesterday's and tomorrow's coarse fields carry information about today
    that a single-day input cannot express — which way a front is moving, how
    fast, whether the airmass is building or decaying. The 5.4 budget says only
    time-varying predictors can reach the 46.4% space-time remainder, and this
    is the cheapest such predictor available: no new download, no architecture
    change, just more channels.

WHY LAGS OF THE COARSE FIELD AND NOT OF THE TARGET
    The target is what we are predicting; feeding its neighbouring days would
    leak the answer at test time, since a model given truth on day t-1 and t+1
    is doing interpolation, not downscaling. The coarse POWER field is an INPUT
    and is available on every day everywhere, so using its neighbours assumes
    nothing the deployed model would not have.

WHY BOTH DIRECTIONS
    This is a reanalysis-style task, not a forecast: the whole period is
    available at once. Using t+1 is legitimate here and would not be in an
    operational forecast setting; the channel list makes that assumption
    visible rather than burying it.

EDGES
    The first and last `lag` days have no neighbour on one side. They are
    filled with the nearest available day (edge-clamped), which is a no-op for
    the vast majority of the record and never invents data.

Usage:
    python -m data.make_context_store --src data_store_sc/aorc/dataset.zarr \\
        --out data_store_sc/aorc_ctx/dataset.zarr --lags 1 2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if os.environ.get("DOWNSCALE_MAX_WORKERS"):
    import dask
    import zarr
    zarr.config.set({"threading.max_workers": int(os.environ["DOWNSCALE_MAX_WORKERS"])})
    dask.config.set(scheduler="synchronous")

from common.normalize import Normalizer            # noqa: E402

BASE = "coarse_tmp"


def main(src, out, lags, tchunk):
    src_p, out_p = Path(src), Path(out)
    ds = xr.open_zarr(src, consolidated=True)
    names = ds["channel_in"].values.tolist()
    if BASE not in names:
        raise SystemExit(f"source store has no {BASE!r} channel")
    base_i = names.index(BASE)
    nt, ny, nx = ds.sizes["time"], ds.sizes["y"], ds.sizes["x"]
    split = ds["split"].values
    is_train = split == "train"
    land = ds["input"].isel(time=0, channel_in=names.index("land_mask")).values > 0.5

    offsets = sorted({o for L in lags for o in (-L, L)})
    new = [f"{BASE}_t{o:+d}".replace("+", "p").replace("-", "m") for o in offsets]
    all_names = names + new
    print(f"[ctx] {len(names)} -> {len(all_names)} channels; offsets {offsets}")

    # The lagged channel is the base channel read at a shifted index, clamped
    # at the ends. Materialise just that one channel for the whole record —
    # 1826 x 93 x 100 float32 is 68 MB, against 2.3 GB for the full stack.
    base = ds["input"].isel(channel_in=base_i).values.astype("float32")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    first = True
    for k0 in range(0, nt, tchunk):
        k1 = min(nt, k0 + tchunk)
        blk = ds["input"].isel(time=slice(k0, k1)).values.astype("float32")
        idx = np.arange(k0, k1)
        extra = [base[np.clip(idx + o, 0, nt - 1)] for o in offsets]
        block = np.concatenate([blk, np.stack(extra, 1)], 1)

        chunk = xr.Dataset(
            {"input": (("time", "channel_in", "y", "x"), block),
             "target": (("time", "channel_out", "y", "x"),
                        ds["target"].isel(time=slice(k0, k1)).values.astype("float32"))},
            coords={"time": ds["time"].values[k0:k1], "channel_in": all_names,
                    "channel_out": ds["channel_out"].values,
                    "y": ds["y"].values, "x": ds["x"].values},
        )
        if first:
            # One day per chunk: a time:64 store makes every loader worker pull
            # 64 days to serve one sample and deadlocked training outright.
            chunk.to_zarr(out, mode="w", consolidated=False, encoding={
                "input": {"chunks": (1, len(all_names), ny, nx)},
                "target": {"chunks": (1, ds.sizes["channel_out"], ny, nx)}})
            first = False
        else:
            chunk.to_zarr(out, mode="a", append_dim="time", consolidated=False)
        print(f"[ctx] days {k0:5d}-{k1:5d}", flush=True)

    side = xr.Dataset(
        {"clim_tmp": (("doy", "y", "x"), ds["clim_tmp"].values),
         "clim_coarse_tmp": (("doy", "y", "x"), ds["clim_coarse_tmp"].values),
         "split": (("time",), split)},
        coords={"doy": ds["doy"].values, "time": ds["time"].values,
                "y": ds["y"].values, "x": ds["x"].values},
    )
    side.to_zarr(out, mode="a", consolidated=True)

    # The lagged channels are the base channel shifted, so they share its
    # statistics exactly; copying rather than refitting keeps that true and
    # keeps an arm that selects only the original channels bit-identical.
    nz = Normalizer.load(str(src_p.parent / "norm_stats.json"))
    base_stats = nz.stats[f"in::{BASE}"]
    for n in new:
        nz.stats[f"in::{n}"] = dict(base_stats)
    nz.save(out_p.parent / "norm_stats.json")

    res = xr.open_zarr(out, consolidated=True)
    chk = res["input"].isel(channel_in=all_names.index(new[0])).values
    o0 = offsets[0]
    ref = base[np.clip(np.arange(nt) + o0, 0, nt - 1)]
    ok = np.allclose(chk[:, land], ref[:, land], equal_nan=True)
    print(f"\n[ctx] wrote {out}")
    print(f"[ctx] lag channels: {', '.join(new)}")
    print(f"[ctx] shift check for {new[0]} (offset {o0:+d}): {'OK' if ok else 'FAILED'}")
    print(f"[ctx] train days {int(is_train.sum())}, land cells {int(land.sum())}")
    if not ok:
        raise SystemExit("lagged channel does not match the shifted base channel")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data_store_sc/aorc/dataset.zarr")
    ap.add_argument("--out", default="data_store_sc/aorc_ctx/dataset.zarr")
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 2],
                    help="day offsets to add, each used in both directions")
    ap.add_argument("--tchunk", type=int, default=256)
    a = ap.parse_args()
    main(a.src, a.out, a.lags, a.tchunk)
