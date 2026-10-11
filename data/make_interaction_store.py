"""Add weather-modulated terrain channels to an existing store.

WHY THESE CHANNELS EXIST
    The 5.4 error budget caps what a STATIC map can reach at 2.8% of the task
    residual, and five static covariate sets duly did nothing. The 46.4%
    space-time remainder can only be reached by predictors that vary in both
    space and time. Adding the dynamic POWER fields as extra channels already
    gave the largest predictor effect measured in the project (RMSE 0.7841 ->
    0.7270, t = -6.6), but it asks the network to discover, from scratch, that
    terrain's influence DEPENDS on the weather.

    That dependence is the physics. A valley pools cold air on a calm, clear
    night and mixes it away when the wind gets up; an urban heat island is
    strongest under exactly the same conditions; a slope's aspect only matters
    while the sun is on it; the lapse rate is steeper in dry air than in moist.
    A static `elev_std` channel gets one fixed coefficient for all of that. The
    products below hand the interaction to the model directly.

WHY THE FACTORS ARE Z-SCORED BEFORE MULTIPLYING
    For a = a_bar + a' and b = b_bar + b', the raw product ab carries both main
    effects (a_bar*b' and b_bar*a') as well as the interaction a'b'. The main
    effects already have their own channels, so a raw product would mostly
    duplicate them and the new channel would be hard to attribute. Centring
    each factor on TRAIN-SPLIT statistics first leaves a'b' alone — a genuinely
    new degree of freedom.

    Statistics come from the train split over land only. Using all days would
    leak test-period information into the channel definition.

WHY THE LAND MASK IS TAKEN FROM ELSEWHERE
    AORC is a CONUS analysis and ERA5-Land is not, so the two products cover
    different cells. --mask-from forces every arm onto one intersected mask, so
    a difference between arms cannot be "different cells" (see
    data/make_aorc_store.py, which established the intersection).

Usage:
    # ERA5-Land target, intersected mask
    python -m data.make_interaction_store \\
        --src   data_store_sc/dyn/dataset.zarr \\
        --mask-from data_store_sc/aorc/dataset.zarr \\
        --out   data_store_sc/ix_era/dataset.zarr

    # AORC target, same inputs
    python -m data.make_interaction_store \\
        --src   data_store_sc/dyn/dataset.zarr \\
        --truth-from data_store_sc/aorc/dataset.zarr \\
        --out   data_store_sc/ix_aorc/dataset.zarr
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# TACC login nodes cap the number of threads a user may start, and zarr v3
# opens an unbounded I/O pool by default — enough to trip "RuntimeError: can't
# start new thread" before a single chunk is written. This has now bitten the
# download, the rechunk and the evaluation paths in this project, so the limit
# is set here rather than remembered at each call site. It is a ceiling, not a
# target: unset, behaviour is unchanged.
if os.environ.get("DOWNSCALE_MAX_WORKERS"):
    import dask
    import zarr
    zarr.config.set({"threading.max_workers": int(os.environ["DOWNSCALE_MAX_WORKERS"])})
    # zarr is only half of it: xarray hands the read to dask, whose default
    # threaded scheduler opens its own pool and trips the same cap one layer
    # down. This build is I/O bound on a single node, so the synchronous
    # scheduler costs nothing that matters.
    dask.config.set(scheduler="synchronous")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common.normalize import Normalizer            # noqa: E402

# (new channel, factor a, factor b, why)
INTERACTIONS: list[tuple[str, str, str, str]] = [
    ("ix_tpi_wind", "tpi", "coarse_wind",
     "cold-air pooling: basins decouple on calm nights, mix out when windy"),
    ("ix_elevstd_wind", "elev_std", "coarse_wind",
     "sub-grid relief only expresses itself while the boundary layer is decoupled"),
    ("ix_dem_dewdep", "dem", "coarse_dewdep",
     "lapse rate is near dry-adiabatic in dry air, much weaker when moist"),
    ("ix_dem_swdown", "dem", "coarse_swdown",
     "elevation-dependent solar heating; thinner air heats faster in strong sun"),
    ("ix_urban_wind", "urban_frac", "coarse_wind",
     "urban heat island is strongest on calm nights and ventilates when windy"),
    ("ix_coast_wind", "coastal_dist", "coarse_wind",
     "sea-breeze penetration inland scales with the synoptic wind"),
    ("ix_forest_swdown", "lc_forest", "coarse_swdown",
     "canopy shading and transpiration only matter under strong insolation"),
    ("ix_north_swdown", "northness", "coarse_swdown",
     "slope aspect changes the surface energy balance only while the sun is out"),
]

# Derived main effect, so its interaction is not the only place it appears.
DERIVED = {
    "coarse_dewdep": ("coarse_tmp", "coarse_dewpt",
                      "dew-point depression: how far the air is from saturation"),
}


def main(src, out, mask_from, truth_from, tchunk):
    src_p, out_p = Path(src), Path(out)
    ds = xr.open_zarr(src, consolidated=True)
    names = ds["channel_in"].values.tolist()
    split = ds["split"].values
    nt = ds.sizes["time"]
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    is_train = split == "train"

    mask_src = truth_from or mask_from
    if mask_src is None:
        raise SystemExit("pass --mask-from or --truth-from: every arm must share one mask")
    other = xr.open_zarr(mask_src, consolidated=True)
    o_names = other["channel_in"].values.tolist()
    land = other["input"].values[0, o_names.index("land_mask")] > 0.5
    print(f"[ix] mask from {mask_src}: {int(land.sum())} land cells "
          f"(source store had {int((ds['input'].values[0, names.index('land_mask')] > 0.5).sum())})")

    need = sorted({f for _, a, b, _ in INTERACTIONS for f in (a, b)}
                  | {f for pair in DERIVED.values() for f in pair[:2]})
    missing = [n for n in need if n not in names and n not in DERIVED]
    if missing:
        raise SystemExit(f"source store lacks channels the interactions need: {missing}")

    # --- factor statistics, TRAIN split and land only -----------------------
    def raw(name):
        if name in DERIVED:
            a, b, _ = DERIVED[name]
            return (ds["input"][:, names.index(a)].values
                    - ds["input"][:, names.index(b)].values)
        return ds["input"][:, names.index(name)].values

    stats = {}
    for n in need:
        v = raw(n)[is_train][:, land]
        stats[n] = (float(np.nanmean(v)), float(np.nanstd(v) + 1e-8))
        print(f"[ix] {n:<18} mean {stats[n][0]:+9.3f}  sd {stats[n][1]:8.3f}")

    new_names = list(DERIVED) + [n for n, *_ in INTERACTIONS]
    all_names = names + new_names
    print(f"[ix] {len(names)} -> {len(all_names)} channels")

    # --- write, one time chunk at a time ------------------------------------
    # The full stack is 1826 x 34 x 93 x 100 float32 = 2.3 GB. Building it in
    # memory works on a workstation and does not on a laptop, so this streams.
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tgt_src = other if truth_from else ds
    template = None

    for k0 in range(0, nt, tchunk):
        k1 = min(nt, k0 + tchunk)
        base = ds["input"][k0:k1].values.astype("float32")        # (t, C, y, x)
        z = {}
        for n in need:
            m, s = stats[n]
            if n in DERIVED:
                a, b, _ = DERIVED[n]
                v = base[:, names.index(a)] - base[:, names.index(b)]
            else:
                v = base[:, names.index(n)]
            z[n] = (v - m) / s

        extra = [z[n] for n in DERIVED]
        extra += [z[a] * z[b] for _, a, b, _ in INTERACTIONS]
        block = np.concatenate([base, np.stack(extra, 1).astype("float32")], 1)

        # Land mask channel follows the intersection, and non-land is zeroed so
        # a cell absent from one product cannot contribute through the other.
        block[:, all_names.index("land_mask")] = land.astype("float32")
        block[:, :, ~land] = 0.0

        tgt = tgt_src["target"][k0:k1].values.astype("float32")
        tgt[:, :, ~land] = np.nan

        chunk = xr.Dataset(
            {"input": (("time", "channel_in", "y", "x"), block),
             "target": (("time", "channel_out", "y", "x"), tgt)},
            coords={"time": ds["time"].values[k0:k1],
                    "channel_in": all_names,
                    "channel_out": ds["channel_out"].values,
                    "y": ds["y"].values, "x": ds["x"].values},
        )
        if template is None:
            # One day per chunk. A time:64 store makes every worker pull 64
            # days to serve one sample; on the AORC build that took a training
            # run from ~10 min/seed to 24 and deadlocked the loader outright.
            chunk.to_zarr(out, mode="w", consolidated=False, encoding={
                "input":  {"chunks": (1, len(all_names), ny, nx)},
                "target": {"chunks": (1, ds.sizes["channel_out"], ny, nx)},
            })
            template = True
        else:
            chunk.to_zarr(out, mode="a", append_dim="time", consolidated=False)
        print(f"[ix] days {k0:5d}-{k1:5d}", flush=True)

    # --- the per-day and per-doy extras, copied or swapped ------------------
    # clim_tmp is (doy, y, x) — no channel axis, unlike target.
    clim = tgt_src["clim_tmp"].values.copy()
    clim[:, ~land] = np.nan
    side = xr.Dataset(
        {"clim_tmp": (("doy", "y", "x"), clim),
         "clim_coarse_tmp": (("doy", "y", "x"), ds["clim_coarse_tmp"].values),
         "split": (("time",), split)},
        coords={"doy": tgt_src["doy"].values, "time": ds["time"].values,
                "y": ds["y"].values, "x": ds["x"].values},
    )
    side.to_zarr(out, mode="a", consolidated=True)

    # --- normalizer ---------------------------------------------------------
    # Input stats for the ORIGINAL channels are reused unchanged, so an arm that
    # selects only those channels is bit-identical to the source store. The new
    # channels and, when the target was swapped, out::tmp are fitted here.
    nz = Normalizer.load(str(src_p.parent / "norm_stats.json"))
    res = xr.open_zarr(out, consolidated=True)
    for n in new_names:
        v = res["input"][:, all_names.index(n)].values[is_train][:, land]
        nz.fit(f"in::{n}", v)
    if truth_from:
        t = res["target"].values[:, 0][is_train][:, land]
        nz.fit("out::tmp", t)
        print(f"[ix] refit out::tmp for the swapped target: "
              f"mean {nz.stats['out::tmp']['mean']:+.4f} "
              f"sd {nz.stats['out::tmp']['std']:.4f}")
    nz.save(out_p.parent / "norm_stats.json")

    print(f"\n[ix] wrote {out}")
    print(f"[ix] target from {'AORC (swapped)' if truth_from else 'source store'}")
    for n, a, b, why in INTERACTIONS:
        print(f"     {n:<18} = z({a}) * z({b})   # {why}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data_store_sc/dyn/dataset.zarr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-from", default=None,
                    help="store whose land_mask every arm should share")
    ap.add_argument("--truth-from", default=None,
                    help="store to take target/clim/mask from (implies --mask-from)")
    ap.add_argument("--tchunk", type=int, default=256)
    a = ap.parse_args()
    main(a.src, a.out, a.mask_from, a.truth_from, a.tchunk)
