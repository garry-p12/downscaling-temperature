"""Is 5 km worth training at? Two measurements and three sanity checks.

NO MODEL IS TRAINED HERE. The point of this script is to cost the question
before spending GPU-days on it, and in particular to establish the number any
5 km model would have to beat.

    CHECK 1  regrid        5 km AORC truth, block-averaged back to 10 km, must
                           match the existing 10 km AORC truth. If it does not,
                           the conservative regrid is wrong and everything
                           below is wrong with it.
    CHECK 2  statics       elev_std is sub-grid relief, so a 5 km cell must
                           hold LESS of it than a 10 km cell. If it does not,
                           the native-resolution path in covariates.py was
                           skipped and the statics were resampled from an
                           already-aggregated field.
    CHECK 3  masks         land fraction and holdout cell count, reported so a
                           silently different domain cannot pass unnoticed.

    MEASURE A  budget      The task residual (AORC truth - interpolated POWER)
                           split into uniform daily offset / static pattern /
                           remainder, exactly as at 10 km. The remainder is the
                           only term a model can attack; if it grows at 5 km
                           there is more to win, and if the offset share stays
                           near half the offset correction transfers unchanged.

    MEASURE B  the control THE KILL CRITERION. Take the 10 km AORC-trained
                           model, interpolate its output onto the 5 km grid,
                           and score it there. A 5 km model that cannot beat
                           this by more than the between-seed noise floor is
                           buying pixels, not resolution.

Usage:
    python scripts/probe_5km.py
    python scripts/probe_5km.py --arms aorc_s1337 aorc_s7 aorc_s42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("DOWNSCALE_MAX_WORKERS", "4")

KM5 = "data_store_sc/km5/dataset.zarr"
KM10 = "data_store_sc/aorc/dataset.zarr"
NOISE = 0.0097          # between-seed RMSE floor measured at 10 km
OUT = REPO / "outputs" / "km5"


def rmse(x) -> float:
    return float(np.sqrt(np.nanmean(np.asarray(x) ** 2)))


def to_grid(field: np.ndarray, src, dst, land=None, order: int = 3) -> np.ndarray:
    """Interpolate a (T, H, W) stack from one projected grid onto another.

    Not a 2x `zoom`. Both grids are snapped OUTWARD to their own resolution in
    common/grid.py, so the 5 km grid is not a clean subdivision of the 10 km
    one — its extent can differ by up to a cell on each side. Mapping through
    the actual projected coordinates is the only alignment that is right at the
    edges, and the edges are where the holdout box sits.

    NaN is filled with each day's land mean before interpolating and the fill
    is not removed afterwards: a cubic kernel spreads a single NaN across its
    whole support, so leaving them in returns NaN almost everywhere. Scores are
    taken over land cells inside the holdout, which are far from the coast, but
    do not read the returned field over water.
    """
    from scipy.ndimage import map_coordinates

    sy = np.asarray(src.ds["y"].values, "float64")
    sx = np.asarray(src.ds["x"].values, "float64")
    dy = np.asarray(dst.ds["y"].values, "float64")
    dx = np.asarray(dst.ds["x"].values, "float64")
    # Regular grids, so the coordinate map is affine: express the destination
    # centres as fractional indices into the source and sample there.
    fy = (dy - sy[0]) / (sy[1] - sy[0])
    fx = (dx - sx[0]) / (sx[1] - sx[0])
    YY, XX = np.meshgrid(fy, fx, indexing="ij")

    if land is None:
        land = getattr(src, "land", None)
    out = np.empty((field.shape[0], dy.size, dx.size), "float32")
    for k in range(field.shape[0]):
        g = field[k]
        if not np.isfinite(g).all():
            ref = np.where(land, g, np.nan) if land is not None else g
            g = np.where(np.isfinite(g), g, np.nanmean(ref))
        out[k] = map_coordinates(g.astype("float64"), [YY, XX], order=order,
                                 mode="nearest")
    return out


def block_mean_to_10km(field: np.ndarray, src, dst) -> np.ndarray:
    """Average a 5 km stack onto the 10 km grid by assigning cells to cells.

    Deliberately not a reshape-and-mean: the two grids are snapped
    independently, so a 2x2 block structure is not guaranteed to line up.
    """
    sy = np.asarray(src.ds["y"].values, "float64")
    sx = np.asarray(src.ds["x"].values, "float64")
    dy = np.asarray(dst.ds["y"].values, "float64")
    dx = np.asarray(dst.ds["x"].values, "float64")
    iy = np.abs(sy[:, None] - dy[None, :]).argmin(1)     # src row -> dst row
    ix = np.abs(sx[:, None] - dx[None, :]).argmin(1)
    T = field.shape[0]
    acc = np.zeros((T, dy.size, dx.size), "float64")
    cnt = np.zeros((dy.size, dx.size), "float64")
    for a in range(sy.size):
        np.add.at(acc[:, iy[a], :], (slice(None), ix), field[:, a, :])
        np.add.at(cnt[iy[a], :], ix, 1.0)
    with np.errstate(invalid="ignore"):
        return acc / np.where(cnt > 0, cnt, np.nan)


def main(arms, out_path, device):
    from evaluation.fields import Store, decompose, predict_all

    if not (REPO / KM5).exists():
        raise SystemExit(
            f"{KM5} does not exist yet.\n"
            f"Build it first (on a compute node, not a login node):\n"
            f"  DOWNSCALE_CONFIG_data=configs/data_sc_5km.yaml "
            f"python -m data.build_dataset")

    s5 = Store(KM5, cfg_name="configs/data_sc_5km.yaml")
    s10 = Store(KM10, cfg_name="configs/data_sc_aorc.yaml")
    te5, hold5 = s5.i_test, s5.m_hold
    rep: dict = {"grid": {"km5": list(s5.shape), "km10": list(s10.shape),
                          "dx_km": s5.dx_km}}

    print(f"\n5 km grid {s5.shape[0]}x{s5.shape[1]}, "
          f"10 km grid {s10.shape[0]}x{s10.shape[1]}")

    # ---- CHECK 3: masks ----------------------------------------------------
    rep["masks"] = {
        "km5_land_frac": float(s5.land.mean()),
        "km10_land_frac": float(s10.land.mean()),
        "km5_holdout_cells": int(hold5.sum()),
        "km10_holdout_cells": int(s10.m_hold.sum()),
    }
    print(f"  land fraction  5 km {s5.land.mean():.4f} | "
          f"10 km {s10.land.mean():.4f}")
    print(f"  holdout cells  5 km {int(hold5.sum())} | "
          f"10 km {int(s10.m_hold.sum())}  "
          f"(expect ~4x: {int(s10.m_hold.sum()) * 4})")

    # ---- CHECK 1: the regrid round-trip ------------------------------------
    t5 = s5.truth(absolute=True)[te5]
    t10 = s10.truth(absolute=True)[s10.i_test]
    back = block_mean_to_10km(t5, s5, s10)
    m = s10.m_hold & np.isfinite(back).all(0)
    d = back[:, m] - t10[:, m]
    rep["regrid_roundtrip"] = {"rmse": rmse(d), "max_abs": float(np.nanmax(np.abs(d))),
                               "cells": int(m.sum())}
    print(f"\nCHECK 1  5 km truth -> 10 km vs the 10 km store: "
          f"rmse {rmse(d):.5f} degC, max |d| {np.nanmax(np.abs(d)):.5f}")

    # ---- CHECK 2: statics --------------------------------------------------
    def static_mean(st, name):
        if name not in st.stored:
            return None
        v = st.ds["input"].isel(time=0, channel_in=st.stored.index(name)).values
        return float(np.nanmean(np.where(st.land, v, np.nan)))

    rep["statics"] = {n: {"km5": static_mean(s5, n), "km10": static_mean(s10, n)}
                      for n in ("elev_std", "slope_mean", "tpi")}
    print("\nCHECK 2  statics, land mean")
    for n, v in rep["statics"].items():
        if v["km5"] is None:
            continue
        arrow = "OK" if (n != "elev_std" or v["km5"] < v["km10"]) else "** CHECK **"
        print(f"  {n:<11} 5 km {v['km5']:>10.4f} | 10 km {v['km10']:>10.4f}  {arrow}")

    # ---- MEASURE A: the error budget at 5 km -------------------------------
    # Both budgets are computed here rather than one being quoted. The famous
    # 50.8 / 2.8 / 46.4 split is against the ERA5-Land target; this store's
    # truth is AORC, and the AORC budget at 10 km is a different number
    # (54.1 / 2.5 / 43.4). Comparing 5 km AORC against the ERA5-Land figure
    # would show a "shrinking remainder" that is purely the change of target.
    sh = decompose((s5.truth() - s5.coarse())[te5], hold5)[3]
    sh10 = decompose((s10.truth() - s10.coarse())[s10.i_test], s10.m_hold)[3]
    rep["budget_km5"], rep["budget_km10_same_truth"] = sh, sh10
    print(f"\nMEASURE A  error budget, both against AORC truth")
    print(f"  {'term':<22}{'5 km':>10}{'10 km':>10}")
    for k, lab in (("offset", "uniform daily offset"),
                   ("static", "static pattern"),
                   ("remainder", "space-time remainder")):
        print(f"  {lab:<22}{sh[k] * 100:9.1f}%{sh10[k] * 100:9.1f}%")
    print(f"  {'total variance':<22}{sh['total_var']:10.4f}{sh10['total_var']:10.4f}  degC^2")
    print(f"  The remainder is the only term a model can attack; if it grows "
          f"at 5 km there is more to win.")

    # ---- MEASURE B: the bicubic control ------------------------------------
    print(f"\nMEASURE B  the number a 5 km model has to beat")
    truth5 = s5.truth(absolute=True)
    floor = rmse((s5.coarse(absolute=True) - truth5)[te5][:, hold5])
    rep["control"] = {"interpolated_power_km5": floor, "arms": {}}
    print(f"  interpolated POWER at 5 km      {floor:.4f}")

    # How much of the 5 km truth is structurally out of reach of ANY method
    # that lives on the 10 km grid: block-average the 5 km truth down to 10 km
    # and bring it straight back. What that loses is detail below the 10 km
    # Nyquist, and no amount of 10 km modelling can return it.
    #
    # This is what makes the control interpretable. The control carries two
    # terms, the 10 km model's own error and this one. If the control sits
    # barely above this floor, almost all of the gap is detail a 5 km model
    # could in principle resolve, and 5 km is worth training. If the control is
    # far above it, the gap is mostly ordinary model error and a finer grid is
    # not what fixes that.
    rt = to_grid(block_mean_to_10km(truth5[te5], s5, s10), s10, s5,
                 land=s10.land)
    detail = rmse((rt - truth5[te5])[:, hold5])
    rep["sub_10km_detail_floor"] = detail
    print(f"  sub-10 km detail, unreachable  {detail:.4f}   "
          f"(5 km truth -> 10 km -> back)")

    vals = []
    for arm in arms:
        if not (REPO / "checkpoints" / arm / "last.pt").exists():
            print(f"  {arm}: no checkpoint, skipping")
            continue
        p10 = s10.to_absolute(predict_all(s10, arm, "last", device=device))
        up = to_grid(p10[te5], s10, s5)
        r = rmse((up - truth5[te5])[:, hold5])
        rep["control"]["arms"][arm] = r
        vals.append(r)
        print(f"  {arm} upsampled to 5 km   {r:.4f}")
    if vals:
        rep["control"]["mean"] = float(np.mean(vals))
        rep["control"]["sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        bar = float(np.mean(vals)) - NOISE
        rep["control"]["beat_threshold"] = bar
        print(f"\n  CONTROL {np.mean(vals):.4f}"
              + (f" +/- {np.std(vals, ddof=1):.4f}" if len(vals) > 1 else ""))
        print(f"  A native 5 km model must score below {bar:.4f} to be real "
              f"(control minus the {NOISE} seed floor).")
        # Split the control, in VARIANCE because these are RMSEs and add in
        # quadrature. Two parts, and they call for different work:
        #
        #   detail^2            what a 10 km grid structurally cannot carry.
        #                       Only a finer grid can attack this.
        #   control^2 - detail^2  the 10 km model's own imperfection. This is
        #                       available at either resolution, so a finer grid
        #                       is not what fixes it.
        c = float(np.mean(vals))
        share = min(1.0, (detail ** 2) / (c ** 2))
        rep["control"]["detail_share_of_variance"] = share
        rep["control"]["model_error_rmse"] = float(np.sqrt(max(c ** 2 - detail ** 2, 0)))
        print(f"  Of the control's variance, {share * 100:.0f}% is sub-10 km "
              f"detail (only a finer grid can reach it)")
        print(f"  and {(1 - share) * 100:.0f}% is the 10 km model's own error "
              f"(a finer grid does not fix that).")
        print("  => " + ("resolution is the right lever." if share >= 0.5 else
                         "resolution is NOT the main lever; most of the gap is "
                         "ordinary model error."))

    OUT.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+",
                    default=["aorc_s1337", "aorc_s7", "aorc_s42"])
    ap.add_argument("--out", default=str(OUT / "probe.json"))
    # predict_all defaults to CPU. The control needs 1,826 forward passes per
    # arm and the job already holds a GH200, so there is no reason to pay for
    # them on the host.
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    main(a.arms, a.out, a.device)
