"""Maps for the two independent truths: AORC and the ISD stations.

    aorc_comparison.png    the two truth products side by side, where they
                           DISAGREE, and the model's error against AORC before
                           and after the offset correction
    station_timeseries.png what the thermometers actually recorded, against
                           every method, for both sites inside the holdout

The AORC-minus-ERA5-Land panel is the point of the first figure. Every metric
in this project is measured against ERA5-Land; that panel shows how much room
there is between the two products, which is the room a model can score in
without being any more correct.

Usage:
    DOWNSCALE_CONFIG_data=configs/data_sc_super.yaml \\
        python scripts/make_truth_figures.py --arch v2_landcov_s1337
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common import load_config                                  # noqa: E402
from common.grid import build_target_grid                       # noqa: E402
from common.normalize import Normalizer                         # noqa: E402
from evaluation.aorc_eval import load_aorc_on_target            # noqa: E402
from evaluation.mapping import (GRAT, INK, INK2, INK3, NODATA,  # noqa: E402
                                ORANGE, SURF, graticule,
                                holdout_box, palettes)
from evaluation.offset_correction import (predict_all,          # noqa: E402
                                          ridge_fit_predict, subregion_means)
from evaluation.station_check import (STATIONS, daily_mean,     # noqa: E402
                                      fetch_isd, model_series, nearest_cell)
from training.dataset import holdout_bounds                     # noqa: E402

FIG = REPO / "image_outputs" / "slides"


def main(arch, zarr, year, day):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    seq, div = palettes()
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                         "savefig.facecolor": SURF, "text.color": INK,
                         "font.size": 10})
    FIG.mkdir(parents=True, exist_ok=True)

    cfg = load_config("data")
    ds = xr.open_zarr(zarr, consolidated=True)
    nz = Normalizer.load(str(Path(zarr).parent / "norm_stats.json"))
    tg = build_target_grid(cfg["domain"], cfg["grid"])
    LAT, LON = tg.latlon()
    stored = ds["channel_in"].values.tolist()
    i0, i1, j0, j1 = box = holdout_bounds(ds, cfg["holdout"])
    land = ds["input"].values[0, stored.index("land_mask")] > 0.5
    hold = np.zeros_like(land); hold[i0:i1, j0:j1] = True
    m_hold, m_train = hold & land, (~hold) & land
    times = pd.DatetimeIndex(ds["time"].values)
    doy = times.dayofyear.values - 1
    clim = ds["clim_tmp"].values
    clim_c = ds["clim_coarse_tmp"].values
    split = ds["split"].values

    # best.pt, NOT last.pt: station_check's model_series resolves through
    # ckpt_path(), which returns best.pt. Using last.pt here would build the
    # offset estimate from different weights than the series it corrects — the
    # figure would then disagree with the published station table (it read
    # 0.975 vs 0.953 before this was fixed) for no visible reason.
    ckpt = REPO / "checkpoints" / arch / "best.pt"
    if not ckpt.exists():
        ckpt = REPO / "checkpoints" / arch / "last.pt"
        print(f"[warn] no best.pt for {arch}; using last.pt — the station panel "
              f"uses best.pt where available, so check they agree")
    print(f"predicting {arch} ({ckpt.name}) over all {len(times)} days ...")
    pa = nz.inverse("out::tmp", predict_all(ckpt, ds, nz, stored, "cpu"))
    resid = np.nan_to_num(ds["target"].values[:, 0]) - pa
    est = ridge_fit_predict(subregion_means(resid, m_train, *land.shape, k=6),
                            resid[:, m_hold].mean(1),
                            np.where(split == "train")[0])

    aorc = load_aorc_on_target(cfg, tg, year)
    at = pd.DatetimeIndex(aorc["time"].values).normalize()
    k = int(np.argmin(np.abs(times - pd.Timestamp(day))))
    day = times[k].strftime("%Y-%m-%d")
    ka = int(np.argmin(np.abs(at - pd.Timestamp(day))))
    A = aorc.values[ka]
    era = ds["target"].values[k, 0] + clim[doy[k]]
    pred = pa[k] + clim[doy[k]]
    corr = pred + est[k]
    cov = np.isfinite(A)

    # ------------------------------------------------------------ figure 1
    fig, ax = plt.subplots(2, 3, figsize=(15.6, 9.6))

    def show(a, f, t, cmap, sub="", **kw):
        im = a.imshow(np.ma.masked_invalid(np.where(land, f, np.nan)), cmap=cmap, **kw)
        graticule(a, LAT, LON); holdout_box(a, box, "")
        a.set_title(t, fontsize=11.5, fontweight="semibold", pad=10)
        if sub:
            a.text(.5, 1.004, sub, transform=a.transAxes, ha="center",
                   va="bottom", fontsize=7.6, color=INK3)
        cb = fig.colorbar(im, ax=a, fraction=.042, pad=.02)
        cb.ax.tick_params(labelsize=7.5, color=GRAT, labelcolor=INK3)
        return cb

    vmin, vmax = np.nanpercentile(era[land], 1), np.nanpercentile(era[land], 99)
    show(ax[0, 0], A, "AORC (observation-informed, 1 km -> 10 km)", seq,
         f"grey = outside CONUS coverage ({100 * cov[land].mean():.0f}% of land)",
         vmin=vmin, vmax=vmax).set_label("degC", fontsize=8, color=INK3)
    show(ax[0, 1], era, "ERA5-Land (what we trained against)", seq,
         "a reanalysis, not observations", vmin=vmin, vmax=vmax
         ).set_label("degC", fontsize=8, color=INK3)
    d_t = np.where(cov, A - era, np.nan)
    dl = float(np.nanpercentile(np.abs(d_t[m_hold]), 99))
    show(ax[0, 2], d_t, "THE TWO TRUTHS DISAGREE  (AORC − ERA5-Land)", div,
         f"held-out region: rms {np.sqrt(np.nanmean(d_t[m_hold] ** 2)):.3f} degC",
         norm=TwoSlopeNorm(0, -dl, dl)).set_label("degC", fontsize=8, color=INK3)

    e_m = np.where(cov, pred - A, np.nan)
    e_c = np.where(cov, corr - A, np.nan)
    el = float(np.nanpercentile(np.abs(e_m[m_hold]), 99))
    rm = np.sqrt(np.nanmean(e_m[m_hold] ** 2))
    rc = np.sqrt(np.nanmean(e_c[m_hold] ** 2))
    show(ax[1, 0], e_m, "Model error vs AORC", div,
         f"held-out rms {rm:.3f} degC", norm=TwoSlopeNorm(0, -el, el)
         ).set_label("degC", fontsize=8, color=INK3)
    show(ax[1, 1], e_c, "After the offset correction", div,
         f"held-out rms {rc:.3f} degC  ({(rc - rm) / rm * 100:+.1f}%), same scale",
         norm=TwoSlopeNorm(0, -el, el)).set_label("degC", fontsize=8, color=INK3)

    import json
    R = json.load(open(REPO / "outputs/aorc_eval.json"))
    lad = ax[1, 2]
    items = [("interpolated POWER", R["interpolated_POWER"], "#c9c6bf"),
             (f"{arch.rsplit('_s', 1)[0]}", R[arch], "#9aa5b1"),
             ("+ offset correction", R[f"{arch} + offset correction"], "#2a78d6"),
             ("ERA5-Land itself", R["ERA5-Land (the training target)"], ORANGE)]
    lad.barh([i[0] for i in items], [i[1] for i in items],
             color=[i[2] for i in items], height=.55)
    for yi, (_, v, _) in enumerate(items):
        lad.text(v, yi, f"  {v:.3f}", va="center", fontsize=10.5,
                 fontweight="semibold", color=INK)
    lad.set_xlim(0, max(i[1] for i in items) * 1.3)
    lad.set_xlabel("RMSE vs AORC, whole test year, degC", fontsize=8.5, color=INK2)
    lad.set_title("All 365 days, 1295 held-out cells", fontsize=11.5,
                  fontweight="semibold", pad=10)
    lad.text(.5, 1.004, "the correction never saw AORC — it was fitted against ERA5-Land",
             transform=lad.transAxes, ha="center", va="bottom", fontsize=7.6, color=INK3)
    for sp in ("top", "right", "left"):
        lad.spines[sp].set_visible(False)
    lad.tick_params(labelsize=9.5, color=GRAT, labelcolor=INK2)

    fig.suptitle(f"A second, independent truth — AORC   ({day})", x=.008, y=.995,
                 ha="left", va="top", fontsize=16.5, fontweight="semibold")
    fig.text(.008, .955,
             "RMSE is NOT comparable across truth products (the bilinear floor is "
             "1.014 against ERA5-Land and 1.425 against AORC). What transfers is "
             "the RELATIVE effect of the correction — read each panel down, never "
             "across to the ERA5-Land tables.",
             ha="left", va="top", fontsize=9.3, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, .925))
    fig.savefig(FIG / "aorc_comparison.png", dpi=155, bbox_inches="tight")
    plt.close(fig)
    print("  aorc_comparison.png")

    # ------------------------------------------------------------ figure 2
    test = ds.sel(time=ds["split"] == "test")
    tt = pd.DatetimeIndex(test["time"].values)
    td = tt.dayofyear.values - 1
    is_te = split == "test"
    fig, ax = plt.subplots(2, 2, figsize=(17, 8.6),
                           gridspec_kw={"width_ratios": [2.6, 1]})
    for n, (nm, (usaf, wban, slat, slon)) in enumerate(STATIONS.items()):
        obs = daily_mean(fetch_isd(usaf, wban, year, REPO / "data_store_sc/raw/isd"))
        si, sj = nearest_cell(ds, slat, slon)
        o = pd.Series(obs).reindex(tt).values
        mser = nz.inverse("out::tmp", model_series(arch, test, nz, si, sj, "cpu"))
        m_abs = mser + clim[td, si, sj]
        c_abs = m_abs + est[is_te]
        e5 = test["target"].values[:, 0, si, sj] + clim[td, si, sj]

        a = ax[n, 0]
        a.plot(tt, o, color=INK, lw=1.1, label="station (ISD)")
        a.plot(tt, e5, color=ORANGE, lw=.9, alpha=.85, label="ERA5-Land")
        a.plot(tt, c_abs, color="#2a78d6", lw=.9, alpha=.85, label="model + correction")
        a.set_ylabel("daily mean 2 m T, degC", fontsize=9, color=INK2)
        a.set_title(f"{nm.split('_')[0]} — {nm.split('_', 1)[1].replace('_', ' ')}"
                    f"  ({slat:.3f} N, {abs(slon):.3f} W)",
                    fontsize=12, fontweight="semibold", pad=9)
        a.legend(fontsize=8, ncol=3, loc="upper left", framealpha=.9)
        a.tick_params(labelsize=8.5, color=GRAT, labelcolor=INK3)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

        b = ax[n, 1]
        ok = np.isfinite(o)
        rows = [("interp POWER",
                 test["input"].values[:, stored.index("coarse_tmp"), si, sj]
                 + clim_c[td, si, sj], "#c9c6bf"),
                ("model", m_abs, "#9aa5b1"),
                ("+ correction", c_abs, "#2a78d6"),
                ("ERA5-Land", e5, ORANGE)]
        vals = [float(np.sqrt(np.nanmean((v[ok] - o[ok]) ** 2))) for _, v, _ in rows]
        b.barh([r[0] for r in rows], vals, color=[r[2] for r in rows], height=.55)
        for yi, v in enumerate(vals):
            b.text(v, yi, f"  {v:.3f}", va="center", fontsize=10,
                   fontweight="semibold", color=INK)
        b.axvline(vals[-1], ls="--", lw=1.2, color=ORANGE, zorder=0)
        b.set_xlim(0, max(vals) * 1.3)
        b.set_xlabel("RMSE vs station, degC", fontsize=8.5, color=INK2)
        b.set_title(f"{int(ok.sum())} days", fontsize=11, fontweight="semibold", pad=9)
        for sp in ("top", "right", "left"):
            b.spines[sp].set_visible(False)
        b.tick_params(labelsize=9, color=GRAT, labelcolor=INK2)

    fig.suptitle("What the thermometers recorded", x=.008, y=.995, ha="left",
                 va="top", fontsize=16.5, fontweight="semibold")
    fig.text(.008, .955,
             f"Both NOAA ISD sites sit inside the held-out region, in the same "
             f"10 km cell. Dashed line = ERA5-Land's own distance from the "
             f"station: the product every model here was trained toward, and "
             f"therefore the ceiling. {arch}, {ckpt.name}.",
             ha="left", va="top", fontsize=9.3, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, .925))
    fig.savefig(FIG / "station_timeseries.png", dpi=155, bbox_inches="tight")
    plt.close(fig)
    print("  station_timeseries.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="v2_landcov_s1337")
    ap.add_argument("--zarr", default="data_store_sc/super/dataset.zarr")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--day", default="2023-07-20")
    a = ap.parse_args()
    main(a.arch, a.zarr, a.year, a.day)
