"""The target swap, drawn: ERA5-Land-trained against AORC-trained.

WHY TWO SHEETS
    `target_swap_inference` answers "is the output better", on one day and over
    the year, scored against AORC for both arms.
    `target_swap_coherence` answers "why", and is the more important one. It
    puts ERA5-Land's own fine-scale field next to AORC's, and each model's next
    to its target's. The model reproduces whatever its target contains: an
    ERA5-Land-trained model is smooth because ERA5-Land is smooth, not because
    the architecture cannot do better.

WHY BOTH ARMS ARE SCORED AGAINST AORC
    They were trained toward different products, so their own-truth RMSEs are
    distances to different references and are not comparable. AORC is the
    common judge for the grid. It does favour the AORC-trained arm — the
    station results are the non-circular check, and they agree.

WHY THE ERA5-LAND PANEL IS ON THE COHERENCE SHEET
    Without it a reader concludes the ERA5-trained model is simply worse. With
    it they can see it is faithfully reproducing a target that has almost no
    sub-60 km structure of its own.

Usage:
    python scripts/make_target_panels.py --era dynera_s1337 --aorc dynaorc_s1337
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.fields import (Store, cellwise_corr, highpass,   # noqa: E402
                               predict_all)
from evaluation.mapping import (GRAT, INK, INK2, INK3, ORANGE,   # noqa: E402
                                SURF, graticule, header, holdout_box, palettes)
from evaluation.spectral import radial_spectrum                  # noqa: E402

FIGDIR = REPO / "image_outputs" / "slides"
CUT_KM = 60.0


def rim_cells(sigma: float) -> int:
    """Cells to drop at the domain edge before quoting any high-pass statistic.

    `highpass` filters with mode="nearest", which reflects the edge value
    outward and leaves a contaminated band a few sigma wide. The band is a
    number of CELLS proportional to sigma, so a literal 4 — correct at 10 km,
    where sigma is 1.12 cells — stops covering it the moment the grid gets
    finer and a frame of fake fine-scale amplitude leaks into every mean.
    """
    import math
    return max(2, int(math.ceil(3.0 * sigma)))


def load_pred(store, name, ckpt):
    """Cached prediction if the evaluation already produced one, else compute."""
    hit = glob.glob(str(REPO / ".field_cache" / f"{name}_{ckpt}_*.npy"))
    return np.load(hit[0]) if hit else predict_all(store, name, ckpt)


def main(era_arm, aorc_arm, ckpt, day):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    seq, div = palettes()
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                         "savefig.facecolor": SURF, "text.color": INK,
                         "font.size": 10})

    st_e = Store("data_store_sc/ix_era/dataset.zarr")
    st_a = Store("data_store_sc/ix_aorc/dataset.zarr")
    SIGMA = st_a.sigma_cells(CUT_KM)
    RIM = rim_cells(SIGMA)
    print(f"[panels] {st_a.dx_km:.1f} km cells -> sigma {SIGMA:.2f} cells, "
          f"rim {RIM}")
    te, land, box, hold = st_e.i_test, st_e.land, st_e.box, st_e.m_hold
    ny, nx = st_e.shape
    inner = np.zeros_like(land); inner[RIM:-RIM, RIM:-RIM] = True
    fine = land & inner
    nan = lambda f: np.where(land, f, np.nan)                    # noqa: E731

    truth_a = st_a.to_absolute(st_a.truth())
    truth_e = st_e.to_absolute(st_e.truth())
    coarse = st_e.coarse(absolute=True)
    p_era = st_e.to_absolute(load_pred(st_e, era_arm, ckpt))
    p_aor = st_a.to_absolute(load_pred(st_a, aorc_arm, ckpt))

    # Representative day: median fine-scale activity in the box, per the other
    # figures in this repo — typical rather than flattering.
    hp_ta = highpass(truth_a[te], SIGMA)
    kk = int(np.argsort(hp_ta[:, hold].std(1))[len(te) // 2])
    kday = int(te[kk])
    date = str(st_e.times[kday])[:10]
    if day:
        import pandas as pd
        kk = int(np.argmin(np.abs(st_e.times[te] - pd.Timestamp(day))))
        kday = int(te[kk]); date = str(st_e.times[kday])[:10]

    def show(fig, a, f, t, cmap, sub="", label="degC", **kw):
        im = a.imshow(np.ma.masked_invalid(f), cmap=cmap, **kw)
        a.set_box_aspect(ny / nx)
        graticule(a, st_e.LAT, st_e.LON, fontsize=8.5)
        holdout_box(a, box, "")
        a.set_title(t, fontsize=13, fontweight="semibold", pad=16)
        if sub:
            a.text(.5, 1.004, sub, transform=a.transAxes, ha="center",
                   va="bottom", fontsize=8.6, color=INK3)
        cb = fig.colorbar(im, ax=a, fraction=.038, pad=.015)
        cb.ax.tick_params(labelsize=8.5, color=GRAT, labelcolor=INK3)
        cb.set_label(label, fontsize=9, color=INK3)

    R = lambda p: float(np.sqrt(((p - truth_a)[te][:, hold] ** 2).mean()))  # noqa: E731
    r_era, r_aor = R(p_era), R(p_aor)

    # ---------------- sheet 1: does the output look better ------------------
    fig, ax = plt.subplots(2, 4, figsize=(24, 11.8))
    vt = np.nanpercentile(truth_a[kday][land], [1, 99])
    for a, f, t, sub in [
            (ax[0, 0], coarse[kday], "Coarse input (POWER)", "what both models are given"),
            (ax[0, 1], truth_a[kday], "Truth (AORC)", "the judge for both arms"),
            (ax[0, 2], p_era[kday], "ERA5-Land-trained output", f"{era_arm}"),
            (ax[0, 3], p_aor[kday], "AORC-trained output", f"{aorc_arm}")]:
        show(fig, a, nan(f), t, seq, sub, vmin=vt[0], vmax=vt[1])

    e_era, e_aor = p_era[kday] - truth_a[kday], p_aor[kday] - truth_a[kday]
    elim = float(np.nanpercentile(np.abs(np.r_[e_era[hold], e_aor[hold]]), 98))
    show(fig, ax[1, 0], nan(e_era), "ERA5-Land-trained error", div,
         "vs AORC, this day", norm=TwoSlopeNorm(0, -elim, elim))
    show(fig, ax[1, 1], nan(e_aor), "AORC-trained error", div,
         "same colour scale", norm=TwoSlopeNorm(0, -elim, elim))
    pc_era = np.sqrt(((p_era - truth_a)[te] ** 2).mean(0))
    pc_aor = np.sqrt(((p_aor - truth_a)[te] ** 2).mean(0))
    vmax = float(np.nanpercentile(pc_era[land], 99))
    show(fig, ax[1, 2], nan(pc_era), "Per-cell RMSE, ERA5-Land-trained", seq,
         f"whole test year; box {r_era:.3f} degC", vmin=0, vmax=vmax)
    show(fig, ax[1, 3], nan(pc_aor), "Per-cell RMSE, AORC-trained", seq,
         f"same scale; box {r_aor:.3f} degC  ({(r_aor-r_era)/r_era:+.1%})",
         vmin=0, vmax=vmax)

    top = header(fig, f"Swapping the training target   ({date}, {ckpt}.pt)",
                 f"Both arms see identical inputs and are scored against AORC. "
                 f"Box RMSE {r_era:.3f} -> {r_aor:.3f} degC "
                 f"({(r_aor-r_era)/r_era:+.1%}).")
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=2.6)
    out1 = FIGDIR / "target_swap_inference.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---------------- sheet 2: where the fine-scale skill comes from --------
    # The absolute fields carry NaN off-land (the stores' climatologies are
    # masked), and a Gaussian filter spreads a single NaN across its whole
    # neighbourhood — high-passing them directly returns NaN everywhere. Fill
    # the sea with each day's land mean first: that removes the NaN without
    # creating a hard coastal step, which a fill of zero would.
    def filled(f):
        g = f[te].copy()
        mean_day = np.nanmean(g[:, land], axis=1)[:, None, None]
        return np.where(np.isfinite(g), g, mean_day)

    hp = {"AORC": highpass(filled(truth_a), SIGMA),
          "ERA5": highpass(filled(truth_e), SIGMA),
          "era": highpass(filled(p_era), SIGMA),
          "aor": highpass(filled(p_aor), SIGMA)}
    sd = {k: v.std(0) for k, v in hp.items()}
    coh_era = cellwise_corr(hp["era"], hp["AORC"], fine)
    coh_aor = cellwise_corr(hp["aor"], hp["AORC"], fine)

    fig, ax = plt.subplots(2, 4, figsize=(24, 11.8))
    vsd = float(np.percentile(sd["AORC"][fine], 95))
    for a, k, t, sub in [
            (ax[0, 0], "AORC", "AORC truth", "the reference"),
            (ax[0, 1], "ERA5", "ERA5-Land truth", "the other product, same scale"),
            (ax[0, 2], "era", "ERA5-Land-trained model", "matches its own target"),
            (ax[0, 3], "aor", "AORC-trained model", "matches its own target")]:
        show(fig, a, np.where(fine, sd[k], np.nan),
             f"{t} — detail below {CUT_KM:.0f} km",
             seq, f"{sub}; box mean {sd[k][hold].mean():.3f} degC", vmin=0, vmax=vsd)

    cmax = float(np.nanpercentile(np.r_[coh_era[fine], coh_aor[fine]], 99))
    show(fig, ax[1, 0], coh_era, "Coherence with AORC, ERA5-Land-trained", seq,
         f"box mean {np.nanmean(coh_era[hold]):.3f}", label="r", vmin=0, vmax=cmax)
    show(fig, ax[1, 1], coh_aor, "Coherence with AORC, AORC-trained", seq,
         f"same scale; box mean {np.nanmean(coh_aor[hold]):.3f}",
         label="r", vmin=0, vmax=cmax)
    d = np.where(fine, coh_aor - coh_era, np.nan)
    dlim = float(np.nanpercentile(np.abs(d[fine]), 99))
    show(fig, ax[1, 2], d, "Coherence gained by the swap", div,
         "red = AORC target better", label="delta r",
         norm=TwoSlopeNorm(0, -dlim, dlim))

    # the curves, so the maps are anchored to a scale axis
    i0, i1, j0, j1 = box
    crop = lambda f: f[te][:, i0:i1, j0:j1] - f[te][:, i0:i1, j0:j1].mean((1, 2))[:, None, None]  # noqa: E731
    A = crop(truth_a)
    a4 = ax[1, 3]
    for lab, f, c in [("ERA5-Land truth", truth_e, INK3),
                      ("ERA5-Land-trained", p_era, "#2a78d6"),
                      ("AORC-trained", p_aor, ORANGE)]:
        rows = radial_spectrum(crop(f), A, dx_km=st_a.dx_km)
        w = [r["wavelength_km"] for r in rows]
        a4.plot(w, [r["coherence"] for r in rows], "-o", ms=3.5, lw=1.6,
                color=c, label=lab)
    a4.set_xscale("log"); a4.invert_xaxis()
    # Explicit ticks: matplotlib's automatic log minor labels collide badly on
    # an inverted axis and render as an unreadable run of 4x10^2 / 2x10^2.
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
    ticks = [400, 200, 100, 60, 40, 25, 15]
    a4.xaxis.set_major_locator(FixedLocator(ticks))
    a4.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    a4.xaxis.set_minor_locator(NullLocator())
    a4.axhline(.5, ls="--", lw=1, color=GRAT)
    a4.text(ticks[0], .52, "coherence 0.5", fontsize=8, color=INK3, va="bottom")
    a4.set_xlabel("wavelength (km) — finer to the right", fontsize=9, color=INK2)
    a4.set_ylabel("coherence with AORC", fontsize=9, color=INK2)
    a4.set_title("Coherence against scale", fontsize=13, fontweight="semibold", pad=16)
    a4.legend(fontsize=8.5, framealpha=.9)
    a4.tick_params(labelsize=8.5, color=GRAT, labelcolor=INK3)
    for sp in ("top", "right"):
        a4.spines[sp].set_visible(False)

    top = header(fig, f"Why the swap works: the model reproduces its target "
                      f"(below {CUT_KM:.0f} km, {ckpt}.pt)",
                 f"Each model carries about as much fine-scale detail as the product "
                 f"it was trained on. Coherence with AORC over the box: "
                 f"{np.nanmean(coh_era[hold]):.3f} -> {np.nanmean(coh_aor[hold]):.3f}.")
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=2.6)
    out2 = FIGDIR / "target_swap_coherence.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)

    print(f"  {out1.name}   vs AORC: {r_era:.4f} -> {r_aor:.4f} ({(r_aor-r_era)/r_era:+.1%})")
    print(f"  {out2.name}   fine-scale sd  AORC {sd['AORC'][hold].mean():.4f} | "
          f"ERA5-Land {sd['ERA5'][hold].mean():.4f} | "
          f"era-trained {sd['era'][hold].mean():.4f} | aorc-trained {sd['aor'][hold].mean():.4f}")
    print(f"                 coherence      {np.nanmean(coh_era[hold]):.4f} -> "
          f"{np.nanmean(coh_aor[hold]):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--era", default="dynera_s1337")
    ap.add_argument("--aorc", default="dynaorc_s1337")
    ap.add_argument("--ckpt", default="last", choices=["best", "last"])
    ap.add_argument("--day", default=None)
    a = ap.parse_args()
    os.environ.setdefault("DOWNSCALE_MAX_WORKERS", "4")
    main(a.era, a.aorc, a.ckpt, a.day)
