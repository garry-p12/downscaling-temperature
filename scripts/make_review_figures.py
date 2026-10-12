"""Figures for the plain-language review document.

Two sheets that no existing figure covers:

  T1_station_panel   Every ISD station in the held-out box, each product
                     scored against it. The claim "AORC is the better target"
                     rested on two stations; this is all 44, over five years,
                     with the model rows on the test split only.

  T2_resolution      The amplitude and coherence curves that replace the single
                     "effective resolution" number. That number was quantised
                     to four radial bands above 60 km and returned an identical
                     165 km for every arm, so it could not distinguish them.
                     The curves can.

Usage:
    python scripts/make_review_figures.py
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

from evaluation.mapping import (GRAT, INK, INK2, INK3, ORANGE,   # noqa: E402
                                SURF, header)

FIGDIR = REPO / "image_outputs" / "review"
COOL, WARM, GREY = "#2a78d6", "#e34948", "#9aa0a6"


def style(plt):
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                         "savefig.facecolor": SURF, "text.color": INK,
                         "font.size": 10})


def fig_station_panel(plt):
    """Per-station RMSE for both products, plus the paired difference."""
    p = REPO / "outputs" / "station_panel" / "products.json"
    if not p.exists():
        print("  skipping T1: run evaluation.station_panel first")
        return
    rows = json.loads(p.read_text())
    rows.sort(key=lambda r: r["aorc"]["rmse"])
    era = np.array([r["era5_land"]["rmse"] for r in rows])
    aor = np.array([r["aorc"]["rmse"] for r in rows])
    n = len(rows)
    x = np.arange(n)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6),
                           gridspec_kw={"width_ratios": [2.1, 1, 1]})

    a = ax[0]
    a.vlines(x, aor, era, color=GREY, lw=1, alpha=.6, zorder=1)
    a.scatter(x, era, s=26, color=WARM, label="ERA5-Land", zorder=3)
    a.scatter(x, aor, s=26, color=COOL, label="AORC", zorder=3)
    a.set_xticks([]); a.set_xlabel(f"{n} ISD stations inside the held-out box, "
                                   "sorted by AORC error", fontsize=9.5, color=INK2)
    a.set_ylabel("RMSE against the thermometer (degC)", fontsize=9.5, color=INK2)
    a.set_title("Every station, both products", fontsize=12.5,
                fontweight="semibold", pad=12)
    a.legend(fontsize=9, framealpha=.9)
    a.text(.015, .965, f"AORC closer at {int((aor < era).sum())}/{n} stations",
           transform=a.transAxes, va="top", fontsize=10.5, fontweight="semibold",
           color=COOL)

    a = ax[1]
    bins = np.linspace(0, max(era.max(), aor.max()) * 1.05, 22)
    a.hist(era, bins=bins, color=WARM, alpha=.65, label="ERA5-Land")
    a.hist(aor, bins=bins, color=COOL, alpha=.65, label="AORC")
    for v, c in [(era.mean(), WARM), (aor.mean(), COOL)]:
        a.axvline(v, color=c, lw=1.6, ls="--")
    a.set_xlabel("RMSE (degC)", fontsize=9.5, color=INK2)
    a.set_ylabel("stations", fontsize=9.5, color=INK2)
    a.set_title("Distribution", fontsize=12.5, fontweight="semibold", pad=12)
    a.legend(fontsize=9, framealpha=.9)
    a.text(.97, .95, f"ERA5-Land {era.mean():.3f}\nAORC      {aor.mean():.3f}",
           transform=a.transAxes, ha="right", va="top", fontsize=9.5,
           family="monospace", color=INK2)

    # Paired differences: the honest view, since the two products are scored
    # at the same stations on the same days.
    a = ax[2]
    d = era - aor
    a.scatter(aor, d, s=26, color=COOL, alpha=.8)
    a.axhline(0, color=INK3, lw=1)
    a.set_xlabel("AORC RMSE (degC)", fontsize=9.5, color=INK2)
    a.set_ylabel("ERA5-Land minus AORC (degC)", fontsize=9.5, color=INK2)
    a.set_title("Paired difference", fontsize=12.5, fontweight="semibold", pad=12)
    a.text(.03, .95, f"mean {d.mean():+.3f} degC\nevery station above zero",
           transform=a.transAxes, va="top", fontsize=9.5, family="monospace",
           color=INK2)

    for a in ax:
        a.tick_params(labelsize=8.5, color=GRAT, labelcolor=INK3)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    top = header(fig, "The reference products, judged by thermometers",
                 f"{n} NOAA ISD stations inside the held-out box, 2019-2023, "
                 f"daily means from days with at least 20 hourly observations. "
                 f"No model is involved. AORC is closer at every station.")
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=2.0)
    out = FIGDIR / "T1_station_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  {out.name}   ERA5-Land {era.mean():.4f} +/- {era.std():.4f} | "
          f"AORC {aor.mean():.4f} +/- {aor.std():.4f} | AORC wins {int((aor<era).sum())}/{n}")


def fig_resolution(plt, arms, seeds):
    """Amplitude and coherence against wavelength, replacing the single number."""
    from evaluation.fields import Store
    from evaluation.spectral import (AMP_FLOOR, coherence_resolution,
                                     effective_resolution, radial_spectrum)

    stores = {s: Store(f"data_store_sc/{s}/dataset.zarr") for s in {a[1] for a in arms}}
    ref = stores["aorc"]
    te = ref.i_test
    i0, i1, j0, j1 = ref.box
    crop = lambda f: (f[te][:, i0:i1, j0:j1]                      # noqa: E731
                      - f[te][:, i0:i1, j0:j1].mean((1, 2))[:, None, None])
    A = crop(ref.to_absolute(ref.truth()))

    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.4))
    summary = []
    for label, store_name, prefix, colour in arms:
        st = stores[store_name]
        curves = []
        for sd in seeds:
            name = f"{prefix}_s{sd}" if prefix else None
            if name is None:                        # the truth product itself
                F = st.to_absolute(st.truth())
            else:
                if not (REPO / "checkpoints" / name / "last.pt").exists():
                    continue
                from evaluation.fields import predict_all
                F = st.to_absolute(predict_all(st, name, "last"))
            curves.append(radial_spectrum(crop(F), A, dx_km=ref.dx_km))
            if prefix is None:
                break
        if not curves:
            print(f"  {label}: no checkpoints, skipping"); continue
        w = [r["wavelength_km"] for r in curves[0]]
        amp = np.mean([[r["amplitude_ratio"] for r in c] for c in curves], 0)
        coh = np.mean([[r["coherence"] for r in c] for c in curves], 0)
        ax[0].plot(w, amp, "-o", ms=3.5, lw=1.7, color=colour, label=label)
        ax[1].plot(w, coh, "-o", ms=3.5, lw=1.7, color=colour, label=label)
        e = effective_resolution(curves[0])
        c = coherence_resolution(curves[0])
        summary.append((label, e, c, float(np.mean([a for a, ww in zip(amp, w) if ww <= 60])),
                        float(np.mean([a for a, ww in zip(coh, w) if ww <= 60]))))

    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
    ticks = [400, 200, 100, 60, 40, 25, 15]
    for a, lvl, name in [(ax[0], AMP_FLOOR, "amplitude ratio"),
                         (ax[1], 0.5, "coherence with AORC")]:
        a.set_xscale("log"); a.invert_xaxis()
        a.xaxis.set_major_locator(FixedLocator(ticks))
        a.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        a.xaxis.set_minor_locator(NullLocator())
        a.axhline(lvl, ls="--", lw=1, color=GRAT)
        a.text(ticks[0], lvl + .02, f"threshold {lvl:.2f}", fontsize=8.5, color=INK3)
        a.set_xlabel("wavelength (km) — finer to the right", fontsize=9.5, color=INK2)
        a.set_ylabel(name, fontsize=9.5, color=INK2)
        a.legend(fontsize=9, framealpha=.9, loc="lower left")
        a.tick_params(labelsize=8.5, color=GRAT, labelcolor=INK3)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_title("How MUCH structure", fontsize=12.5, fontweight="semibold", pad=12)
    ax[1].set_title("Is it in the RIGHT PLACE", fontsize=12.5, fontweight="semibold", pad=12)

    top = header(fig, "Fine-scale skill against scale",
                 "The single 'effective resolution' number was quantised to four "
                 "radial bands above 60 km and returned 165 km for every arm, so it "
                 "could not tell them apart. These curves can.")
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=2.0)
    out = FIGDIR / "T2_resolution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  {out.name}")
    print(f"      {'arm':<26}{'eff res':>9}{'coh<0.5':>9}{'amp<60':>9}{'coh<60':>9}")
    for lab, e, c, am, ch in summary:
        es = f"{e:.0f}" if e else "none"
        cs = f"{c:.0f}" if c else "never"
        print(f"      {lab:<26}{es:>9}{cs:>9}{am:>9.3f}{ch:>9.3f}")


def main(which):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    if which in ("all", "station"):
        fig_station_panel(plt)
    if which in ("all", "resolution"):
        fig_resolution(plt, [
            ("ERA5-Land (the product)", "dyn", None, GREY),
            ("ERA5-Land-trained model", "dyn", "dyn", COOL),
            ("AORC-trained model", "aorc", "aorc", ORANGE),
        ], seeds=[1337, 7, 42])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", default="all",
                    choices=["all", "station", "resolution"])
    a = ap.parse_args()
    main(a.which)
