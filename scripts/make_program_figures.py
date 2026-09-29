"""Two figures for the loss/offset program: the estimator ladder, and the
station validation that decides whether any of it is a real accuracy gain.

    offset_estimator_ladder.png   how much of the offset oracle each estimator
                                  recovers, and where the k-sweep turns over
    station_validation.png        every method scored against THERMOMETERS,
                                  with ERA5-Land's own row as the ceiling

The station figure is the important one. Everything else in this project is
measured against ERA5-Land, and the corrected model now sits ~0.55 degC from
it while ERA5-Land is itself ~0.9 degC from these stations — so a gain against
the reanalysis can be an artefact of fitting the reanalysis. Only thermometers
can tell the difference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from evaluation.mapping import GRAT, INK, INK2, INK3, ORANGE, SURF  # noqa: E402

FIG = REPO / "image_outputs" / "slides"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                         "savefig.facecolor": SURF, "text.color": INK,
                         "font.size": 10.5})
    FIG.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- ladder
    ks = json.load(open(REPO / "outputs/offset_ksweep2.json"))["of0_s1337"]["metrics"]
    dyn = json.load(open(REPO / "outputs/offset_dyn.json"))["of0_s1337"]["metrics"]
    u = ks["uncorrected"]
    g = lambda v: (v - u) / u * 100            # noqa: E731

    bars = [
        ("dynamic POWER alone", g(dyn["FAIR: dynamic POWER over the holdout ONLY"]), "#c9c6bf"),
        ("1 global mean", g(ks["spatial (train-region offset, same day)"]), "#9ec5f4"),
        ("9 blocks (k=3)", g(ks["FAIR: 9 sub-region means (k=3)"]), "#6aa5e8"),
        ("36 blocks (k=6)", g(ks["FAIR: 36 sub-region means (k=6)"]), "#2a78d6"),
        ("36 blocks + dynamic", g(dyn["FAIR: sub-regions k=6 + dynamic POWER"]), "#7d8996"),
        ("oracle (true offset)", g(ks["oracle (true holdout offset)"]), ORANGE),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.4),
                           gridspec_kw={"width_ratios": [1.25, 1]})
    a = ax[0]
    a.barh([b[0] for b in bars], [-b[1] for b in bars],
           color=[b[2] for b in bars], height=.6)
    for i, (_, v, _) in enumerate(bars):
        a.text(-v, i, f"  {-v:.1f}%", va="center", fontsize=10.5,
               fontweight="semibold", color=INK)
    a.set_xlim(0, -min(b[1] for b in bars) * 1.22)
    a.set_xlabel("RMSE reduction over the held-out region (%)", fontsize=9.5, color=INK2)
    a.set_title("What recovers the daily bias", fontsize=13, fontweight="semibold", pad=12)
    a.text(0, 1.012, "of0_s1337; none of these use held-out truth",
           transform=a.transAxes, fontsize=8.2, color=INK3)
    for sp in ("top", "right", "left"):
        a.spines[sp].set_visible(False)
    a.tick_params(labelsize=10, color=GRAT, labelcolor=INK2)

    # k sweep
    b = ax[1]
    kk = [(3, "FAIR: 9 sub-region means (k=3)"), (6, "FAIR: 36 sub-region means (k=6)"),
          (8, "FAIR: 64 sub-region means (k=8)"), (10, "FAIR: 100 sub-region means (k=10)"),
          (12, "FAIR: 144 sub-region means (k=12)"), (16, "FAIR: 256 sub-region means (k=16)")]
    xs = [k for k, _ in kk]
    ys = [-g(ks[n]) for _, n in kk]
    b.plot(xs, ys, "-o", color="#2a78d6", lw=2, ms=6)
    b.axhline(-g(ks["oracle (true holdout offset)"]), ls="--", lw=1.4,
              color=ORANGE, label="oracle")
    b.axhline(-g(ks["spatial (train-region offset, same day)"]), ls=":", lw=1.4,
              color=INK3, label="1 global mean")
    b.scatter([6], [-g(ks["FAIR: 36 sub-region means (k=6)"])], s=160,
              facecolor="none", edgecolor=ORANGE, lw=2.2, zorder=5)
    b.annotate("k=6 chosen:\nnear the plateau,\nfewest parameters",
               xy=(6, -g(ks["FAIR: 36 sub-region means (k=6)"])),
               xytext=(7.6, -g(ks["spatial (train-region offset, same day)"]) + 3),
               fontsize=8.6, color=INK2,
               arrowprops=dict(arrowstyle="->", color=INK3, lw=1))
    b.set_xlabel("blocks per side (k)", fontsize=9.5, color=INK2)
    b.set_ylabel("RMSE reduction (%)", fontsize=9.5, color=INK2)
    b.set_title("Where more blocks stop helping", fontsize=13,
                fontweight="semibold", pad=12)
    b.text(0, 1.012, "k=16 degrades: 256 features start fitting train-split noise",
           transform=b.transAxes, fontsize=8.2, color=INK3)
    b.legend(fontsize=8.5, loc="lower right", framealpha=.9)
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)
    b.tick_params(labelsize=9.5, color=GRAT, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "offset_estimator_ladder.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  offset_estimator_ladder.png")

    # ------------------------------------------------------------- stations
    sts = ["KAUS_Austin_Bergstrom", "KATT_Austin_Executive"]
    data = {}
    for st in sts:
        f = REPO / f"outputs/station_check_{st}_2023.json"
        if f.exists():
            data[st] = json.load(open(f))["methods"]
    if not data:
        print("  (no station json yet)"); return

    order = ["interpolated_POWER", "v2_landcov_s1337",
             "v2_landcov_s1337 + offset correction", "ERA5-Land (our 'truth')"]
    label = {"interpolated_POWER": "interpolated POWER",
             "v2_landcov_s1337": "model (V2)",
             "v2_landcov_s1337 + offset correction": "model + offset correction",
             "ERA5-Land (our 'truth')": "ERA5-Land  (the target itself)"}
    col = {"interpolated_POWER": "#c9c6bf", "v2_landcov_s1337": "#9aa5b1",
           "v2_landcov_s1337 + offset correction": "#2a78d6",
           "ERA5-Land (our 'truth')": ORANGE}

    fig, ax = plt.subplots(1, len(data), figsize=(7.6 * len(data), 5.2),
                           squeeze=False)
    for n, (st, M) in enumerate(data.items()):
        a = ax[0, n]
        keys = [k for k in order if k in M]
        vals = [M[k]["rmse"] for k in keys]
        a.barh([label[k] for k in keys], vals, color=[col[k] for k in keys], height=.55)
        for i, v in enumerate(vals):
            a.text(v, i, f"  {v:.3f}", va="center", fontsize=11,
                   fontweight="semibold", color=INK)
        a.axvline(M["ERA5-Land (our 'truth')"]["rmse"], ls="--", lw=1.3,
                  color=ORANGE, zorder=0)
        a.set_xlim(0, max(vals) * 1.28)
        a.set_xlabel("RMSE against station observations, degC", fontsize=9.5, color=INK2)
        a.set_title(f"{st.split('_')[0]} — {M[keys[0]]['n_days']} days of 2023",
                    fontsize=13, fontweight="semibold", pad=12)
        for sp in ("top", "right", "left"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=10, color=GRAT, labelcolor=INK2)
    fig.suptitle("The correction survives the only independent test — and meets the ceiling",
                 x=.006, y=1.01, ha="left", va="top", fontsize=15.5, fontweight="semibold")
    fig.text(.006, .945,
             "Scored against NOAA ISD thermometers, not ERA5-Land. The dashed line is "
             "ERA5-Land's OWN distance from the station: everything in this project is "
             "trained toward that product, so it is the accuracy ceiling. The corrected "
             "model reaches it at both sites.",
             ha="left", va="top", fontsize=9.4, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(FIG / "station_validation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  station_validation.png")




def spectral_figure():
    """C4/C5: amplitude ratio and coherence vs wavelength, affine vs deep.

    The claim this figure has to carry is the CROSSOVER: the deep model holds
    more fine-scale amplitude than the affine one while having LESS coherence
    with truth there. Amplitude alone looks like sharpness; only the pair shows
    it is manufactured.
    """
    import json
    import re

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = REPO / "outputs/linear_probe/spectral_5seed.json"
    if not src.exists():
        print("  (no spectral_5seed.json)"); return
    d = json.load(open(src))

    def arm(p):
        ks = [k for k in d if re.fullmatch(rf"{p}_s\d+", k)]
        lam = np.array([r["wavelength_km"] for r in d[ks[0]]["spectrum"]])
        A = np.array([[r["amplitude_ratio"] for r in d[k]["spectrum"]] for k in ks])
        C = np.array([[r["coherence"] for r in d[k]["spectrum"]] for k in ks])
        return lam, A, C

    lam, Al, Cl = arm("lin")
    _, Ad, Cd = arm("v2_landcov")
    keep = (lam >= 20) & (lam <= 500)
    lam = lam[keep]; Al, Cl, Ad, Cd = Al[:, keep], Cl[:, keep], Ad[:, keep], Cd[:, keep]

    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                         "savefig.facecolor": SURF, "text.color": INK,
                         "font.size": 10.5})
    fig, ax = plt.subplots(1, 2, figsize=(14.6, 5.6))
    DEEP, LIN = "#2a78d6", "#eb6834"

    for a, (Y1, Y2, lab, thr) in zip(ax, [
            (Al, Ad, "amplitude ratio   (1.0 = correct variance)", np.sqrt(0.75)),
            (Cl, Cd, "coherence with truth   (1.0 = perfectly in phase)", None)]):
        for Y, c, nm in [(Y2, DEEP, "deep (210k params)"),
                         (Y1, LIN, "affine control (2.2k params)")]:
            m, s = Y.mean(0), Y.std(0, ddof=1)
            a.plot(lam, m, "-o", color=c, lw=2, ms=4.5, label=nm)
            a.fill_between(lam, m - s, m + s, color=c, alpha=.18, lw=0)
        a.axvspan(20, 60, color="#c9c6bf", alpha=.28, lw=0, zorder=0)
        from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
        a.set_xscale("log")
        # Kill the minor decade labels: on a log axis matplotlib adds 3x10^2,
        # 6x10^1 etc, which collide with the wavelengths we actually want read.
        a.xaxis.set_minor_locator(NullLocator())
        a.xaxis.set_minor_formatter(NullFormatter())
        a.xaxis.set_major_locator(FixedLocator([20, 30, 50, 100, 200, 500]))
        a.set_xticklabels(["20", "30", "50", "100", "200", "500"])
        a.invert_xaxis()
        a.set_xlabel("wavelength (km)   — finer scales to the right",
                     fontsize=9.5, color=INK2)
        a.set_ylabel(lab, fontsize=9.5, color=INK2)
        if thr:
            a.axhline(thr, ls="--", lw=1.2, color=INK3)
            a.text(300, thr - .075, "effective-resolution threshold",
                   fontsize=8, color=INK3)
        a.legend(fontsize=9, loc="lower left", framealpha=.9)
        a.tick_params(labelsize=9, color=GRAT, labelcolor=INK3)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    ax[0].set_title("The deep model holds MORE fine-scale variance",
                    fontsize=12.5, fontweight="semibold", pad=10)
    ax[1].set_title("...but is LESS in phase with truth there",
                    fontsize=12.5, fontweight="semibold", pad=10)
    ax[1].annotate("deep model worse below 60 km\n(Welch t = −9.3)",
                   xy=(35, Cd[:, lam <= 60].mean()),
                   xytext=(90, 0.42), fontsize=9, color=INK2,
                   arrowprops=dict(arrowstyle="->", color=INK3, lw=1.1))
    fig.suptitle("Capacity is not the constraint — and excess capacity manufactures structure",
                 x=.006, y=1.0, ha="left", va="top", fontsize=15.5,
                 fontweight="semibold")
    fig.text(.006, .935,
             "Shaded band 20–60 km. Bands are ±1 sd across seeds (5 affine, 3 deep). "
             "In that band the deep model produces ~3x the variance of an affine one at "
             "LOWER coherence: amplitude alone reads as sharpness, but the pair shows it "
             "is not resolving truth. Left panel also gives C5: both models fall below "
             "the threshold by ~165 km, on a 10 km grid.",
             ha="left", va="top", fontsize=9.2, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, .885))
    fig.savefig(FIG / "spectral_affine_vs_deep.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  spectral_affine_vs_deep.png")


if __name__ == "__main__":
    main()
    spectral_figure()
