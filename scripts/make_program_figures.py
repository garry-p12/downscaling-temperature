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


if __name__ == "__main__":
    main()
