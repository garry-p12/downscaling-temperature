"""Where the fine-scale detail is invented — 5.5 and 5.8 drawn as maps.

The spectral figure answers "how much detail, and is it in phase?" as two
curves against wavelength. It does not answer WHERE, and the two questions come
apart on the ground: the deep model's excess fine-scale amplitude is not spread
evenly, it is concentrated over the rough western terrain where a sharpener has
the most edges to key off and the least skill to place them.

    row 1   context, then the fine-scale standard deviation of truth, the deep
            model and the affine control, on ONE shared colour scale
    row 2   per-cell amplitude ratio / per-cell coherence for each model /
            the difference between them

Row 1 shows the YEAR's fine-scale sd per cell rather than one day's high-pass
field. One day cannot be drawn on a shared scale and stay legible: the truth's
sub-60 km amplitude is five to ten times the models', so the two model panels
render as blank paper. That gap IS the finding, but a blank panel does not
communicate it — the year's sd does, and keeps every panel on one scale.

AMPLITUDE vs COHERENCE, which is the whole point of 5.8. Panel 5 asks whether a
cell carries the right AMOUNT of fine-scale variation; panels 6-7 ask whether
that variation is in the RIGHT PLACE, as the correlation through time between
the model's high-pass field and the truth's, cell by cell. A sharpener scores
well on the first and badly on the second. Reading either alone is how a blurry
model passes for a resolving one.

THE FILTER. Gaussian high-pass, cut at HALF AMPLITUDE (not half power) at the
requested wavelength: the Gaussian smoother's transfer is exp(-2 pi^2 k^2
sigma^2), so the high-pass transfer 1 - exp(-2 pi^2 k^2 sigma^2) reaches 1/2 at
k = 1/lambda when sigma = sqrt(ln 2)/(pi sqrt 2) * lambda. At 60 km on a 10 km
grid that is sigma = 1.12 cells: a wave of exactly 60 km comes through at half
its amplitude, longer waves weaker, shorter waves stronger. Edges use 'nearest'
rather than zero padding, so the domain rim does not manufacture its own
high-pass signal.

Usage:
    python scripts/make_coherence_panels.py --deep v2_landcov_s1337 --affine lin_s1337
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.fields import Store, cellwise_corr, highpass, predict_all  # noqa: E402
from evaluation.mapping import (  # noqa: E402
    GRAT,
    INK,
    INK3,
    SURF,
    graticule,
    header,
    holdout_box,
    palettes,
)

FIGDIR = REPO / "image_outputs" / "slides"


def sigma_for_cutoff(lambda_km: float, dx_km: float = 10.0) -> float:
    """Gaussian sigma, in CELLS, whose high-pass halves AMPLITUDE at lambda_km."""
    return float(np.sqrt(np.log(2)) / (np.pi * np.sqrt(2)) * lambda_km / dx_km)


def main(deep, affine, zarr, ckpt, cut_km, day):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    seq, div = palettes()
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                         "savefig.facecolor": SURF, "text.color": INK,
                         "font.size": 10})

    st = Store(zarr)
    te = st.i_test
    land, box, hold = st.land, st.box, st.m_hold
    nan = lambda f: np.where(land, f, np.nan)                  # noqa: E731
    sig = sigma_for_cutoff(cut_km)

    truth = st.truth()[te]
    p_deep = predict_all(st, deep, ckpt)[te]
    p_aff = predict_all(st, affine, ckpt)[te]

    hp_t, hp_d, hp_a = (highpass(f, sig) for f in (truth, p_deep, p_aff))

    # Per-cell fine-scale amplitude and phase agreement over the test year.
    sd_t, sd_d, sd_a = hp_t.std(0), hp_d.std(0), hp_a.std(0)
    # Two rings of cells are dropped from every fine-scale panel. The Gaussian
    # filter's 'nearest' edge rule and the networks' own convolution padding
    # both fabricate high-pass signal at the domain rim, which shows up as a
    # bright frame on the sd maps and as a ratio blow-up where the truth's
    # high-pass sd is near zero. Neither is a result. Four rings clear it; the
    # evaluated box is rows 36-73 of 93 and columns 45-80 of 100, so nothing
    # quoted depends on the cells dropped.
    RIM = 4
    inner = np.zeros_like(land)
    inner[RIM:-RIM, RIM:-RIM] = True
    fine = land & inner
    with np.errstate(invalid="ignore", divide="ignore"):
        amp_d = np.where(fine, sd_d / sd_t, np.nan)
        amp_a = np.where(fine, sd_a / sd_t, np.nan)
    coh_d = cellwise_corr(hp_d, hp_t, fine)
    coh_a = cellwise_corr(hp_a, hp_t, fine)
    dcoh = np.where(fine, coh_d - coh_a, np.nan)

    def box_mean(f):
        return float(np.nanmean(f[hold]))

    # Share of box cells where the deep model places detail WORSE than the
    # affine control. Masking with np.where(..., np.nan) and then comparing
    # would silently count NaN as False over the whole grid, giving the
    # fraction of the DOMAIN rather than of the box.
    worse = float(np.mean(dcoh[hold] < 0) * 100)

    # Representative day: MEDIAN fine-scale activity in the box, so the maps
    # show a typical day rather than the most or least structured one.
    if day is None:
        act = hp_t[:, hold].std(axis=1)
        kk = int(np.argsort(act)[len(act) // 2])
    else:
        import pandas as pd
        kk = int(np.argmin(np.abs(st.times[te] - pd.Timestamp(day))))
    day = st.times[te][kk].strftime("%Y-%m-%d")

    fig, ax = plt.subplots(2, 4, figsize=(24.0, 11.8))

    def show(a, f, t, cmap, sub="", label="degC", **kw):
        im = a.imshow(np.ma.masked_invalid(f), cmap=cmap, **kw)
        # Pin the axes box to the grid's own aspect. Without it the axes stays
        # as wide as its subplot slot, imshow centres a narrower image inside
        # it, and the colour bar ends up floating in the gap.
        a.set_box_aspect(st.shape[0] / st.shape[1])
        graticule(a, st.LAT, st.LON, fontsize=8.5)
        holdout_box(a, box, "")
        a.set_title(t, fontsize=13, fontweight="semibold", pad=16)
        if sub:
            a.text(.5, 1.004, sub, transform=a.transAxes, ha="center",
                   va="bottom", fontsize=8.6, color=INK3)
        cb = fig.colorbar(im, ax=a, fraction=.038, pad=.015)
        cb.ax.tick_params(labelsize=8.5, color=GRAT, labelcolor=INK3)
        cb.set_label(label, fontsize=9, color=INK3)

    # --- row 1: context, then fine-scale amplitude --------------------------
    full = st.to_absolute(st.truth())[te][kk]
    show(ax[0, 0], nan(full), f"Truth — full field, {day}", seq,
         "ERA5-Land, absolute degC — context for the scale of the signal",
         vmin=float(np.nanpercentile(full[land], 1)),
         vmax=float(np.nanpercentile(full[land], 99)))

    # Shared scale set from the TRUTH's own distribution, clipped at p92 so the
    # coastal front (which is an order of magnitude sharper than anywhere
    # inland) does not flatten the whole interior to one colour.
    vsd = float(np.percentile(sd_t[fine], 92))
    for a, f, t, sub in [
            (ax[0, 1], sd_t, f"Truth — detail below {cut_km:.0f} km",
             f"per-cell sd over {len(te)} days; box mean {sd_t[hold].mean():.3f} degC"),
            (ax[0, 2], sd_d, "Deep model — same band",
             f"box mean {sd_d[hold].mean():.3f} degC — same colour scale"),
            (ax[0, 3], sd_a, "Affine control — same band",
             f"box mean {sd_a[hold].mean():.3f} degC — same colour scale")]:
        show(a, np.where(fine, f, np.nan), t, seq, sub, vmin=0, vmax=vsd)

    # --- row 2: the year ----------------------------------------------------
    show(ax[1, 0], amp_d, "Fine-scale AMPLITUDE, deep / truth", div,
         f"1.0 = correct amount of detail; box mean {box_mean(amp_d):.2f}",
         label="ratio", norm=TwoSlopeNorm(1.0, 0.0, 2.0))
    cmax = float(np.nanpercentile(np.r_[coh_d[fine], coh_a[fine]], 99))
    show(ax[1, 1], coh_d, "Fine-scale COHERENCE, deep", seq,
         f"per-cell r with truth over {len(te)} days; box mean "
         f"{box_mean(coh_d):.3f}", label="r", vmin=0, vmax=cmax)
    show(ax[1, 2], coh_a, "Fine-scale COHERENCE, affine control", seq,
         f"same scale; box mean {box_mean(coh_a):.3f}", label="r",
         vmin=0, vmax=cmax)
    dlim = float(np.nanpercentile(np.abs(dcoh[fine]), 99))
    show(ax[1, 3], dcoh, "Coherence: deep MINUS affine", div,
         f"blue = the 2.2k-parameter control places detail better — "
         f"{worse:.0f}% of box cells",
         label="delta r", norm=TwoSlopeNorm(0, -dlim, dlim))

    top = header(
        fig,
        f"Fine-scale detail: how much, and in the right place?   "
        f"(below {cut_km:.0f} km, {ckpt}.pt)",
        f"Sub-{cut_km:.0f} km detail, {len(te)} test days. AMPLITUDE = is there the "
        f"right AMOUNT of it; COHERENCE = is it in the right PLACE. The deep model wins "
        f"the first and loses the second.")
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=2.6)
    out = FIGDIR / f"coherence_{deep.rsplit('_s', 1)[0]}.png"
    fig.savefig(out, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}   BOX amplitude deep {box_mean(amp_d):.3f} / affine "
          f"{box_mean(amp_a):.3f}   coherence deep {box_mean(coh_d):.3f} / affine "
          f"{box_mean(coh_a):.3f}   deep worse in {worse:.0f}% of box cells")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deep", default="v2_landcov_s1337")
    ap.add_argument("--affine", default="lin_s1337")
    ap.add_argument("--zarr", default="data_store_sc/super/dataset.zarr")
    ap.add_argument("--ckpt", default="last", choices=["best", "last"])
    ap.add_argument("--cut-km", type=float, default=60.0)
    ap.add_argument("--day", default=None,
                    help="default: the median fine-scale-activity test day")
    a = ap.parse_args()
    main(a.deep, a.affine, a.zarr, a.ckpt, a.cut_km, a.day)
