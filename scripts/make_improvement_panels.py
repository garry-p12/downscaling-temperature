"""What the offset correction buys, cell by cell and region by region (5.7).

make_offset_panels.py shows ONE day: the mechanism, the scatter, the ladder.
This shows the whole test year on the ground — which cells get better, by how
much, and whether the result survives being asked for somewhere other than
Austin.

    row 1   model RMSE / corrected RMSE / the difference / the difference as %
    row 2   model bias / corrected bias / the five LORO regions / their scores

Rows 1 and 2 answer different objections. Row 1 is "does it help everywhere, or
is the headline number carried by a few cells?" Row 2 panels 1-2 are "is it
really the bias it claims to be fixing?" — the correction subtracts one number
per day, so if the mechanism is what 5.4 says it is, the time-mean error map
should collapse toward zero while the RMSE map improves more modestly. Row 2
panels 3-4 are "does this only work for Austin?"

LORO CAVEAT. The figure carries the short form of this ("austin is the only
region actually held out of training"); the long form is here. The checkpoint
was trained with AUSTIN held out. For the other four regions the model has seen
those cells in training, so their model RMSE is optimistic. What transfers is
the ESTIMATOR — whether a region's daily bias can be recovered from the rest of
the domain — not a second measurement of model skill. The % of oracle is the honest number
across regions because it is a ratio within each region.

Usage:
    python scripts/make_improvement_panels.py --archs of0_s1337 v2_landcov_s1337
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.fields import Store, predict_all  # noqa: E402
from evaluation.mapping import (  # noqa: E402
    GRAT,
    INK,
    INK2,
    INK3,
    ORANGE,
    SURF,
    graticule,
    header,
    holdout_box,
    palettes,
)
from evaluation.offset_correction import (  # noqa: E402
    REGIONS,
    ridge_fit_predict,
    subregion_means,
)
from training.dataset import holdout_bounds  # noqa: E402

FIGDIR = REPO / "image_outputs" / "slides"
TITLES = {"of0": "Offset-split loss (of0)",
          "v2_landcov": "V2 land-cover (project default)",
          "v0_base": "V0 baseline",
          "lin": "Linear probe (affine control)"}


def correct(resid, m_hold, m_train, shape, i_train, k=6):
    """Per-day offset estimate for m_hold, fit on the train split only.

    Returns (pred_offset, true_offset) for EVERY day in the store. The ridge is
    fit on the train days and applied to all of them, so the test-day estimates
    are out of sample.
    """
    ny, nx = shape
    off_true = resid[:, m_hold].mean(1)
    X = subregion_means(resid, m_train, ny, nx, k=k)
    return ridge_fit_predict(X, off_true, i_train), off_true


def loro(st, resid, k=6):
    """Run the estimator once per leave-one-region-out box."""
    rows = []
    for name, hcfg in REGIONS.items():
        i0, i1, j0, j1 = holdout_bounds(st.ds, hcfg)
        hold = np.zeros_like(st.land)
        hold[i0:i1, j0:j1] = True
        mh, mt = hold & st.land, (~hold) & st.land
        if mh.sum() < 50:
            continue
        est, true = correct(resid, mh, mt, st.shape, st.i_train, k)
        te = st.i_test
        r = resid[te][:, mh]
        r_model = float(np.sqrt((r ** 2).mean()))
        r_corr = float(np.sqrt(((r - est[te, None]) ** 2).mean()))
        r_orac = float(np.sqrt(((r - true[te, None]) ** 2).mean()))
        rows.append(dict(name=name, box=(i0, i1, j0, j1), cells=int(mh.sum()),
                         model=r_model, corrected=r_corr, oracle=r_orac,
                         pct=(r_corr - r_model) / r_model * 100,
                         of_oracle=(r_corr - r_model) / (r_orac - r_model) * 100))
    return rows


def main(archs, zarr, ckpt, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    seq, div = palettes()
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                         "savefig.facecolor": SURF, "text.color": INK,
                         "font.size": 10})

    st = Store(zarr)
    te, land, box, hold = st.i_test, st.land, st.box, st.m_hold
    truth = st.truth()
    nan = lambda f: np.where(land, f, np.nan)                    # noqa: E731

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for arch in archs:
        resid = truth - predict_all(st, arch, ckpt)              # (T, H, W)
        est, true_off = correct(resid, hold, st.m_train, st.shape, st.i_train, k)

        r_m = resid[te]
        r_c = r_m - est[te, None, None]
        rmse_m, rmse_c = nan(np.sqrt((r_m ** 2).mean(0))), nan(np.sqrt((r_c ** 2).mean(0)))
        bias_m, bias_c = nan(r_m.mean(0)), nan(r_c.mean(0))
        d_abs = rmse_c - rmse_m
        with np.errstate(invalid="ignore", divide="ignore"):
            d_pct = nan(d_abs / rmse_m * 100)

        bm = lambda f: float(np.nanmean(f[hold]))                # noqa: E731
        R_m = float(np.sqrt((r_m[:, hold] ** 2).mean()))
        R_c = float(np.sqrt((r_c[:, hold] ** 2).mean()))
        R_o = float(np.sqrt(((r_m - true_off[te, None, None])[:, hold] ** 2).mean()))
        rows = loro(st, resid, k)

        fig, ax = plt.subplots(2, 4, figsize=(24.0, 11.8))

        def show(a, f, t, cmap, sub="", label="degC", **kw):
            im = a.imshow(np.ma.masked_invalid(f), cmap=cmap, **kw)
            # Pin the axes box to the grid's own aspect. Without it the axes
            # stays as wide as its subplot slot, imshow centres a narrower
            # image inside it, and the colour bar ends up floating in the gap.
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
            return a

        vmax = float(np.nanpercentile(rmse_m, 99))
        show(ax[0, 0], rmse_m, "Model RMSE, per cell", seq,
             f"{len(te)} test days; box {R_m:.3f} degC", vmin=0, vmax=vmax)
        show(ax[0, 1], rmse_c, f"After the k={k} offset correction", seq,
             f"same colour scale; box {R_c:.3f} degC", vmin=0, vmax=vmax)
        dlim = float(np.nanpercentile(np.abs(d_abs[land]), 99))
        show(ax[0, 2], nan(d_abs), "Change in RMSE", div,
             f"blue = better; box mean {bm(nan(d_abs)):+.3f} degC",
             norm=TwoSlopeNorm(0, -dlim, dlim))
        plim = float(np.nanpercentile(np.abs(d_pct[land]), 99))
        show(ax[0, 3], d_pct, "Change in RMSE, per cent", div,
             f"box {(R_c - R_m) / R_m:+.1%} overall; improves in "
             f"{np.mean(d_abs[hold] < 0) * 100:.0f}% of box cells",
             label="%", norm=TwoSlopeNorm(0, -plim, plim))

        blim = float(np.nanpercentile(np.abs(bias_m[land]), 99))
        show(ax[1, 0], bias_m, "Model bias (time-mean error)", div,
             f"box mean {bm(bias_m):+.3f} degC", norm=TwoSlopeNorm(0, -blim, blim))
        show(ax[1, 1], bias_c, "Bias after correction", div,
             f"same colour scale; box mean {bm(bias_c):+.3f} degC",
             norm=TwoSlopeNorm(0, -blim, blim))

        # --- LORO map --------------------------------------------------------
        a = ax[1, 2]
        base = np.where(land, 0.0, np.nan)
        a.imshow(np.ma.masked_invalid(base), cmap=seq, vmin=0, vmax=1)
        a.set_box_aspect(st.shape[0] / st.shape[1])
        graticule(a, st.LAT, st.LON, fontsize=8.5)
        from matplotlib import colormaps
        from matplotlib.cm import ScalarMappable
        vals = [r["of_oracle"] for r in rows]
        norm = plt.Normalize(min(vals) - 6, max(vals) + 2)
        for r in rows:
            i0, i1, j0, j1 = r["box"]
            c = colormaps["Blues"](norm(r["of_oracle"]))
            a.add_patch(plt.Rectangle((j0 - .5, i0 - .5), j1 - j0, i1 - i0,
                                      facecolor=c, alpha=.85,
                                      ec=ORANGE if r["name"] == "austin" else INK2,
                                      lw=1.8 if r["name"] == "austin" else 1.1, zorder=5))
            a.text((j0 + j1) / 2, (i0 + i1) / 2,
                   f"{r['name']}\n{r['of_oracle']:.0f}%\n{r['pct']:+.1f}%",
                   ha="center", va="center", fontsize=7.4, zorder=6,
                   color=INK, fontweight="semibold")
        a.set_title("Leave-one-region-out", fontsize=13,
                    fontweight="semibold", pad=17)
        a.text(.5, 1.004, "shade and top number = % of oracle recovered; "
               "below it = RMSE change", transform=a.transAxes, ha="center",
               va="bottom", fontsize=8.6, color=INK3)
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap="Blues"), ax=a,
                          fraction=.038, pad=.015)
        cb.ax.tick_params(labelsize=8.5, color=GRAT, labelcolor=INK3)
        cb.set_label("% of oracle", fontsize=8, color=INK3)

        # --- LORO bars -------------------------------------------------------
        a = ax[1, 3]
        order = sorted(rows, key=lambda r: r["of_oracle"])
        y = np.arange(len(order))
        a.barh(y, [r["of_oracle"] for r in order], height=.55,
               color=["#2a78d6" if r["name"] == "austin" else "#9ec5f4"
                      for r in order])
        for i, r in enumerate(order):
            a.text(r["of_oracle"] + 1.2, i, f"  {r['of_oracle']:.0f}%   "
                   f"({r['pct']:+.1f}% RMSE)", va="center", fontsize=9,
                   color=INK, fontweight="semibold" if r["name"] == "austin" else "normal")
        m, sdv = np.mean(vals), np.std(vals)
        a.axvline(m, color=INK3, ls="--", lw=1)
        a.set_yticks(y); a.set_yticklabels([r["name"] for r in order], fontsize=9)
        a.set_xlim(0, max(vals) * 1.52)
        a.set_xlabel("% of the oracle offset recovered", fontsize=8.5, color=INK2)
        a.set_title(f"Across regions: {m:.0f}% +/- {sdv:.0f}", fontsize=13,
                    fontweight="semibold", pad=16)
        a.text(.5, 1.004, "austin is the only region actually held out of training",
               transform=a.transAxes, ha="center", va="bottom",
               fontsize=7.6, color=INK3)
        a.tick_params(labelsize=9, color=GRAT, labelcolor=INK2)
        for sp in ("top", "right", "left"):
            a.spines[sp].set_visible(False)

        b = arch.rsplit("_s", 1)[0]
        top = header(
            fig,
            f"What the offset correction buys — {TITLES.get(b, b)}   "
            f"({arch}, {ckpt}.pt, k={k})",
            f"One number per day, predicted from {k * k} block means of the training "
            f"region, never from held-out truth. Box RMSE {R_m:.3f} -> {R_c:.3f} degC "
            f"({(R_c - R_m) / R_m:+.1%}), {(R_c - R_m) / (R_o - R_m) * 100:.0f}% of the "
            f"oracle floor {R_o:.3f}.")
        fig.tight_layout(rect=(0, 0, 1, top), h_pad=2.6)
        out = FIGDIR / f"improvement_{b}.png"
        fig.savefig(out, dpi=155, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out.name}   BOX {R_m:.4f} -> {R_c:.4f} ({(R_c - R_m) / R_m:+.1%}, "
              f"{(R_c - R_m) / (R_o - R_m) * 100:.0f}% of oracle {R_o:.4f})   "
              f"bias {bm(bias_m):+.3f} -> {bm(bias_c):+.3f}   "
              f"improves in {np.mean(d_abs[hold] < 0) * 100:.0f}% of box cells")
        for r in order:
            print(f"      {r['name']:<10} {r['cells']:>5} cells   "
                  f"{r['model']:.4f} -> {r['corrected']:.4f} ({r['pct']:+.1f}%)   "
                  f"{r['of_oracle']:.0f}% of oracle")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=["of0_s1337", "v2_landcov_s1337"])
    ap.add_argument("--zarr", default="data_store_sc/super/dataset.zarr")
    ap.add_argument("--ckpt", default="last", choices=["best", "last"])
    ap.add_argument("-k", type=int, default=6, help="k x k blocks of the train region")
    a = ap.parse_args()
    main(a.archs, a.zarr, a.ckpt, a.k)
