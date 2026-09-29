"""Where the error actually lives — the 5.4 error budget drawn as maps.

The budget is normally quoted as three numbers (50.8% uniform daily offset,
2.8% static spatial pattern, 46.4% space-time remainder). Three numbers do not
show that the 2.8% term is the ONLY one a map of terrain or land cover can
address, and they do not show WHERE on the ground each term bites.

WHAT IS DECOMPOSED. The budget is taken on the TASK residual — truth minus the
interpolated POWER input, i.e. the signal a downscaler is asked to produce —
not on any model's error. That is what makes it a statement about the problem
rather than about one architecture, and it is why it predicts the outcome of
experiments that had not been run when it was measured. The trained model's own
residual is decomposed alongside it in the bar panel, which is where the point
lands: the model removes most of the static term and leaves the offset term
essentially where it found it.

    row 1   task residual / static pattern / remainder / share explained by mu(t)
    row 2   mu(t) through the test year / both budgets / model error /
            what the model actually removed, per cell

The decomposition is orthogonal by construction: mu(t) is removed before s(x) is
measured and both are removed before the remainder, so the shares sum to 1.

Two regions are reported and they are NOT interchangeable. The bars quote the
held-out box, which is the paper's number. The maps decompose over the whole
domain, because a map of the box alone is 37x35 cells and shows no geography.
The two differ a lot and the difference is the point of 5.7: the offset is
close to uniform INSIDE the box (50.8%) but not across the whole domain
(20.3%), which is exactly why a k=6 grid of block means beats one global mean.

Usage:
    python scripts/make_budget_panels.py --archs v2_landcov_s1337 of0_s1337
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.fields import Store, decompose, predict_all  # noqa: E402
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

FIGDIR = REPO / "image_outputs" / "slides"
TITLES = {"v2_landcov": "V2 land-cover (project default)",
          "of0": "Offset-split loss (of0)",
          "v0_base": "V0 baseline",
          "lin": "Linear probe (affine control)",
          "dyn": "Dynamic covariates"}


def main(archs, zarr, ckpt):
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
    truth, coarse = st.truth(), st.coarse()
    task = (truth - coarse)[te]                    # the signal to be produced
    land, box = st.land, st.box
    nan = lambda f: np.where(land, f, np.nan)      # noqa: E731

    # Paper's budget: task residual over the held-out box.
    mu_h, s_h, eps_h, sh_h = decompose(task, st.m_hold)
    # Map budget: same decomposition domain-wide, so the panels have geography.
    mu_d, s_d, eps_d, sh_d = decompose(task, land)

    rmse_task = nan(np.sqrt((task ** 2).mean(0)))
    sd_eps = nan(eps_d.std(0))
    v_tot = np.var(task, axis=0)
    v_after = np.var(task - mu_d[:, None, None], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = nan(1.0 - v_after / v_tot)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for arch in archs:
        model_r = (truth - predict_all(st, arch, ckpt))[te]
        _, _, _, sh_m = decompose(model_r, st.m_hold)
        rmse_model = nan(np.sqrt((model_r ** 2).mean(0)))
        fixed = rmse_task - rmse_model             # degC of RMSE the model removed

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

        vtask = float(np.nanpercentile(rmse_task, 99))
        slim = float(np.nanpercentile(np.abs(s_d), 99))
        show(ax[0, 0], rmse_task, "The task: truth - POWER input", seq,
             f"per-cell RMSE over {len(te)} test days", vmin=0, vmax=vtask)
        show(ax[0, 1], nan(s_d), "Static spatial pattern  s(x)", div,
             f"always-wrong-here term — box {sh_h['static'] * 100:.1f}%, "
             f"domain {sh_d['static'] * 100:.1f}%",
             norm=TwoSlopeNorm(0, -slim, slim))
        show(ax[0, 2], sd_eps, "Space-time remainder  sd of eps(x,t)", seq,
             f"box {sh_h['remainder'] * 100:.1f}%, domain "
             f"{sh_d['remainder'] * 100:.1f}%", vmin=0,
             vmax=float(np.nanpercentile(sd_eps, 99)))
        show(ax[0, 3], share, "Variance removed by ONE number per day", seq,
             f"mu(t) alone — box {sh_h['offset'] * 100:.1f}%, domain "
             f"{sh_d['offset'] * 100:.1f}%", label="fraction", vmin=0, vmax=1)

        # --- mu(t) through the test year -----------------------------------
        a = ax[1, 0]
        tt = st.times[te]
        a.plot(tt, mu_h, lw=.9, color="#2a78d6", label="held-out box")
        a.plot(tt, mu_d, lw=.9, color=ORANGE, alpha=.7, label="whole domain")
        a.axhline(0, color=GRAT, lw=.6)
        a.set_title("The daily offset mu(t)", fontsize=13,
                    fontweight="semibold", pad=10)
        a.text(.5, 1.004, f"box sd {mu_h.std():.3f} degC;  r(box, domain) = "
               f"{np.corrcoef(mu_h, mu_d)[0, 1]:.2f} — close, not identical",
               transform=a.transAxes, ha="center", va="bottom",
               fontsize=7.6, color=INK3)
        a.set_ylabel("degC", fontsize=8.5, color=INK2)
        a.legend(fontsize=7.5, framealpha=.9)
        a.tick_params(labelsize=7.5, color=GRAT, labelcolor=INK3)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for lb in a.get_xticklabels():
            lb.set_rotation(20); lb.set_ha("right")

        # --- the budget, before and after the model --------------------------
        a = ax[1, 1]
        rows = [("uniform daily\noffset  mu(t)", "offset", "#2a78d6"),
                ("static spatial\npattern  s(x)", "static", "#9ec5f4"),
                ("space-time\nremainder", "remainder", "#c9c6bf")]
        y = np.arange(len(rows))
        a.barh(y - .19, [sh_h[k] * 100 for _, k, _ in rows], height=.34,
               color=[c for _, _, c in rows], label="task residual (the problem)")
        a.barh(y + .19, [sh_m[k] * 100 for _, k, _ in rows], height=.34,
               color=[c for _, _, c in rows], alpha=.42, hatch="///",
               edgecolor=SURF, label="model residual (what is left)")
        for i, (_, k, _) in enumerate(rows):
            a.text(sh_h[k] * 100 + 1.2, i - .19, f"{sh_h[k] * 100:.1f}%", va="center",
                   fontsize=10, fontweight="semibold", color=INK)
            a.text(sh_m[k] * 100 + 1.2, i + .19, f"{sh_m[k] * 100:.1f}%", va="center",
                   fontsize=8.5, color=INK3)
        a.set_yticks(y); a.set_yticklabels([r[0] for r in rows], fontsize=8.5)
        a.invert_yaxis(); a.set_xlim(0, 78)
        a.set_xlabel("share of residual variance over the box (%)",
                     fontsize=8.5, color=INK2)
        a.set_title("The model leaves the offset\nexactly where it found it",
                    fontsize=13, fontweight="semibold", pad=16)
        a.legend(fontsize=7.2, loc="center right", framealpha=.95)
        a.tick_params(labelsize=8, color=GRAT, labelcolor=INK2)
        for sp in ("top", "right", "left"):
            a.spines[sp].set_visible(False)

        # --- what the model achieved, as maps --------------------------------
        show(ax[1, 2], rmse_model, "Model error after training", seq,
             "same colour scale as the first panel", vmin=0, vmax=vtask)
        flim = float(np.nanpercentile(np.abs(fixed), 99))
        show(ax[1, 3], nan(fixed), "What the model removed", div,
             f"task RMSE - model RMSE; box mean "
             f"{np.nanmean(np.where(st.m_hold, fixed, np.nan)):+.3f} degC",
             norm=TwoSlopeNorm(0, -flim, flim))

        base = arch.rsplit("_s", 1)[0]
        top = header(
            fig,
            f"Error budget — the problem, and what {TITLES.get(base, base)} does to it"
            f"   ({arch}, {ckpt}.pt)",
            f"Truth minus interpolated POWER, {len(te)} test days: one number per day "
            f"+ a fixed map + the rest. Only the fixed map is reachable by terrain or "
            f"land cover. Bars = the box, maps = the domain.")
        fig.tight_layout(rect=(0, 0, 1, top), h_pad=2.6)
        out = FIGDIR / f"budget_{base}.png"
        fig.savefig(out, dpi=155, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out.name}   TASK box {sh_h['offset'] * 100:.1f}/"
              f"{sh_h['static'] * 100:.1f}/{sh_h['remainder'] * 100:.1f}"
              f"   domain {sh_d['offset'] * 100:.1f}/{sh_d['static'] * 100:.1f}/"
              f"{sh_d['remainder'] * 100:.1f}"
              f"   |   MODEL box {sh_m['offset'] * 100:.1f}/{sh_m['static'] * 100:.1f}/"
              f"{sh_m['remainder'] * 100:.1f}"
              f"   |   box RMSE {np.nanmean(np.where(st.m_hold, rmse_task, np.nan)):.3f}"
              f" -> {np.nanmean(np.where(st.m_hold, rmse_model, np.nan)):.3f} degC")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=["v2_landcov_s1337", "of0_s1337"])
    ap.add_argument("--zarr", default="data_store_sc/super/dataset.zarr")
    ap.add_argument("--ckpt", default="last", choices=["best", "last"])
    a = ap.parse_args()
    main(a.archs, a.zarr, a.ckpt)
