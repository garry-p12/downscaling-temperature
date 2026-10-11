"""Publication figures for the downscaling paper.

Six figures at journal column widths, vector + raster, captions left to the
manuscript. These are not restyled slide sheets. A slide sheet carries eight
panels and bakes its own caption into the image because it has to stand alone
on a screen; a paper figure carries the fewest panels that make one point,
because the text around it does the rest.

    F1  error budget        what the error is made of
    F2  offset estimator    how the daily bias is recovered, and the ceiling
    F3  validation          one correction, three independent judges
    F4  fine-scale skill    amplitude and coherence against scale
    F5  target swap         each model reproduces its own target
    F6  interventions       everything tried, against the noise floor

EVERY NUMBER ON THESE FIGURES IS COMPUTED HERE OR READ FROM A FILE.
The one exception is outputs/followup_station.json, which carries the Vista
follow-up means with its provenance recorded inside it. Nothing is typed in
from memory: a literal in a figure script is a number no one can re-check.

NOTE FOR A FRESH CLONE: outputs/ is gitignored, so that JSON is not in the
repository. F6 needs it and will raise FileNotFoundError without it; the
numbers come from a station_panel run on Vista (job 1055429) and have to be
retrieved from there. Every other figure rebuilds from the stores and
checkpoints alone.

Usage:
    python scripts/make_paper_figures.py                  # all six
    python scripts/make_paper_figures.py --only F3 F4
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

from evaluation import paperstyle as ps                       # noqa: E402

OUT = REPO / "image_outputs" / "paper"
SUPER = "data_store_sc/super/dataset.zarr"
NOISE = 0.0097          # measured between-seed RMSE floor
KBEST = 6               # the block grid the paper reports


def rmse(x) -> float:
    return float(np.sqrt(np.nanmean(np.asarray(x) ** 2)))


def offset(st, model: str, k: int = KBEST):
    """The estimator exactly as the paper fits it.

    Returns (resid, est, true_off). The ridge sees only train days and only
    the training region's block means; `true_off` is the held-out region's own
    daily mean residual, which is what the estimator is trying to guess and is
    never an input to it.
    """
    from evaluation.fields import predict_all
    from evaluation.offset_correction import ridge_fit_predict, subregion_means

    resid = st.truth() - predict_all(st, model, "last")
    true_off = resid[:, st.m_hold].mean(1)
    X = subregion_means(resid, st.m_train, *st.shape, k=k)
    return resid, ridge_fit_predict(X, true_off, st.i_train), true_off


# --------------------------------------------------------------------------- #
def f1_budget(plt):
    """Variance shares, the static pattern, and the daily offset series."""
    import matplotlib.dates as mdates

    from evaluation.fields import Store, decompose, predict_all

    st = Store(SUPER)
    te, hold = st.i_test, st.m_hold
    task = (st.truth() - st.coarse())[te]
    model = (st.truth() - predict_all(st, "of0_s1337", "last"))[te]
    mu, s_map, _, sh_t = decompose(task, hold)
    _, _, _, sh_m = decompose(model, hold)

    fig = plt.figure(figsize=(ps.WIDE, 2.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.20, 0.80, 1.50], wspace=0.46)

    ax = fig.add_subplot(gs[0])
    rows = [("uniform daily\noffset $\\mu(t)$", "offset"),
            ("static pattern\n$s(x)$", "static"),
            ("space-time\nremainder", "remainder")]
    y = np.arange(3)
    ax.barh(y - .19, [sh_t[k] * 100 for _, k in rows], height=.34,
            color=ps.BLUE, label="task residual")
    ax.barh(y + .19, [sh_m[k] * 100 for _, k in rows], height=.34,
            color=ps.BLUE, alpha=.40, label="model residual")
    for i, (_, k) in enumerate(rows):
        ax.text(sh_t[k] * 100 + 1.5, i - .19, f"{sh_t[k]*100:.1f}",
                va="center", fontsize=6.6)
        ax.text(sh_m[k] * 100 + 1.5, i + .19, f"{sh_m[k]*100:.1f}",
                va="center", fontsize=6.6, color=ps.GREY)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 82)
    ax.set_xlabel("share of variance (%)")
    ax.legend(loc="center right", fontsize=6.2)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ps.panel(ax, "a", dx=-0.56)

    ax = fig.add_subplot(gs[1])
    lim = float(np.nanpercentile(np.abs(s_map[hold]), 98))
    # s(x) is defined only where the budget is quoted, so the full domain would
    # be a grey slab with a small stamp in the middle. Crop to the holdout.
    i0, i1, j0, j1 = st.box
    pad = 3
    sub = s_map[i0 - pad:i1 + pad, j0 - pad:j1 + pad]
    im = ax.imshow(np.ma.masked_invalid(sub), cmap=ps.ramp(ps.DIVERGING),
                   vmin=-lim, vmax=lim)
    ax.add_patch(plt.Rectangle((pad - .5, pad - .5), j1 - j0, i1 - i0,
                               fill=False, ec=ps.ORANGE, lw=0.8))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    cb.ax.tick_params(labelsize=6, length=1.8)
    cb.set_label("$s(x)$  (\u00b0C)", fontsize=6.8)
    ps.panel(ax, "b", dx=-0.12)

    ax = fig.add_subplot(gs[2])
    ax.plot(st.times[te], mu, lw=0.5, color=ps.BLACK)
    ax.axhline(0, lw=0.5, color=ps.GREY)
    ax.set_ylabel("$\\mu(t)$  (\u00b0C)")
    ax.set_xlabel("2023")
    ax.text(.02, .05, f"sd {mu.std():.3f} \u00b0C", transform=ax.transAxes,
            fontsize=6.8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ps.panel(ax, "c", dx=-0.15)

    ps.save(fig, OUT, "F1_error_budget")
    plt.close(fig)
    print(f"      task  offset {sh_t['offset']:.3f} static {sh_t['static']:.3f} "
          f"rem {sh_t['remainder']:.3f}  total var {sh_t['total_var']:.4f}")
    print(f"      model offset {sh_m['offset']:.3f} static {sh_m['static']:.3f} "
          f"rem {sh_m['remainder']:.3f}  total var {sh_m['total_var']:.4f}")


# --------------------------------------------------------------------------- #
def f2_estimator(plt):
    """Estimator scatter, the k sweep, and the RMSE ladder with its ceiling."""
    from evaluation.fields import Store
    from evaluation.offset_correction import ridge_fit_predict, subregion_means

    st = Store(SUPER)
    te, hold = st.i_test, st.m_hold
    resid, est6, true_off = offset(st, "of0_s1337")
    glob = resid[:, st.m_train].mean(1)

    r_model = rmse(resid[te][:, hold])
    r_oracle = rmse((resid[te] - true_off[te, None, None])[:, hold])
    r_glob = rmse((resid[te] - glob[te, None, None])[:, hold])
    frac = lambda r: (r_model - r) / (r_model - r_oracle) * 100   # noqa: E731

    ks, rs = [], []
    for k in (2, 3, 4, 5, 6, 8, 10, 12):
        X = subregion_means(resid, st.m_train, *st.shape, k=k)
        e = ridge_fit_predict(X, true_off, st.i_train)
        ks.append(k)
        rs.append(rmse((resid[te] - e[te, None, None])[:, hold]))

    fig, ax = plt.subplots(1, 3, figsize=(ps.WIDE, 2.05))
    fig.subplots_adjust(wspace=0.42)

    a = ax[0]
    lim = float(np.percentile(np.abs(np.r_[true_off[te], est6[te], glob[te]]), 99.5))
    a.scatter(glob[te], true_off[te], s=1.8, color=ps.GREY, alpha=.45, lw=0,
              label=f"region mean, $r$={np.corrcoef(glob[te], true_off[te])[0,1]:.2f}")
    a.scatter(est6[te], true_off[te], s=1.8, color=ps.BLUE, alpha=.70, lw=0,
              label=f"$k$={KBEST} ridge, $r$={np.corrcoef(est6[te], true_off[te])[0,1]:.2f}")
    a.plot([-lim, lim], [-lim, lim], ls=(0, (3, 2)), lw=0.6, color=ps.BLACK)
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_aspect("equal")
    a.set_xlabel("estimated offset (\u00b0C)")
    a.set_ylabel("true offset (\u00b0C)")
    a.legend(loc="upper left", fontsize=5.9)
    ps.panel(a, "a")

    a = ax[1]
    a.plot(ks, [frac(r) for r in rs], "-o", color=ps.BLUE, ms=3)
    a.axhline(frac(r_glob), ls=(0, (3, 2)), lw=0.7, color=ps.GREY)
    a.text(ks[-1], frac(r_glob) + 2, "one region-wide mean", ha="right",
           fontsize=6.2, color=ps.GREY)
    a.scatter([KBEST], [frac(rs[ks.index(KBEST)])], s=34, facecolor="none",
              edgecolor=ps.ORANGE, lw=1.1, zorder=5)
    a.set_xlabel("$k$  (block grid is $k \\times k$)")
    a.set_ylabel("% of the oracle recovered")
    a.set_ylim(35, 100)
    ps.panel(a, "b")

    a = ax[2]
    items = [("interpolation", rmse((st.truth() - st.coarse())[te][:, hold]), ps.GREY),
             ("model", r_model, ps.BLACK),
             ("+ region mean", r_glob, ps.SKY),
             (f"+ $k$={KBEST} ridge", rs[ks.index(KBEST)], ps.BLUE),
             ("oracle", r_oracle, ps.ORANGE)]
    yy = np.arange(len(items))
    a.barh(yy, [i[1] for i in items], height=.56, color=[i[2] for i in items])
    for i, (_, v, _) in enumerate(items):
        a.text(v + .015, i, f"{v:.3f}", va="center", fontsize=6.6)
    a.set_yticks(yy)
    a.set_yticklabels([i[0] for i in items], fontsize=6.6)
    a.invert_yaxis()
    a.set_xlim(0, max(i[1] for i in items) * 1.22)
    a.set_xlabel("RMSE over the held-out region (\u00b0C)")
    a.spines["left"].set_visible(False)
    a.tick_params(axis="y", length=0)
    ps.panel(a, "c", dx=-0.40)

    ps.save(fig, OUT, "F2_offset_estimator")
    plt.close(fig)
    print(f"      model {r_model:.4f}  k={KBEST} {rs[ks.index(KBEST)]:.4f}  "
          f"oracle {r_oracle:.4f}  -> {frac(rs[ks.index(KBEST)]):.0f}% of oracle")


# --------------------------------------------------------------------------- #
def _station_rmse(st, fields: dict, idx) -> dict:
    """Mean RMSE across the ISD panel for each named field, on given days."""
    import pandas as pd

    from evaluation.station_panel import fetch, load_panel

    keep = np.zeros(len(st.times), bool)
    keep[idx] = True
    acc = {k: [] for k in fields}
    for s in load_panel():
        series = [o for o in (fetch(s["usaf"], s["wban"], y)
                              for y in (2019, 2020, 2021, 2022, 2023))
                  if o is not None and len(o)]
        if not series:
            continue
        obs = pd.concat(series).sort_index()
        obs = obs[~obs.index.duplicated()]
        sel = st.times.isin(obs.index) & keep
        if sel.sum() < 100:
            continue
        o = obs.reindex(st.times[sel]).values
        for k, F in fields.items():
            acc[k].append(rmse(F[sel, s["row"], s["col"]] - o))
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in acc.items()}


def f3_validation(plt):
    """The two reference products judged by thermometers, and one correction
    measured by three independent judges."""
    from evaluation.fields import Store, predict_all

    rows = json.loads((REPO / "outputs/station_panel/products.json").read_text())
    rows.sort(key=lambda r: r["aorc"]["rmse"])
    era = np.array([r["era5_land"]["rmse"] for r in rows])
    aor = np.array([r["aorc"]["rmse"] for r in rows])
    n = len(rows)

    # One model, one offset series, three judges. The offset is fitted against
    # ERA5-Land in the training region only, so the AORC and station columns
    # are genuinely out-of-sample for it.
    # of0_s1337 throughout, the same arm as F1 and F2, so the three figures
    # describe one model rather than three. It lives on the `super` store; a
    # checkpoint scored against another store's channel list fails loudly here
    # and would otherwise be scored against the wrong climatology.
    st = Store(SUPER)
    te, hold = st.i_test, st.m_hold
    _, est, _ = offset(st, "of0_s1337")
    pred = predict_all(st, "of0_s1337", "last")
    raw_abs = st.to_absolute(pred)
    cor_abs = st.to_absolute(pred + est[:, None, None])

    judges = []
    tru = st.truth()
    judges.append(("ERA5-Land\nthe target",
                   rmse((tru - pred)[te][:, hold]),
                   rmse((tru - pred - est[:, None, None])[te][:, hold])))

    sa = Store("data_store_sc/aorc/dataset.zarr")
    m = hold & sa.land
    A = sa.to_absolute(sa.truth())
    judges.append(("AORC\nanother grid",
                   rmse((A - raw_abs)[te][:, m]),
                   rmse((A - cor_abs)[te][:, m])))

    sr = _station_rmse(st, {"raw": raw_abs, "cor": cor_abs}, te)
    judges.append((f"{sr['raw'][2]} stations\nthermometers",
                   sr["raw"][0], sr["cor"][0]))

    fig, ax = plt.subplots(1, 3, figsize=(ps.WIDE, 2.15))
    fig.subplots_adjust(wspace=0.40)

    a = ax[0]
    x = np.arange(n)
    a.vlines(x, aor, era, color=ps.GREY, lw=0.5, alpha=.65, zorder=1)
    a.scatter(x, era, s=5, color=ps.ORANGE, zorder=3, label="ERA5-Land", lw=0)
    a.scatter(x, aor, s=5, color=ps.BLUE, zorder=3, label="AORC", lw=0)
    a.set_xticks([])
    a.set_xlabel(f"{n} ISD stations, sorted")
    a.set_ylabel("RMSE vs thermometer (\u00b0C)")
    a.legend(loc="upper left", fontsize=6.2)
    a.text(.97, .06, f"AORC closer at {int((aor < era).sum())}/{n}",
           transform=a.transAxes, ha="right", fontsize=6.4, color=ps.BLUE)
    ps.panel(a, "a")

    a = ax[1]
    bins = np.linspace(min(era.min(), aor.min()) * .95, era.max() * 1.03, 20)
    a.hist(era, bins=bins, color=ps.ORANGE, alpha=.60,
           label=f"ERA5-Land, mean {era.mean():.3f}")
    a.hist(aor, bins=bins, color=ps.BLUE, alpha=.60,
           label=f"AORC, mean {aor.mean():.3f}")
    for v, c in ((era.mean(), ps.ORANGE), (aor.mean(), ps.BLUE)):
        a.axvline(v, color=c, lw=0.9, ls=(0, (3, 2)))
    a.set_xlabel("RMSE (\u00b0C)")
    a.set_ylabel("stations")
    a.legend(loc="upper right", fontsize=6.0)
    ps.panel(a, "b")

    a = ax[2]
    xx = np.arange(len(judges))
    a.bar(xx - .18, [j[1] for j in judges], width=.34, color=ps.GREY,
          label="model")
    a.bar(xx + .18, [j[2] for j in judges], width=.34, color=ps.BLUE,
          label="+ offset correction")
    top = max(max(j[1], j[2]) for j in judges)
    for i, j in enumerate(judges):
        a.text(i + .18, j[2] + top * .02, f"{(j[2]-j[1])/j[1]*100:+.0f}%",
               ha="center", fontsize=6.0, color=ps.BLUE)
    a.set_xticks(xx)
    a.set_xticklabels([j[0] for j in judges], fontsize=6.2)
    a.set_ylabel("RMSE (\u00b0C)")
    a.set_ylim(0, top * 1.30)
    a.legend(loc="upper left", fontsize=6.2)
    ps.panel(a, "c")

    ps.save(fig, OUT, "F3_validation")
    plt.close(fig)
    for lab, r0, r1 in judges:
        print(f"      {lab.replace(chr(10), ' '):<28} {r0:.4f} -> {r1:.4f}  "
              f"({(r1-r0)/r0*100:+.1f}%)")


# --------------------------------------------------------------------------- #
def f4_resolution(plt):
    """Amplitude and coherence against wavelength, replacing a single number."""
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    from evaluation.fields import Store, predict_all
    from evaluation.spectral import (AMP_FLOOR, coherence_resolution,
                                     effective_resolution, radial_spectrum)

    sa = Store("data_store_sc/aorc/dataset.zarr")
    sd_ = Store("data_store_sc/dyn/dataset.zarr")
    te = sa.i_test
    i0, i1, j0, j1 = sa.box
    crop = lambda f: (f[te][:, i0:i1, j0:j1]                      # noqa: E731
                      - f[te][:, i0:i1, j0:j1].mean((1, 2))[:, None, None])
    A = crop(sa.to_absolute(sa.truth()))

    series = [("ERA5-Land (the product)", sd_, None, ps.GREY, (0, (4, 2))),
              ("ERA5-Land-trained model", sd_, "dyn", ps.BLUE, "-"),
              ("AORC-trained model", sa, "aorc", ps.ORANGE, "-")]
    fig, ax = plt.subplots(1, 2, figsize=(ps.MID, 2.10))
    fig.subplots_adjust(wspace=0.34)
    summary = []

    for lab, st, pref, col, ls in series:
        curves = []
        for sd in ([None] if pref is None else [1337, 7, 42]):
            if pref is None:
                F = st.to_absolute(st.truth())
            else:
                if not (REPO / "checkpoints" / f"{pref}_s{sd}" / "last.pt").exists():
                    continue
                F = st.to_absolute(predict_all(st, f"{pref}_s{sd}", "last"))
            curves.append(radial_spectrum(crop(F), A))
        if not curves:
            continue
        w = [r["wavelength_km"] for r in curves[0]]
        amp = np.mean([[r["amplitude_ratio"] for r in c] for c in curves], 0)
        coh = np.mean([[r["coherence"] for r in c] for c in curves], 0)
        ax[0].plot(w, amp, color=col, ls=ls, lw=1.1, label=lab)
        ax[1].plot(w, coh, color=col, ls=ls, lw=1.1, label=lab)
        mean = [curves[0][i] | {"amplitude_ratio": amp[i], "coherence": coh[i]}
                for i in range(len(w))]
        summary.append((lab, effective_resolution(mean), coherence_resolution(mean),
                        float(np.mean([a for a, ww in zip(amp, w) if ww <= 60])),
                        float(np.mean([c for c, ww in zip(coh, w) if ww <= 60]))))

    ticks = [400, 200, 100, 50, 25]
    for a, lvl, name in ((ax[0], AMP_FLOOR, "amplitude ratio"),
                         (ax[1], 0.5, "coherence with AORC")):
        a.set_xscale("log")
        a.invert_xaxis()
        a.xaxis.set_major_locator(FixedLocator(ticks))
        a.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        a.xaxis.set_minor_locator(NullLocator())
        a.axhline(lvl, ls=(0, (2, 2)), lw=0.6, color=ps.BLACK)
        a.text(ticks[0], lvl, f" {lvl:.2f} threshold", fontsize=5.8,
               va="bottom", ha="left", color=ps.GREY)
        a.set_xlabel("wavelength (km)")
        a.set_ylabel(name)
    ax[0].legend(loc="lower left", fontsize=5.9)
    ps.panel(ax[0], "a")
    ps.panel(ax[1], "b")
    ps.save(fig, OUT, "F4_fine_scale_skill")
    plt.close(fig)
    print(f"      {'arm':<26}{'eff res':>9}{'coh<0.5':>9}{'amp<60':>9}{'coh<60':>9}")
    for lab, e, c, am, ch in summary:
        f = lambda v: "none" if v is None or not np.isfinite(v) else f"{v:.0f}"  # noqa: E731
        print(f"      {lab:<26}{f(e):>9}{f(c):>9}{am:>9.3f}{ch:>9.3f}")


# --------------------------------------------------------------------------- #
def f5_target(plt):
    """Fine-scale amplitude of each truth product and of each model's output."""
    from evaluation.fields import Store, highpass, predict_all

    sa = Store("data_store_sc/aorc/dataset.zarr")
    sd_ = Store("data_store_sc/dyn/dataset.zarr")
    te, land, hold = sa.i_test, sa.land, sa.m_hold
    # 60 km half-amplitude cut-off, expressed as a Gaussian sigma in cells.
    sigma = float(np.sqrt(np.log(2)) / (np.pi * np.sqrt(2)) * 60.0 / 10.0)
    rim = np.zeros_like(land)
    rim[4:-4, 4:-4] = True
    fine = land & rim

    def filled(F):
        """NaN off-land would spread through the Gaussian and manufacture a
        halo of fake detail; each day is filled with its own land mean."""
        g = F[te].copy()
        return np.where(np.isfinite(g), g,
                        np.nanmean(np.where(land, g, np.nan), axis=(1, 2))[:, None, None])

    panels = [("AORC truth", sa.to_absolute(sa.truth())),
              ("ERA5-Land truth", sd_.to_absolute(sd_.truth())),
              ("ERA5-Land-trained", sd_.to_absolute(predict_all(sd_, "dyn_s1337", "last"))),
              ("AORC-trained", sa.to_absolute(predict_all(sa, "aorc_s1337", "last")))]
    sds = [highpass(filled(F), sigma).std(0) for _, F in panels]
    vmax = float(np.percentile(sds[0][fine], 95))

    fig, ax = plt.subplots(1, 4, figsize=(ps.WIDE, 2.00))
    fig.subplots_adjust(wspace=0.08)
    i0, i1, j0, j1 = sa.box
    for k, (a, (lab, _), sdv) in enumerate(zip(ax, panels, sds)):
        im = a.imshow(np.ma.masked_invalid(np.where(fine, sdv, np.nan)),
                      cmap=ps.ramp(ps.SEQUENTIAL), vmin=0, vmax=vmax)
        a.add_patch(plt.Rectangle((j0, i0), j1 - j0, i1 - i0, fill=False,
                                  ec=ps.ORANGE, lw=0.8))
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True)
        a.set_title(f"{lab}\n{sdv[hold].mean():.4f} \u00b0C", fontsize=6.8,
                    loc="center", pad=3)
        ps.panel(a, "abcd"[k], dx=-0.05, dy=1.30)
    cb = fig.colorbar(im, ax=ax, fraction=.018, pad=.012)
    cb.ax.tick_params(labelsize=6, length=1.8)
    cb.set_label("sd of detail below 60 km (\u00b0C)", fontsize=6.8)
    ps.save(fig, OUT, "F5_target_swap")
    plt.close(fig)
    for (lab, _), sdv in zip(panels, sds):
        print(f"      {lab:<22}{sdv[hold].mean():.4f}")


# --------------------------------------------------------------------------- #
def f6_interventions(plt):
    """Every intervention, as a change in RMSE set against the noise floor."""
    from evaluation.fields import Store

    d = json.loads((REPO / "outputs/followup_station.json").read_text())
    arms = d["arms"]
    base = next(a for a in arms if a["arm"] == d["_provenance"]["baseline"])
    se = lambda a: np.sqrt(a["sd"] ** 2 / a["n"] + base["sd"] ** 2 / base["n"])  # noqa: E731
    items = sorted([(a["label"], a["mean"] - base["mean"], se(a))
                    for a in arms if a is not base], key=lambda t: t[1])

    # The offset correction's own effect is recomputed here rather than quoted,
    # so the two panels of this figure cannot drift apart.
    st = Store(SUPER)
    te, hold = st.i_test, st.m_hold
    resid, est, _ = offset(st, "of0_s1337")
    r0 = rmse(resid[te][:, hold])
    r1 = rmse((resid[te] - est[te, None, None])[:, hold])
    worked = [("offset correction", r1 - r0, 0.0)] + \
             [(w["label"], w["delta"], w["se"]) for w in d["worked"]]

    fig, ax = plt.subplots(1, 2, figsize=(ps.WIDE, 2.00))
    fig.subplots_adjust(wspace=0.50)

    a = ax[0]
    y = np.arange(len(items))
    a.barh(y, [i[1] for i in items], xerr=[i[2] for i in items], height=.55,
           color=[ps.GREY if abs(i[1]) < NOISE else ps.ORANGE for i in items],
           error_kw=dict(lw=0.7, capsize=2))
    a.axvspan(-NOISE, NOISE, color=ps.BLACK, alpha=.08, lw=0)
    a.axvline(0, lw=0.6, color=ps.BLACK)
    a.text(.98, .06, "shading: between-seed noise floor\norange: outside it",
           transform=a.transAxes, ha="right", va="bottom", fontsize=5.8,
           color=ps.GREY)
    a.set_yticks(y)
    a.set_yticklabels([i[0] for i in items], fontsize=6.4)
    a.invert_yaxis()
    a.set_xlabel("$\\Delta$ RMSE vs DeepSD (\u00b0C)\n"
                 f"{len(json.loads((REPO/'outputs/station_panel/products.json').read_text()))}"
                 " stations, test split")
    a.spines["left"].set_visible(False)
    a.tick_params(axis="y", length=0)
    ps.panel(a, "a", dx=-0.42)

    a = ax[1]
    y = np.arange(len(worked))
    a.barh(y, [w[1] for w in worked], xerr=[w[2] for w in worked], height=.45,
           color=ps.BLUE, error_kw=dict(lw=0.7, capsize=2))
    a.axvline(0, lw=0.6, color=ps.BLACK)
    for i, w in enumerate(worked):
        a.text(w[1] - .012, i, f"{w[1]:.3f}", va="center", ha="right", fontsize=6.4)
    a.set_yticks(y)
    a.set_yticklabels([w[0] for w in worked], fontsize=6.4)
    a.invert_yaxis()
    a.set_xlim(min(w[1] for w in worked) * 1.35, abs(min(w[1] for w in worked)) * .18)
    a.set_xlabel("$\\Delta$ RMSE (\u00b0C)\nthe two that moved the needle")
    a.spines["left"].set_visible(False)
    a.tick_params(axis="y", length=0)
    ps.panel(a, "b", dx=-0.42)

    ps.save(fig, OUT, "F6_interventions")
    plt.close(fig)
    for lab, dl, s in items:
        print(f"      {lab:<28}{dl:+.4f} +/- {s:.4f}"
              f"{'   < floor' if abs(dl) < NOISE else ''}")
    print(f"      offset correction           {r1-r0:+.4f}  ({r0:.4f} -> {r1:.4f})")


# --------------------------------------------------------------------------- #
def f7_capacity(plt):
    """Ten architectures, five seeds each, against parameter count."""
    rep = json.loads((REPO / "image_outputs/multiseed/holdout_report.json").read_text())
    m = rep["methods"]
    by_arch: dict[str, list] = {}
    for k, v in m.items():
        if "_s" not in k:
            continue
        arch = k.rsplit("_s", 1)[0]
        by_arch.setdefault(arch, []).append((v["rmse"], v.get("params_M", np.nan)))
    base = m["interpolated_POWER"]["rmse"]

    fig, ax = plt.subplots(1, 2, figsize=(ps.MID, 2.10),
                           gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(wspace=0.36)

    a = ax[0]
    order = sorted(by_arch.items(), key=lambda t: t[1][0][1])
    # Names go in a legend, not beside the markers: Restormer (23 M) and Swin
    # (41 M) are a thumb-width apart on a log axis and within 0.003 degC of
    # each other, so any in-place label overprints its neighbour.
    for i, (arch, rows) in enumerate(order):
        r = np.array([x[0] for x in rows])
        pm = rows[0][1]
        a.scatter([pm] * len(r), r, s=5, color=ps.GREY, lw=0, zorder=2)
        a.scatter([pm], [r.mean()], s=20, color=ps.CYCLE[i % len(ps.CYCLE)],
                  zorder=3, label=arch)
    a.axhline(base, ls=(0, (3, 2)), lw=0.7, color=ps.ORANGE)
    a.text(a.get_xlim()[1], base, "interpolation ", ha="right", va="bottom",
           fontsize=5.8, color=ps.ORANGE)
    a.set_xscale("log")
    pmin = min(v[0][1] for v in by_arch.values())
    pmax = max(v[0][1] for v in by_arch.values())
    a.set_xlim(pmin / 3.0, pmax * 3.0)
    lo = min(x[0] for v in by_arch.values() for x in v)
    a.set_ylim(lo - 0.010, base + 0.075)      # room for the legend
    a.legend(loc="upper left", fontsize=5.6, ncol=3, columnspacing=.8,
             handletextpad=.25, borderaxespad=.2)
    a.set_xlabel("parameters (millions)")
    a.set_ylabel("held-out RMSE (\u00b0C)")
    ps.panel(a, "a")

    # The comparison that matters is not arch-to-arch, it is arch-to-seed.
    # Reported three ways rather than one, because the all-six spread is
    # larger than the seed spread and that is entirely one architecture:
    # dropping ConvNeXt, which is the only arm measurably worse than the rest,
    # halves it. Showing only the flattering number would be cherry-picking;
    # showing only the unflattering one would hide where it comes from.
    a = ax[1]
    worst = max(by_arch, key=lambda k: np.mean([x[0] for x in by_arch[k]]))
    means = np.array([np.mean([x[0] for x in v]) for v in by_arch.values()])
    rest = np.array([np.mean([x[0] for x in v]) for k, v in by_arch.items()
                     if k != worst])
    within = np.concatenate([np.array([x[0] for x in v])
                             - np.mean([x[0] for x in v])
                             for v in by_arch.values()])
    vals = [means.std(ddof=1), rest.std(ddof=1), within.std(ddof=1)]
    a.bar([0, 1, 2], vals, width=.55, color=[ps.BLUE, ps.BLUE, ps.GREY])
    for i, v in enumerate(vals):
        a.text(i, v + max(vals) * .02, f"{v:.4f}", ha="center", fontsize=6.0)
    a.set_xticks([0, 1, 2])
    a.set_xticklabels([f"between all\n{len(by_arch)} architectures",
                       f"without\n{worst}", "between seeds\nof one arch"],
                      fontsize=5.8)
    a.set_ylabel("sd of RMSE (\u00b0C)")
    a.set_ylim(0, max(vals) * 1.28)
    ps.panel(a, "b")

    ps.save(fig, OUT, "F7_capacity")
    plt.close(fig)
    print(f"      {len(by_arch)} architectures, {len(next(iter(by_arch.values())))} "
          f"seeds each: between-arch sd {vals[0]:.4f}, without {worst} "
          f"{vals[1]:.4f}, between-seed {vals[2]:.4f}")


# --------------------------------------------------------------------------- #
def f8_improvement(plt):
    """Where the offset correction helps, cell by cell, in lat/lon."""
    from evaluation.fields import Store

    st = Store(SUPER)
    te, hold = st.i_test, st.m_hold
    resid, est, _ = offset(st, "of0_s1337")
    r0 = resid[te]
    r1 = resid[te] - est[te, None, None]
    pc0 = np.sqrt((r0 ** 2).mean(0))
    pc1 = np.sqrt((r1 ** 2).mean(0))

    i0, i1, j0, j1 = st.box
    pad = 8
    sl = (slice(max(i0 - pad, 0), i1 + pad), slice(max(j0 - pad, 0), j1 + pad))
    LAT, LON = st.LAT[sl], st.LON[sl]
    show = lambda f: np.ma.masked_invalid(np.where(st.land, f, np.nan)[sl])  # noqa: E731
    d = pc1 - pc0

    fig, ax = plt.subplots(1, 3, figsize=(ps.WIDE, 2.30))
    fig.subplots_adjust(wspace=0.46)
    # A floor of zero puts every cell in the top third of the ramp and the two
    # maps look identical; the range that separates them is the range they
    # actually occupy.
    vmin = float(np.nanpercentile(pc1[hold], 1))
    vmax = float(np.nanpercentile(pc0[hold], 99))
    dlim = float(np.nanpercentile(np.abs(d[hold]), 99))

    for k, (a, f, cm, vmin, vmx, lab, title) in enumerate([
            (ax[0], pc0, ps.ramp(ps.SEQUENTIAL), vmin, vmax, "RMSE (\u00b0C)",
             f"model, box mean {np.sqrt((r0[:, hold]**2).mean()):.3f}\u00b0C"),
            (ax[1], pc1, ps.ramp(ps.SEQUENTIAL), vmin, vmax, "RMSE (\u00b0C)",
             f"corrected, box mean {np.sqrt((r1[:, hold]**2).mean()):.3f}\u00b0C"),
            (ax[2], d, ps.ramp(ps.DIVERGING), -dlim, dlim,
             "\u0394 RMSE (\u00b0C)",
             f"change, improves {100*(d[hold] < 0).mean():.0f}% of cells")]):
        im = a.imshow(show(f), cmap=cm, vmin=vmin, vmax=vmx)
        ps.graticule(a, LAT, LON)
        a.add_patch(plt.Rectangle((j0 - sl[1].start - .5, i0 - sl[0].start - .5),
                                  j1 - j0, i1 - i0, fill=False, ec=ps.ORANGE,
                                  lw=0.9))
        a.set_title(title, fontsize=6.6, loc="center", pad=3)
        cb = fig.colorbar(im, ax=a, fraction=.046, pad=.03)
        cb.ax.tick_params(labelsize=5.6, length=1.6)
        cb.set_label(lab, fontsize=6.2)
        ps.panel(a, "abc"[k], dx=-0.14, dy=1.12)

    ps.save(fig, OUT, "F8_improvement_map")
    plt.close(fig)
    print(f"      box {np.sqrt((r0[:, hold]**2).mean()):.4f} -> "
          f"{np.sqrt((r1[:, hold]**2).mean()):.4f}, improves "
          f"{100*(d[hold] < 0).mean():.1f}% of held-out cells")


# --------------------------------------------------------------------------- #
def f9_target_maps(plt):
    """Target swap, cell by cell: accuracy and fine-scale placement."""
    from evaluation.fields import Store, cellwise_corr, highpass, predict_all

    sa = Store("data_store_sc/aorc/dataset.zarr")
    sd_ = Store("data_store_sc/dyn/dataset.zarr")
    te, land, hold = sa.i_test, sa.land & sd_.land, sa.m_hold & sd_.land
    A = sa.to_absolute(sa.truth())
    p_e = sd_.to_absolute(predict_all(sd_, "dyn_s1337", "last"))
    p_a = sa.to_absolute(predict_all(sa, "aorc_s1337", "last"))

    pc = {k: np.sqrt(((v[te] - A[te]) ** 2).mean(0)) for k, v in
          (("era", p_e), ("aorc", p_a))}
    sigma = float(np.sqrt(np.log(2)) / (np.pi * np.sqrt(2)) * 60.0 / 10.0)
    rim = np.zeros_like(land)
    rim[4:-4, 4:-4] = True
    fine = land & rim

    def filled(F):
        g = F[te].copy()
        return np.where(np.isfinite(g), g,
                        np.nanmean(np.where(land, g, np.nan), axis=(1, 2))[:, None, None])

    hp_A = highpass(filled(A), sigma)
    coh_e = cellwise_corr(highpass(filled(p_e), sigma), hp_A, fine)
    coh_a = cellwise_corr(highpass(filled(p_a), sigma), hp_A, fine)

    i0, i1, j0, j1 = sa.box
    pad = 8
    sl = (slice(max(i0 - pad, 0), i1 + pad), slice(max(j0 - pad, 0), j1 + pad))
    LAT, LON = sa.LAT[sl], sa.LON[sl]
    show = lambda f, m: np.ma.masked_invalid(np.where(m, f, np.nan)[sl])  # noqa: E731

    vmin = float(np.nanpercentile(pc["aorc"][hold], 1))
    vmax = float(np.nanpercentile(pc["era"][hold], 99))
    dcoh = coh_a - coh_e
    clim = float(np.nanpercentile(np.abs(dcoh[fine]), 99))

    fig, ax = plt.subplots(1, 3, figsize=(ps.WIDE, 2.30))
    fig.subplots_adjust(wspace=0.46)
    spec = [(pc["era"], land, ps.ramp(ps.SEQUENTIAL), vmin, vmax, "RMSE (\u00b0C)",
             f"ERA5-Land-trained, box {np.sqrt((( p_e[te]-A[te])**2)[:, hold].mean()):.3f}\u00b0C"),
            (pc["aorc"], land, ps.ramp(ps.SEQUENTIAL), vmin, vmax, "RMSE (\u00b0C)",
             f"AORC-trained, box {np.sqrt((((p_a[te]-A[te])**2))[:, hold].mean()):.3f}\u00b0C"),
            (dcoh, fine, ps.ramp(ps.DIVERGING[::-1]), -clim, clim,
             "\u0394 correlation",
             f"fine-scale placement, better in {100*(dcoh[hold] > 0).mean():.0f}% of cells")]
    for k, (a, (f, m, cm, vmin, vmx, lab, title)) in enumerate(zip(ax, spec)):
        im = a.imshow(show(f, m), cmap=cm, vmin=vmin, vmax=vmx)
        ps.graticule(a, LAT, LON)
        a.add_patch(plt.Rectangle((j0 - sl[1].start - .5, i0 - sl[0].start - .5),
                                  j1 - j0, i1 - i0, fill=False, ec=ps.ORANGE,
                                  lw=0.9))
        a.set_title(title, fontsize=6.4, loc="center", pad=3)
        cb = fig.colorbar(im, ax=a, fraction=.046, pad=.03)
        cb.ax.tick_params(labelsize=5.6, length=1.6)
        cb.set_label(lab, fontsize=6.2)
        ps.panel(a, "abc"[k], dx=-0.14, dy=1.12)

    ps.save(fig, OUT, "F9_target_maps")
    plt.close(fig)
    print(f"      box RMSE vs AORC  ERA5-trained "
          f"{np.sqrt(((p_e[te]-A[te])**2)[:, hold].mean()):.4f}  AORC-trained "
          f"{np.sqrt(((p_a[te]-A[te])**2)[:, hold].mean()):.4f}")
    print(f"      box coherence     {np.nanmean(coh_e[hold]):.4f} -> "
          f"{np.nanmean(coh_a[hold]):.4f}")


FIGS = {"F1": f1_budget, "F2": f2_estimator, "F3": f3_validation,
        "F4": f4_resolution, "F5": f5_target, "F6": f6_interventions,
        "F7": f7_capacity, "F8": f8_improvement,
        "F9": f9_target_maps}


def main(only, sans):
    ps.use(serif=not sans)
    import matplotlib.pyplot as plt
    for key in (only or list(FIGS)):
        print(f"{key} ...")
        FIGS[key](plt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=list(FIGS), default=None)
    ap.add_argument("--sans", action="store_true", help="for sans-serif journals")
    a = ap.parse_args()
    main(a.only, a.sans)
