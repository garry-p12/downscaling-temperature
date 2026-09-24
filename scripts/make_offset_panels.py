"""Inference sheet for the offset correction (README 5.7).

The correction is not a model, so it needs a different figure from
make_inference_panels.py. What matters is (a) the error map before and after on
ONE shared colour scale, and (b) the mechanism that makes it work at all — the
offset measured over the training region tracking the offset in the held-out
region, on days where no holdout truth was used.

    row 1   coarse input / truth / model output / offset-corrected output
    row 2   model error / corrected error / offset scatter / RMSE ladder

The correction subtracts a single number per day, so the two error maps differ
by a constant. That is exactly the point: if half the residual really is a
uniform daily bias (5.4), removing one number per day should visibly flatten
the map, and the ladder shows how much of the oracle that captures.

Usage:
    python scripts/make_offset_panels.py --archs of0_s1337 v2_landcov_s1337
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common import load_config                                  # noqa: E402
from common.grid import build_target_grid                       # noqa: E402
from common.normalize import Normalizer                         # noqa: E402
from data.build_dataset import NO_NORMALIZE                     # noqa: E402
from evaluation.mapping import (GRAT, INK, INK2, INK3, ORANGE,  # noqa: E402
                                SURF, graticule, holdout_box, palettes)
from models.model import load_checkpoint                        # noqa: E402
from training.dataset import holdout_bounds                     # noqa: E402

FIGDIR = REPO / "image_outputs" / "slides"
TITLES = {"of0": "Offset-split loss (of0) + spatial offset correction",
          "of1": "Offset-split loss (of1) + spatial offset correction",
          "v2_landcov": "V2 land-cover (project default) + spatial offset correction",
          "v0_base": "V0 baseline + spatial offset correction",
          "lin": "Linear probe + spatial offset correction"}


def main(archs, day, zarr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    seq, div = palettes()
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                         "savefig.facecolor": SURF, "text.color": INK,
                         "font.size": 10})

    cfg = load_config("data")
    ds = xr.open_zarr(zarr, consolidated=True)
    nz = Normalizer.load(str(Path(zarr).parent / "norm_stats.json"))
    stored = ds["channel_in"].values.tolist()
    tg = build_target_grid(cfg["domain"], cfg["grid"])
    LAT, LON = tg.latlon()
    i0, i1, j0, j1 = box = holdout_bounds(ds, cfg["holdout"])

    split = ds["split"].values
    te = np.where(split == "test")[0]
    land = ds["input"].values[0, stored.index("land_mask")] > 0.5
    hold = np.zeros_like(land); hold[i0:i1, j0:j1] = True
    m_hold, m_train = hold & land, (~hold) & land
    times = pd.DatetimeIndex(ds["time"].values)
    day_is_default = day is None
    if not day_is_default:
        kday = int(np.argmin(np.abs(times - pd.Timestamp(day))))
        day = times[kday].strftime("%Y-%m-%d")
    doy = times.dayofyear.values - 1
    clim = ds["clim_tmp"].values
    clim_c = ds["clim_coarse_tmp"].values

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for arch in archs:
        ck = REPO / "checkpoints" / arch / "last.pt"
        if not ck.exists():
            print(f"  {arch}: no last.pt, skipping"); continue
        model, mcfg = load_checkpoint(ck, "cpu", load_config("model"))
        names = list(mcfg.get("use_channels") or stored)
        idx = [stored.index(n) for n in names]

        def predict(k):
            x = ds["input"].isel(time=int(k)).values[idx].copy()
            for c, nm in enumerate(names):
                if nm not in NO_NORMALIZE:
                    x[c] = nz.transform(f"in::{nm}", x[c])
            np.nan_to_num(x, copy=False)
            with torch.no_grad():
                o = model(torch.from_numpy(x).unsqueeze(0), {"temp"})["temp"][0, 0].numpy()
            return nz.inverse("out::tmp", o)

        # offsets over every test day: the scatter is the mechanism
        off_tr, off_ho = [], []
        for k in te:
            r = np.nan_to_num(ds["target"].isel(time=int(k)).values[0]) - predict(k)
            off_tr.append(r[m_train].mean()); off_ho.append(r[m_hold].mean())
        off_tr, off_ho = np.array(off_tr), np.array(off_ho)
        rr = float(np.corrcoef(off_tr, off_ho)[0, 1])

        # DAY CHOICE. 2023-07-20 (used by the other figures) happens to have a
        # near-zero offset, so the correction can only hurt there — an
        # unrepresentative illustration in the flattering direction's opposite.
        # Default instead to the day whose holdout offset magnitude is the
        # MEDIAN over the test year: typical, not favourable. --day overrides.
        if day_is_default:
            kk = int(np.argsort(np.abs(off_ho))[len(off_ho) // 2])
            kday = int(te[kk]); day = times[kday].strftime("%Y-%m-%d")
        else:
            kk = int(np.where(te == kday)[0][0])
        pred_a = predict(kday)
        truth_a = np.nan_to_num(ds["target"].isel(time=kday).values[0])
        ct, cc = clim[doy[kday]], clim_c[doy[kday]]
        # Year-mean RMSE: the ladder must show the actual result (5.7), not one
        # day, or a single unlucky day would read as the finding.
        yr_m = yr_c = yr_o = yr_i = 0.0
        for n_, k_ in enumerate(te):
            t_ = np.nan_to_num(ds["target"].isel(time=int(k_)).values[0])
            p_ = predict(k_)
            c_ = ds["input"].isel(time=int(k_)).values[stored.index("coarse_tmp")]
            yr_m += np.mean((p_ - t_)[m_hold] ** 2)
            yr_c += np.mean((p_ + off_tr[n_] - t_)[m_hold] ** 2)
            yr_o += np.mean((p_ + off_ho[n_] - t_)[m_hold] ** 2)
            yr_i += np.mean((c_ + clim_c[doy[k_]] - (t_ + clim[doy[k_]]))[m_hold] ** 2)
        Y = [np.sqrt(v / len(te)) for v in (yr_i, yr_m, yr_c, yr_o)]
        coarse = ds["input"].isel(time=kday).values[stored.index("coarse_tmp")] + cc
        truth, pred = truth_a + ct, pred_a + ct
        corr_field = pred + off_tr[kk]

        e_m, e_c = pred - truth, corr_field - truth
        rm = float(np.sqrt(np.mean(e_m[m_hold] ** 2)))
        rc = float(np.sqrt(np.mean(e_c[m_hold] ** 2)))
        ri = float(np.sqrt(np.mean((coarse - truth)[m_hold] ** 2)))
        ro = float(np.sqrt(np.mean(((pred + off_ho[kk]) - truth)[m_hold] ** 2)))

        fig, ax = plt.subplots(2, 4, figsize=(19.5, 9.4))
        vmin, vmax = np.nanpercentile(truth[land], 1), np.nanpercentile(truth[land], 99)
        elim = float(np.nanpercentile(np.abs(e_m[m_hold]), 99))

        def show(a, f, t, cmap, sub="", **kw):
            im = a.imshow(np.ma.masked_where(~land, f), cmap=cmap, **kw)
            graticule(a, LAT, LON); holdout_box(a, box, "")
            a.set_title(t, fontsize=11.5, fontweight="semibold", pad=10)
            if sub:
                a.text(.5, 1.004, sub, transform=a.transAxes, ha="center",
                       va="bottom", fontsize=7.6, color=INK3)
            cb = fig.colorbar(im, ax=a, fraction=.042, pad=.02)
            cb.ax.tick_params(labelsize=7.5, color=GRAT, labelcolor=INK3)
            return cb

        for a, f, t, sub in [
            (ax[0, 0], coarse, "Coarse input (POWER)", f"RMSE {ri:.3f} degC"),
            (ax[0, 1], truth, "Truth (ERA5-Land)", "the target"),
            (ax[0, 2], pred, "Model output", f"RMSE {rm:.3f} degC"),
            (ax[0, 3], corr_field, "Offset-corrected output",
             f"RMSE {rc:.3f} degC  —  {(rc - rm) / rm * 100:+.1f}%")]:
            show(a, f, t, seq, sub, vmin=vmin, vmax=vmax).set_label(
                "degC", fontsize=8, color=INK3)

        show(ax[1, 0], e_m, "Model error (output − truth)", div,
             f"mean over held-out region {e_m[m_hold].mean():+.3f} degC",
             norm=TwoSlopeNorm(0, -elim, elim)).set_label("degC", fontsize=8, color=INK3)
        show(ax[1, 1], e_c, "Corrected error", div,
             f"same colour scale; mean {e_c[m_hold].mean():+.3f} degC",
             norm=TwoSlopeNorm(0, -elim, elim)).set_label("degC", fontsize=8, color=INK3)

        sc = ax[1, 2]
        lim = float(np.nanpercentile(np.abs(np.r_[off_tr, off_ho]), 99.5))
        sc.scatter(off_tr, off_ho, s=9, alpha=.45, color="#2a78d6", edgecolors="none")
        sc.plot([-lim, lim], [-lim, lim], ls="--", lw=1, color=INK3, label="perfect")
        sc.scatter([off_tr[kk]], [off_ho[kk]], s=70, facecolor="none",
                   edgecolor=ORANGE, lw=2, zorder=5, label=day)
        sc.axhline(0, color=GRAT, lw=.6); sc.axvline(0, color=GRAT, lw=.6)
        sc.set_xlim(-lim, lim); sc.set_ylim(-lim, lim)
        sc.set_xlabel("offset measured over the TRAINING region, degC",
                      fontsize=8.5, color=INK2)
        sc.set_ylabel("true offset in the HELD-OUT region, degC",
                      fontsize=8.5, color=INK2)
        sc.set_title(f"Why it works: r = {rr:.3f}", fontsize=11.5,
                     fontweight="semibold", pad=10)
        sc.text(.5, 1.004, f"{len(te)} test days; no holdout truth is used by the correction",
                transform=sc.transAxes, ha="center", va="bottom", fontsize=7.6, color=INK3)
        sc.legend(fontsize=7.5, loc="upper left", framealpha=.9)
        sc.tick_params(labelsize=8, color=GRAT, labelcolor=INK3)

        lad = ax[1, 3]
        items = [("interpolation", Y[0], "#c9c6bf"), ("model", Y[1], "#9aa5b1"),
                 ("+ offset correction", Y[2], "#2a78d6"),
                 ("oracle offset", Y[3], "#eb6834")]
        lad.barh([i[0] for i in items], [i[1] for i in items],
                 color=[i[2] for i in items], height=.55)
        for yi, (_, v, _) in enumerate(items):
            lad.text(v, yi, f"  {v:.3f}", va="center", fontsize=10.5,
                     fontweight="semibold", color=INK)
        lad.set_xlim(0, max(i[1] for i in items) * 1.35)
        lad.set_xlabel("RMSE over the held-out region, degC", fontsize=8.5, color=INK2)
        lad.tick_params(labelsize=9.5, color=GRAT, labelcolor=INK2)
        for sp in ("top", "right", "left"):
            lad.spines[sp].set_visible(False)
        lad.set_title(f"Whole test year: {(Y[2] - Y[1]) / Y[1] * 100:+.1f}%",
                      fontsize=11.5, fontweight="semibold", pad=10)
        lad.text(.5, 1.004, f"all {len(te)} days, not the one mapped above; "
                 f"orange = ceiling if the offset were known exactly",
                 transform=lad.transAxes, ha="center", va="bottom",
                 fontsize=7.6, color=INK3)

        base = arch.rsplit("_s", 1)[0]
        fig.suptitle(f"{TITLES.get(base, base)}   —   {day}", x=.012, y=.995,
                     ha="left", va="top", fontsize=16.5, fontweight="semibold")
        fig.text(.012, .958,
                 f"Seed {arch.rsplit('_s', 1)[-1]}. The correction subtracts ONE "
                 f"number per day, measured over the training region on the same "
                 f"day — it never sees held-out truth. This is the MEDIAN-offset "
                 f"test day, chosen as typical rather than favourable. Maps show "
                 f"the whole domain; "
                 f"metrics are over the {int(m_hold.sum())} land cells in the "
                 f"orange box. Dotted lines are true lat/lon.",
                 ha="left", va="top", fontsize=9.3, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, .93))
        out = FIGDIR / f"offset_{base}.png"
        fig.savefig(out, dpi=155, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out.name}   interp {ri:.3f} | model {rm:.3f} -> corrected "
              f"{rc:.3f} ({(rc - rm) / rm * 100:+.1f}%) | oracle {ro:.3f} | r={rr:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=["of0_s1337", "v2_landcov_s1337"])
    ap.add_argument("--day", default=None,
                    help="default: the median-|offset| test day, i.e. typical")
    ap.add_argument("--zarr", default="data_store_sc/super/dataset.zarr")
    a = ap.parse_args()
    main(a.archs, a.day, a.zarr)
