"""One inference sheet per variant: what the model was given, what it produced,
and where it went wrong.

Each sheet shows the same day for every variant so they can be compared
directly, and every panel carries a true lat/lon graticule plus the outline of
the held-out region the metrics are computed over.

    row 1   coarse input / ERA5-Land truth / model output      absolute degC,
                                                               one shared scale
    row 2   interpolation error / model error                  diverging, one
                                                               shared scale, so
                                                               the shrinkage IS
                                                               the improvement
            local SSIM map                                     where structure
                                                               is reproduced
            residual scatter                                   the correction
                                                               needed vs the
                                                               correction made

WHY THE TWO ERROR PANELS SHARE A COLOUR SCALE
    Plotted independently each would auto-scale to its own range and the model
    would look no better than the interpolation it corrects. Sharing the scale
    is the whole point of the comparison.

WHY RESIDUAL CORRELATION GETS A SCATTER RATHER THAN A MAP
    It is a single number per model, defined against the interpolation
    baseline: corr(truth - interp, model - interp) over the held-out cells. The
    honest picture is the scatter it is computed from — x is the correction the
    day actually needed, y is the correction the model made. A perfect model
    sits on y = x; a model that does nothing sits on y = 0. The fitted slope
    shows the amplitude shrinkage README 5.5 measures spectrally.

Usage:
    python scripts/make_inference_panels.py --day 2023-07-20
    python scripts/make_inference_panels.py --archs v2_landcov_s1337 of0_s1337
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
from common.normalize import Normalizer                         # noqa: E402
from data.build_dataset import NO_NORMALIZE                     # noqa: E402
from evaluation.mapping import (GRAT, INK, INK2, INK3, ORANGE,  # noqa: E402
                                SURF, graticule, holdout_box, palettes)
from models.model import load_checkpoint                        # noqa: E402
from training.dataset import holdout_bounds                     # noqa: E402
from training.metrics import ssim as ssim_metric                # noqa: E402

FIGDIR = REPO / "image_outputs" / "slides"
DEFAULT = ["v0_base_s1337", "v1_terrain_s1337", "v2_landcov_s1337",
           "v3_nocoast_s1337", "v4_elevstd_s1337", "of0_s1337", "of1_s1337"]

TITLES = {
    "v0_base": "V0 baseline — 8 channels",
    "v1_terrain": "V1 + terrain shape — 12 channels",
    "v2_landcov": "V2 land-cover composition — 13 channels (project default)",
    "v3_nocoast": "V3 distance-to-coast removed — 7 channels",
    "v4_elevstd": "V4 + roughness only — 9 channels",
    "of0": "Offset-split loss, offset_weight 0.0 — 13 channels",
    "of1": "Offset-split loss, offset_weight 1.0 — 13 channels",
    "lin": "Linear probe (affine, 2.2k params) — 13 channels",
}


def local_ssim(a, b, ws=11, sigma=1.5):
    """Per-pixel SSIM so the structural term can be SEEN, not just quoted."""
    import torch.nn.functional as F
    c = torch.arange(ws, dtype=torch.float32) - ws // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    win = torch.outer(g, g).view(1, 1, ws, ws)
    A = torch.from_numpy(a[None, None].astype("float32"))
    B = torch.from_numpy(b[None, None].astype("float32"))
    p = ws // 2
    mu_a, mu_b = F.conv2d(A, win, padding=p), F.conv2d(B, win, padding=p)
    saa = F.conv2d(A * A, win, padding=p) - mu_a ** 2
    sbb = F.conv2d(B * B, win, padding=p) - mu_b ** 2
    sab = F.conv2d(A * B, win, padding=p) - mu_a * mu_b
    rng = float(np.nanmax(b) - np.nanmin(b))          # the DATA's range, not 1.0
    c1, c2 = (0.01 * rng) ** 2, (0.03 * rng) ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / \
        ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return s[0, 0].numpy()


def main(archs, day, zarr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    seq, div = palettes()
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                         "savefig.facecolor": SURF, "text.color": INK,
                         "font.size": 10})

    cfg = load_config("configs/data_southcentral.yaml")
    ds = xr.open_zarr(zarr, consolidated=True)
    nz = Normalizer.load(str(Path(zarr).parent / "norm_stats.json"))
    stored = ds["channel_in"].values.tolist()
    times = pd.DatetimeIndex(ds["time"].values)
    k = int(np.argmin(np.abs(times - pd.Timestamp(day))))
    day = times[k].strftime("%Y-%m-%d")
    doy = times[k].dayofyear - 1

    from common.grid import build_target_grid
    tg = build_target_grid(cfg["domain"], cfg["grid"])
    LAT, LON = tg.latlon()
    i0, i1, j0, j1 = box = holdout_bounds(ds, cfg["holdout"])
    hm = np.zeros(LAT.shape, bool); hm[i0:i1, j0:j1] = True

    clim_t = ds["clim_tmp"].values[doy]
    clim_c = ds["clim_coarse_tmp"].values[doy]
    land = ds["input"].isel(time=k).values[stored.index("land_mask")] > 0.5
    truth_a = np.nan_to_num(ds["target"].isel(time=k).values[0])
    coarse_a = ds["input"].isel(time=k).values[stored.index("coarse_tmp")]
    truth, coarse = truth_a + clim_t, coarse_a + clim_c
    score = hm & land                                  # where metrics are taken

    FIGDIR.mkdir(parents=True, exist_ok=True)
    print(f"reference day {day} ('{ds['split'].values[k]}' split), "
          f"{int(score.sum())} scored cells\n")

    for arch in archs:
        ck = REPO / "checkpoints" / arch / "last.pt"
        if not ck.exists():
            print(f"  {arch}: no last.pt, skipping"); continue
        model, mcfg = load_checkpoint(ck, "cpu", load_config("model"))
        names = list(mcfg.get("use_channels") or stored)
        if any(n not in stored for n in names):
            print(f"  {arch}: store lacks {[n for n in names if n not in stored]}")
            continue
        x = ds["input"].isel(time=k).values[[stored.index(n) for n in names]].copy()
        for c, nm in enumerate(names):
            if nm not in NO_NORMALIZE:
                x[c] = nz.transform(f"in::{nm}", x[c])
        np.nan_to_num(x, copy=False)
        with torch.no_grad():
            pred_a = model(torch.from_numpy(x).unsqueeze(0), {"temp"})["temp"][0, 0].numpy()
        pred_a = nz.inverse("out::tmp", pred_a)
        pred = pred_a + clim_t

        err_m, err_i = pred - truth, coarse - truth
        rmse_m = float(np.sqrt(np.mean(err_m[score] ** 2)))
        rmse_i = float(np.sqrt(np.mean(err_i[score] ** 2)))
        need, made = (truth - coarse)[score], (pred - coarse)[score]
        rc = float(np.corrcoef(need, made)[0, 1])
        slope = float(np.polyfit(need, made, 1)[0])
        sm = local_ssim(np.where(land, pred_a, 0), np.where(land, truth_a, 0))
        # metrics.ssim expects a (N, H, W) stack; one day is N=1.
        ssim_all = float(ssim_metric(pred_a[None, i0:i1, j0:j1],
                                     truth_a[None, i0:i1, j0:j1]))

        fig, axes = plt.subplots(2, 4, figsize=(19.5, 9.4))
        vmin, vmax = (np.nanpercentile(truth[land], 1),
                      np.nanpercentile(truth[land], 99))
        elim = float(np.nanpercentile(np.abs(err_i[score]), 99))

        def show(ax, field, title, cmap, sub="", **kw):
            im = ax.imshow(np.ma.masked_where(~land, field), cmap=cmap, **kw)
            graticule(ax, LAT, LON); holdout_box(ax, box, "")
            ax.set_title(title, fontsize=11.5, fontweight="semibold", pad=10)
            if sub:
                ax.text(.5, 1.004, sub, transform=ax.transAxes, ha="center",
                        va="bottom", fontsize=7.6, color=INK3)
            cb = fig.colorbar(im, ax=ax, fraction=.042, pad=.02)
            cb.ax.tick_params(labelsize=7.5, color=GRAT, labelcolor=INK3)
            return cb

        show(axes[0, 0], coarse, "Coarse input (NASA POWER, interpolated)",
             seq, f"RMSE {rmse_i:.3f} degC over the held-out region",
             vmin=vmin, vmax=vmax).set_label("degC", fontsize=8, color=INK3)
        show(axes[0, 1], truth, "Truth (ERA5-Land)", seq,
             "the target, 10 km", vmin=vmin, vmax=vmax
             ).set_label("degC", fontsize=8, color=INK3)
        show(axes[0, 2], pred, "Model output (downscaled)", seq,
             f"RMSE {rmse_m:.3f} degC  —  skill {rmse_i / rmse_m:.2f}x",
             vmin=vmin, vmax=vmax).set_label("degC", fontsize=8, color=INK3)

        ab = axes[0, 3]
        bars = [("interpolation", rmse_i, "#9aa5b1"), ("model", rmse_m, "#2a78d6")]
        ab.barh([b[0] for b in bars], [b[1] for b in bars],
                color=[b[2] for b in bars], height=.45)
        for yi, (_, v, _) in enumerate(bars):
            ab.text(v, yi, f"  {v:.3f}", va="center", fontsize=11,
                    fontweight="semibold", color=INK)
        ab.set_xlim(0, max(rmse_i, rmse_m) * 1.4)
        ab.set_xlabel("RMSE over the held-out region, degC", fontsize=8.5, color=INK2)
        ab.tick_params(labelsize=10, color=GRAT, labelcolor=INK2)
        ab.set_aspect("auto")
        for sp in ("top", "right", "left"):
            ab.spines[sp].set_visible(False)
        ab.set_title(f"Skill {rmse_i / rmse_m:.2f}x", fontsize=11.5,
                     fontweight="semibold", pad=10)
        ab.text(.5, 1.004, f"SSIM {ssim_all:.4f}   residual corr {rc:.4f}",
                transform=ab.transAxes, ha="center", va="bottom",
                fontsize=7.6, color=INK3)

        show(axes[1, 0], err_i, "Interpolation error (input − truth)", div,
             "same colour scale as the panel to its right",
             norm=TwoSlopeNorm(0, -elim, elim)).set_label("degC", fontsize=8, color=INK3)
        show(axes[1, 1], err_m, "Model error (output − truth)", div,
             "shrinkage against the panel on the left IS the improvement",
             norm=TwoSlopeNorm(0, -elim, elim)).set_label("degC", fontsize=8, color=INK3)
        show(axes[1, 2], sm, "Local SSIM (structure)",
             seq, f"windowed 11x11; mean over the held-out region {ssim_all:.3f}",
             vmin=0, vmax=1).set_label("SSIM", fontsize=8, color=INK3)

        sc = axes[1, 3]
        lim = float(np.nanpercentile(np.abs(need), 99.5))
        sc.hexbin(need, made, gridsize=44, cmap="Blues", mincnt=1,
                  extent=(-lim, lim, -lim, lim))
        sc.plot([-lim, lim], [-lim, lim], color=INK3, lw=1, ls="--", label="perfect")
        sc.plot([-lim, lim], [-lim * slope, lim * slope], color=ORANGE, lw=1.6,
                label=f"fit, slope {slope:.2f}")
        sc.axhline(0, color=GRAT, lw=.6); sc.axvline(0, color=GRAT, lw=.6)
        sc.set_xlim(-lim, lim); sc.set_ylim(-lim, lim)
        sc.set_xlabel("correction needed  (truth − input), degC", fontsize=8.5, color=INK2)
        sc.set_ylabel("correction made  (output − input), degC", fontsize=8.5, color=INK2)
        sc.set_title(f"Residual correlation {rc:.4f}", fontsize=11.5,
                     fontweight="semibold", pad=10)
        sc.text(.5, 1.004, "slope < 1 = the model under-corrects (README 5.5)",
                transform=sc.transAxes, ha="center", va="bottom",
                fontsize=7.6, color=INK3)
        sc.legend(fontsize=7.5, loc="upper left", framealpha=.9)
        sc.tick_params(labelsize=8, color=GRAT, labelcolor=INK3)

        base = arch.rsplit("_s", 1)[0]
        fig.suptitle(f"{TITLES.get(base, base)}   —   {day}",
                     x=.012, y=.995, ha="left", va="top", fontsize=16.5,
                     fontweight="semibold")
        fig.text(.012, .958,
                 f"Seed {arch.rsplit('_s', 1)[-1]}. Maps cover the whole domain; "
                 f"every metric is computed over the {int(score.sum())} land "
                 f"cells inside the orange held-out region, which no model "
                 f"trained on. Dotted lines are true lat/lon (EPSG:5070 Albers, "
                 f"so parallels curve); ticks are exact on the left and bottom "
                 f"edges.", ha="left", va="top", fontsize=9.3, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, .93))
        out = FIGDIR / f"inference_{base}.png"
        fig.savefig(out, dpi=155, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out.name}   RMSE {rmse_i:.3f} -> {rmse_m:.3f}  "
              f"SSIM {ssim_all:.4f}  resid {rc:.4f}  slope {slope:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=DEFAULT)
    ap.add_argument("--day", default="2023-07-20")
    ap.add_argument("--zarr", default="data_store_sc/super/dataset.zarr")
    a = ap.parse_args()
    main(a.archs, a.day, a.zarr)
