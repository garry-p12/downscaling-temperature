"""Assemble the Austin evaluation figure set.

Produces, in ``austin_eval/``:
  panels/<arch>_holdout_panel.png   per-model coarse | prediction | truth | error
  all_models_<date>.png             every model on ONE day, same colour scale
  rmse_ranking.png                  RMSE with 95% CI, single-hue magnitude chart
  error_maps.png                    mean |error| per model, shared scale

Design notes (why the colours are what they are):
  * Temperature fields use a DIVERGING map: temperature around a domain mean is
    a bipolar quantity, and a diverging ramp with a neutral midpoint is the
    correct encoding. Never a rainbow.
  * Error magnitude uses a SEQUENTIAL single-hue ramp — magnitude, one hue,
    light to dark.
  * The ranking chart is ONE measure, so it is one colour. No categorical
    palette is involved and no legend is needed; bars are directly labelled.

Usage:
    DOWNSCALE_CONFIG_data=configs/data_southcentral.yaml \\
        python -m evaluation.make_austin_figures
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from common import Normalizer, load_config
from models.model import load_checkpoint
from training.dataset import holdout_bounds
from training.run_all import ckpt_path

INK = "#1a1a1a"
MUTED = "#6b7280"
BAR = "#2563eb"          # single hue: this chart shows one measure
BASELINE_BAR = "#9ca3af"  # the reference row, deliberately recessive


@torch.no_grad()
def predict_all(archs, test, nz, i0, i1, j0, j1, clim_t, lm):
    """Absolute-degC predictions cropped to the holdout, per architecture."""
    names = test["channel_in"].values.tolist()
    inp = test["input"].values
    out = {}
    for arch in archs:
        p = ckpt_path(arch)
        if not p.exists():
            continue
        m, _ = load_checkpoint(p, "cpu", load_config("model"))
        pred = np.empty((inp.shape[0], i1 - i0, j1 - j0), "float32")
        for k in range(inp.shape[0]):
            x = inp[k].copy()
            for c, nm in enumerate(names):
                if nm != "land_mask":
                    x[c] = nz.transform(f"in::{nm}", x[c])
            np.nan_to_num(x, copy=False)
            full = m(torch.from_numpy(x).unsqueeze(0), {"temp"})["temp"][0, 0].numpy()
            pred[k] = full[i0:i1, j0:j1]
        pred = nz.inverse("out::tmp", pred) + clim_t[:, i0:i1, j0:j1]
        out[arch] = np.where(lm[None], pred, np.nan)
        print(f"[fig] {arch} predicted")
    return out


def fig_all_models(preds, truth, interp, day, times, dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["truth (ERA5-Land)", "interpolated POWER"] + list(preds)
    fields = [truth[day], interp[day]] + [preds[a][day] for a in preds]
    n = len(fields)
    cols = 4
    rows = int(np.ceil(n / cols))
    vmin, vmax = np.nanpercentile(np.concatenate([f.ravel() for f in fields]), [2, 98])

    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.2 * rows),
                             squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis("off"); continue
        im = ax.imshow(fields[k], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        weight = "bold" if k < 2 else "normal"
        ax.set_title(names[k], fontsize=10, color=INK, fontweight=weight)
        for s in ax.spines.values():
            s.set_edgecolor("#e5e7eb")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("daily mean 2 m temperature (°C)", color=MUTED, fontsize=9)
    cb.outline.set_edgecolor("#e5e7eb")
    fig.suptitle(f"Austin spatial holdout — {pd.Timestamp(times[day]).date()}"
                 f"   (no model was trained on this region)",
                 fontsize=13, color=INK)
    fig.savefig(dest, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] wrote {dest}")


def fig_error_maps(preds, truth, dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    errs = {a: np.nanmean(np.abs(p - truth), axis=0) for a, p in preds.items()}
    vmax = float(np.nanpercentile(np.concatenate([e.ravel() for e in errs.values()]), 98))
    cols = 4
    rows = int(np.ceil(len(errs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.2 * rows),
                             squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        if k >= len(errs):
            ax.axis("off"); continue
        a = list(errs)[k]
        im = ax.imshow(errs[a], cmap="magma", vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{a}   {np.nanmean(errs[a]):.3f} °C", fontsize=10, color=INK)
        for s in ax.spines.values():
            s.set_edgecolor("#e5e7eb")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("mean |error| over 2023 (°C)", color=MUTED, fontsize=9)
    cb.outline.set_edgecolor("#e5e7eb")
    fig.suptitle("Where each model is wrong — mean absolute error, Austin holdout",
                 fontsize=13, color=INK)
    fig.savefig(dest, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] wrote {dest}")


def fig_ranking(report_path, dest):
    """One measure -> one hue, no legend, direct labels, recessive axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = json.loads(Path(report_path).read_text())["methods"]
    rows = sorted(rep.items(), key=lambda kv: kv[1]["rmse"], reverse=True)
    labels = [k.replace("interpolated_POWER", "interpolated POWER") for k, _ in rows]
    vals = [m["rmse"] for _, m in rows]
    los = [m.get("rmse_ci95", [v, v])[0] for (_, m), v in zip(rows, vals)]
    his = [m.get("rmse_ci95", [v, v])[1] for (_, m), v in zip(rows, vals)]
    colors = [BASELINE_BAR if "POWER" in lbl else BAR for lbl in labels]

    fig, ax = plt.subplots(figsize=(8, 0.46 * len(rows) + 1.4))
    y = np.arange(len(rows))
    ax.barh(y, vals, height=0.62, color=colors,
            xerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
            error_kw=dict(ecolor="#374151", lw=1.2, capsize=3))
    # Offset the label from the CI's UPPER bound, not the bar value — otherwise
    # it lands on top of the error-bar cap.
    pad = max(his) * 0.02
    for yi, v, hi, lab in zip(y, vals, his, labels):
        params = rep[lab.replace("interpolated POWER", "interpolated_POWER")].get("params_M")
        tag = f"{v:.3f}" + (f"   ({params:.2f} M)" if params else "")
        ax.text(hi + pad, yi, tag, va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.set_xlabel("RMSE (°C) vs ERA5-Land, Austin holdout — lower is better",
                  fontsize=10, color=MUTED)
    ax.set_xlim(0, max(his) * 1.22)
    ax.grid(axis="x", color="#eef0f3", lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#e5e7eb")
    ax.tick_params(colors=MUTED, length=0)
    ax.set_title("Austin spatial holdout — error bars are 95% bootstrap CIs\n"
                 "nine of ten architectures overlap across a 319× parameter range",
                 fontsize=12, color=INK, loc="left", pad=12)
    fig.savefig(dest, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] wrote {dest}")


def main(archs, out_root, day):
    cfg = load_config("data")
    zarr = cfg["dataset"]["out_zarr"]
    nz = Normalizer.load(str(Path(zarr).parent / "norm_stats.json"))
    root = Path(out_root); (root / "panels").mkdir(parents=True, exist_ok=True)

    full = xr.open_zarr(zarr, consolidated=True)
    test = full.sel(time=full["split"] == "test")
    i0, i1, j0, j1 = holdout_bounds(full, cfg["holdout"])
    names = full["channel_in"].values.tolist()
    times = test["time"].values
    doy = pd.DatetimeIndex(times).dayofyear.values - 1
    clim_t = full["clim_tmp"].values[doy]
    clim_c = full["clim_coarse_tmp"].values[doy]
    lm = (full["input"].values[0, names.index("land_mask")] > 0.5)[i0:i1, j0:j1]

    truth = np.where(lm[None],
                     (test["target"].values[:, 0] + clim_t)[:, i0:i1, j0:j1], np.nan)
    interp = np.where(lm[None],
                      (test["input"].values[:, names.index("coarse_tmp")]
                       + clim_c)[:, i0:i1, j0:j1], np.nan)

    # Collect the per-model panels produced by holdout_eval.
    src = Path("image_outputs/austin_holdout")
    for p in sorted(src.glob("*/*_holdout_panel.png")):
        shutil.copy(p, root / "panels" / p.name)
    print(f"[fig] copied {len(list((root/'panels').glob('*.png')))} panels")

    preds = predict_all(archs, test, nz, i0, i1, j0, j1, clim_t, lm)
    fig_all_models(preds, truth, interp, day, times, root / "all_models.png")
    fig_error_maps(preds, truth, root / "error_maps.png")
    if (src / "holdout_report.json").exists():
        fig_ranking(src / "holdout_report.json", root / "rmse_ranking.png")
    shutil.copy(src / "holdout_report.json", root / "holdout_report.json")
    for s in Path("outputs").glob("station_check_*.json"):
        shutil.copy(s, root / s.name)
    print(f"[fig] done -> {root}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+",
                    default=["deepsd", "esrt", "edsr", "restormer", "segformer",
                             "swinir_light", "vit", "swin", "maxvit", "convnext"])
    ap.add_argument("--out", default="austin_eval")
    ap.add_argument("--day", type=int, default=200, help="test-split day index")
    args = ap.parse_args()
    main(args.archs, args.out, args.day)
