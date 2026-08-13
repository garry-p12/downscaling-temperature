"""Evaluate on the spatially held-out region (Austin), with bootstrap CIs.

The model runs on the FULL domain and predictions are then cropped to the
holdout box. That is deliberate: cropping the INPUT to the box first would
deny the network the surrounding context it was trained with, and it is not
how the model would be deployed. What must never leak is training signal, and
that is already guaranteed — holdout cells were zeroed in the loss mask for
every training and validation batch.

Everything is scored in absolute degC (climatology added back to the predicted
anomaly) over land cells only, so numbers are comparable to earlier work.

Confidence intervals come from bootstrapping over DAYS, not cells: adjacent
grid cells are strongly correlated, so a cell-wise bootstrap would report
absurdly tight intervals. Days are the closer thing to independent samples —
though even they are autocorrelated, so the intervals remain optimistic.

Usage:
    DOWNSCALE_CONFIG_data=configs/data_southcentral.yaml \\
        python -m evaluation.holdout_eval --archs deepsd edsr vit esrt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from common import Normalizer, load_config
from evaluation.export_panels import _panel
from models.model import load_checkpoint
from training.dataset import holdout_bounds
from training.metrics import (
    bias,
    extreme_bias,
    mae,
    residual_spatial_corr,
    rmse,
    spatial_corr,
    ssim,
)
from training.run_all import ckpt_path


def _bootstrap(pred, truth, n_boot: int, seed: int = 0) -> dict:
    """95% CI for RMSE and SSIM by resampling DAYS with replacement."""
    rng = np.random.RandomState(seed)
    n = pred.shape[0]
    r, s = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        r.append(rmse(pred[idx], truth[idx]))
        s.append(ssim(pred[idx], truth[idx]))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))  # noqa: E731
    return {"rmse_ci95": q(r), "ssim_ci95": q(s)}


def _metrics(pred, truth, base, n_boot: int) -> dict:
    m = {
        "rmse": rmse(pred, truth), "mae": mae(pred, truth),
        "bias": bias(pred, truth), "ssim": ssim(pred, truth),
        "spatial_corr": spatial_corr(pred, truth),
        "resid_corr_vs_interp": residual_spatial_corr(pred, truth, base),
    }
    m.update(extreme_bias(pred, truth))
    if n_boot:
        m.update(_bootstrap(pred, truth, n_boot))
    return m


@torch.no_grad()
def predict(arch: str, ds_full: xr.Dataset, nz: Normalizer, device) -> np.ndarray | None:
    """Full-domain prediction (normalized anomaly space) for the test split."""
    ckpt = ckpt_path(arch)
    if not ckpt.exists():
        print(f"[holdout] {arch}: no checkpoint yet, skipping")
        return None
    model, mcfg = load_checkpoint(ckpt, device, load_config("model"))

    names = ds_full["channel_in"].values.tolist()
    inp = ds_full["input"].values
    out = np.empty((inp.shape[0], inp.shape[2], inp.shape[3]), "float32")
    for k in range(inp.shape[0]):
        x = inp[k].copy()
        for c, nm in enumerate(names):
            if nm != "land_mask":
                x[c] = nz.transform(f"in::{nm}", x[c])
        np.nan_to_num(x, copy=False)
        t = torch.from_numpy(x).unsqueeze(0).to(device)
        out[k] = model(t, {"temp"})["temp"][0, 0].cpu().numpy()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[holdout] {arch:14s} predicted ({n_par:.2f}M params)")
    return out, n_par


def main(archs: list[str], n_boot: int, out_root: str, device_name: str) -> None:
    dcfg = load_config("data")
    zarr = dcfg["dataset"]["out_zarr"]
    nz = Normalizer.load(str(Path(zarr).parent / "norm_stats.json"))
    device = torch.device(device_name)

    full = xr.open_zarr(zarr, consolidated=True)
    test = full.sel(time=full["split"] == "test")
    i0, i1, j0, j1 = holdout_bounds(full, dcfg["holdout"])
    names = full["channel_in"].values.tolist()

    # Absolute degC = anomaly + climatology(day-of-year).
    doy = pd.DatetimeIndex(test["time"].values).dayofyear.values - 1
    clim_t = full["clim_tmp"].values[doy]
    clim_c = full["clim_coarse_tmp"].values[doy]
    truth = test["target"].values[:, 0] + clim_t
    interp = test["input"].values[:, names.index("coarse_tmp")] + clim_c
    land = full["input"].values[0, names.index("land_mask")] > 0.5

    def crop(a):
        return a[:, i0:i1, j0:j1]

    lm = crop(land[None])[0]
    def maskit(a):
        return np.where(lm[None], crop(a), np.nan)

    truth_h, interp_h = maskit(truth), maskit(interp)
    print(f"[holdout] region rows {i0}:{i1} cols {j0}:{j1} "
          f"= {i1-i0}x{j1-j0}, {int(lm.sum())} land cells, "
          f"{test.sizes['time']} days")

    report = {"domain": dcfg["domain"]["name"],
              "holdout": dcfg["holdout"]["name"],
              "n_days": int(test.sizes["time"]),
              "n_cells": int(lm.sum()), "methods": {}}

    report["methods"]["interpolated_POWER"] = _metrics(
        interp_h, truth_h, interp_h, n_boot)

    root = Path(out_root)
    for arch in archs:
        got = predict(arch, test, nz, device)
        if got is None:
            continue
        pred_anom, n_par = got
        pred = maskit(nz.inverse("out::tmp", pred_anom) + clim_t)
        m = _metrics(pred, truth_h, interp_h, n_boot)
        m["params_M"] = n_par
        report["methods"][arch] = m

        d = root / arch
        d.mkdir(parents=True, exist_ok=True)
        idx = np.linspace(0, len(pred) - 1, 4).astype(int)
        _panel([interp_h[i] for i in idx], [pred[i] for i in idx],
               [truth_h[i] for i in idx],
               f"{arch} — AUSTIN SPATIAL HOLDOUT (never trained on)  |  "
               f"RMSE {m['rmse']:.3f} °C, SSIM {m['ssim']:.4f}",
               d / f"{arch}_holdout_panel.png")

    root.mkdir(parents=True, exist_ok=True)
    (root / "holdout_report.json").write_text(json.dumps(report, indent=2))

    floor = report["methods"]["interpolated_POWER"]["rmse"]
    print(f"\n{'method':<20}{'RMSE':>8}{'  95% CI':>16}{'SSIM':>8}"
          f"{'resid':>7}{'skill':>7}{'bias':>8}{'p95':>8}{'params':>9}")
    print("-" * 92)
    for name, m in sorted(report["methods"].items(), key=lambda kv: kv[1]["rmse"]):
        lo, hi = m.get("rmse_ci95", (float("nan"),) * 2)
        pm = m.get("params_M")
        print(f"{name:<20}{m['rmse']:>8.4f}  [{lo:.3f}, {hi:.3f}]"
              f"{m['ssim']:>8.4f}{m['resid_corr_vs_interp']:>7.3f}"
              f"{floor / m['rmse']:>7.3f}{m['bias']:>+8.3f}"
              f"{m['p95_bias']:>+8.3f}"
              f"{('-' if pm is None else f'{pm:.2f}M'):>9}")
    print("\nskill = interpolated_POWER_rmse / method_rmse  (>1 beats interpolation)")
    print(f"[holdout] wrote {root / 'holdout_report.json'}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out", default="image_outputs/austin_holdout")
    ap.add_argument("--device", default="cpu",
                    help="cpu keeps MPS free for a concurrent training sweep")
    args = ap.parse_args()
    main(args.archs, args.n_boot, args.out, args.device)
