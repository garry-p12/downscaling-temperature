"""Evaluate a trained model on the test split against the baselines.

Reports temperature RMSE/MAE/bias/spatial-corr for: model, bilinear (the
coarse POWER field on the target grid), lapse-rate, BCSD. Also dumps a
radially-averaged power spectrum comparison (model vs. truth) to check for
over-smoothing.

Usage:
    python -m evaluation.evaluate --ckpt checkpoints/best.pt
    python -m evaluation.evaluate --ckpt checkpoints/best.pt --wandb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from common import load_config
from evaluation.baselines import BCSD, LapseRateBaseline, bilinear_baseline
from models.model import load_checkpoint
from training.dataset import DownscaleDataset
from training.metrics import power_spectrum, residual_spatial_corr, temperature_metrics
from training.train import active_tasks, resolve_device


@torch.no_grad()
def model_predict(ckpt: str, zarr: str, norm: str, tasks: set[str],
                  device) -> dict:
    model, mcfg = load_checkpoint(ckpt, device, load_config("model"))
    print(f"[eval] arch={mcfg.get('arch', 'swin')} "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    ds = DownscaleDataset(zarr, norm, "test", None)
    nz = ds.nz
    preds: dict[str, list] = {"tmp": []}
    for k in range(len(ds)):
        x, *_ = ds[k]
        out = model(x.unsqueeze(0).to(device), tasks)
        if "temp" in out:
            p = out["temp"][0, 0].cpu().numpy()
            preds["tmp"].append(nz.inverse("out::tmp", p))
    meta = {"arch": mcfg.get("arch", "swin"),
            "params_M": sum(p.numel() for p in model.parameters()) / 1e6}
    return {k: np.stack(v) for k, v in preds.items() if v}, meta


def main(ckpt: str, to_wandb: bool = False) -> None:
    dcfg = load_config("data")
    tcfg = load_config("train")
    zarr = dcfg["dataset"]["out_zarr"]
    norm = str(Path(zarr).parent / "norm_stats.json")
    device = resolve_device(tcfg["device"])
    tasks = active_tasks(tcfg)

    full = xr.open_zarr(zarr, consolidated=True)
    test = full.sel(time=full["split"] == "test")
    truth_tmp = test["target"].sel(channel_out="tmp").values

    report: dict = {"n_test": int(test.sizes["time"])}

    # Temperature baselines (bilinear = the hard-to-beat floor's weak sibling;
    # lapse-rate = the real floor: bilinear + elevation correction).
    bilin_t = bilinear_baseline(test, "coarse_tmp")
    lapse = LapseRateBaseline().fit(full, "coarse_tmp", "tmp")
    lapse_t = lapse.predict(test, "coarse_tmp")

    # Model.
    mpred, meta = model_predict(ckpt, zarr, norm, tasks, device)
    report["arch"] = meta["arch"]
    report["params_M"] = meta["params_M"]
    if "tmp" in mpred:
        report["model_temp"] = temperature_metrics(mpred["tmp"], truth_tmp)
        # Residual corr vs BOTH floors — fine-scale skill after each is removed.
        report["model_temp"]["resid_corr_vs_bilinear"] = \
            residual_spatial_corr(mpred["tmp"], truth_tmp, bilin_t)
        report["model_temp"]["resid_corr_vs_lapse"] = \
            residual_spatial_corr(mpred["tmp"], truth_tmp, lapse_t)

    report["bilinear_temp"] = temperature_metrics(bilin_t, truth_tmp)
    report["lapse_temp"] = temperature_metrics(lapse_t, truth_tmp)
    report["lapse_temp"]["a_per_dem_unit"] = float(lapse.a)
    report["lapse_temp"]["b_offset"] = float(lapse.b)

    bcsd_t = BCSD().fit(full, "coarse_tmp", "tmp").predict(test, "coarse_tmp")
    report["bcsd_temp"] = temperature_metrics(bcsd_t, truth_tmp)

    # --- Edge/interior disambiguation (the test) ---------------------------- #
    # Crop a window_size boundary strip and recompute temperature skill on the
    # interior only. If model beats bilinear on the interior but not overall,
    # the boundary artifact alone is dragging the aggregate below baseline.
    if "tmp" in mpred:
        margin = load_config("model")["encoder"]["window_size"]

        def _rmse(a, b, m):
            a = a[:, m:-m, m:-m] if m else a
            b = b[:, m:-m, m:-m] if m else b
            return float(np.sqrt(np.nanmean((a - b) ** 2)))

        dis = {}
        for label, m in (("full", 0), (f"interior_m{margin}", margin)):
            mr = _rmse(mpred["tmp"], truth_tmp, m)
            br = _rmse(bilin_t, truth_tmp, m)
            lr = _rmse(lapse_t, truth_tmp, m)
            dis[label] = {
                "model_rmse": mr, "bilinear_rmse": br, "lapse_rmse": lr,
                "skill_vs_bilinear": br / (mr + 1e-9),
                "skill_vs_lapse": lr / (mr + 1e-9),
            }
        report["edge_disambiguation"] = dis
        print("\n=== edge/interior disambiguation (temp) ===")
        for label, d in dis.items():
            print(f"{label:14s} model={d['model_rmse']:.3f} "
                  f"bilinear={d['bilinear_rmse']:.3f} lapse={d['lapse_rmse']:.3f} "
                  f"| skill_vs_bilinear={d['skill_vs_bilinear']:.3f} "
                  f"skill_vs_lapse={d['skill_vs_lapse']:.3f}")
        print()

    # Power spectrum (first test field, model vs truth). Over-smoothing shows
    # up as missing power at high wavenumbers — the key check for a 5x
    # deterministic corrector, so it runs on temperature too.
    if mpred.get("tmp") is not None:
        k, ps_m = power_spectrum(mpred["tmp"][0])
        _, ps_t = power_spectrum(truth_tmp[0])
        report["power_spectrum_tmp"] = {
            "k": k.tolist(), "model": ps_m.tolist(), "truth": ps_t.tolist()}

    # One report per arch, so comparison runs accumulate instead of clobbering.
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / f"eval_report_{meta['arch']}.json"
    report_path.write_text(json.dumps(report, indent=2))
    if meta["arch"] == "swin":       # keep the original filename working too
        (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2))
    # Console summary.
    for key in ("model_temp", "bilinear_temp", "lapse_temp", "bcsd_temp"):
        if key in report:
            print(f"{key:18s} {report[key]}")
    print(f"[eval] wrote {report_path}")

    if to_wandb:
        log_report_to_wandb(report, tcfg, ckpt)


# --------------------------------------------------------------------------- #
# wandb
# --------------------------------------------------------------------------- #
def log_report_to_wandb(report: dict, tcfg: dict, ckpt: str) -> None:
    """Push the test-split report to a separate wandb run (stage=eval).

    Training streams *val* metrics; this is the held-out *test* number, so it
    gets its own run rather than overwriting the training run's history.
    Scalars land in the run summary as ``test/<block>/<metric>``; the power
    spectra go up as log-log figures (the over-smoothing check).
    """
    from training.tracking import Tracker

    wcfg = dict(tcfg["logging"].get("wandb", {}))
    wcfg["enabled"] = True
    base = wcfg.get("run_name") or Path(ckpt).stem
    wcfg["run_name"] = f"{base}-eval"
    tracker = Tracker(wcfg, {"train": dict(tcfg)},
                      {"stage": "eval", "ckpt": str(ckpt)})
    if not tracker.enabled:
        print("[eval] wandb disabled or unavailable; skipped upload")
        return

    flat: dict[str, float] = {"test/n_test": report["n_test"]}
    for block, val in report.items():
        if not isinstance(val, dict) or block.startswith("power_spectrum"):
            continue
        for k, v in val.items():
            if isinstance(v, dict):                       # edge_disambiguation
                for kk, vv in v.items():
                    flat[f"test/{block}/{k}/{kk}"] = vv
            else:
                flat[f"test/{block}/{k}"] = v
    for k, v in flat.items():
        tracker.summary(k, v)
    tracker.log(flat)

    for block, val in report.items():
        if not block.startswith("power_spectrum"):
            continue
        tracker.log_figure(f"test/{block}", _spectrum_figure(block, val))

    print(f"[eval] wandb: {tracker.run.url}")
    tracker.finish()


def _spectrum_figure(title: str, spec: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = np.asarray(spec["k"][1:])          # drop k=0 (the mean) for log axes
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(k, np.asarray(spec["model"][1:]), label="model")
    ax.loglog(k, np.asarray(spec["truth"][1:]), label="truth")
    ax.set_xlabel("wavenumber k"); ax.set_ylabel("power")
    ax.set_title(f"{title} — power below truth at high k == over-smoothing")
    ax.legend(); fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--wandb", action="store_true",
                    help="also push the test report to wandb (stage=eval run)")
    args = ap.parse_args()
    main(args.ckpt, args.wandb)
