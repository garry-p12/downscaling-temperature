"""Zero-shot transfer evaluation: apply trained weights to an unseen domain.

Answers the open question in RESULTS.md §7.1 — the Colorado models are trained
and tested on the same grid cells, so per-cell corrections could be memorised
rather than learned. Running them unchanged on a different region separates the
two.

Distinct from ``evaluation/evaluate.py``, which fits BCSD and the lapse-rate
baseline on the *train* split. A transfer domain has no train split by design,
so those baselines are refit here in the only way that keeps the test honest:
the lapse-rate coefficient is carried over FROM THE SOURCE DOMAIN, exactly like
the network weights. Fitting it on the target would give the baseline an
advantage the model is denied.

Normalization also comes from the source domain (see build_dataset --norm-from);
re-fitting on the target would hide the distribution shift from the model.

Usage:
    DOWNSCALE_CONFIG_data=configs/data_austin.yaml \\
        python -m evaluation.transfer_eval --archs restormer segformer swin \\
            --source-report outputs/eval_report_{arch}.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from common import load_config
from evaluation.export_panels import _panel
from models.model import load_checkpoint
from training.dataset import DownscaleDataset
from training.metrics import bias, mae, residual_spatial_corr, rmse, spatial_corr, ssim
from training.run_all import ckpt_path
from training.train import resolve_device


def _metrics(pred, truth, baseline) -> dict:
    return {
        "ssim": ssim(pred, truth),
        "spatial_corr": spatial_corr(pred, truth),
        "resid_corr_vs_bilinear": residual_spatial_corr(pred, truth, baseline),
        "rmse": rmse(pred, truth),
        "mae": mae(pred, truth),
        "bias": bias(pred, truth),
    }


@torch.no_grad()
def run_arch(arch: str, zarr: str, norm: str, device, n_panel: int,
             out_root: Path) -> dict | None:
    ckpt = ckpt_path(arch)
    if not ckpt.exists():
        print(f"[transfer] {arch}: no checkpoint at {ckpt}")
        return None

    model, mcfg = load_checkpoint(ckpt, device, load_config("model"))

    ds = DownscaleDataset(zarr, norm, "test", None)
    nz = ds.nz
    ci = ds.in_names.index("coarse_tmp")
    ti = ds.target_index("tmp")

    coarse, pred, truth = [], [], []
    for k in range(len(ds)):
        x, y, _ = ds[k]
        out = model(x.unsqueeze(0).to(device), {"temp"})
        coarse.append(nz.inverse("in::coarse_tmp", x[ci].numpy()))
        pred.append(nz.inverse("out::tmp", out["temp"][0, 0].cpu().numpy()))
        truth.append(nz.inverse("out::tmp", y[ti].numpy()))
    c, p, t = map(np.stack, (coarse, pred, truth))

    m = _metrics(p, t, c)
    m["params_M"] = sum(q.numel() for q in model.parameters()) / 1e6

    idx = np.linspace(0, len(p) - 1, min(n_panel, len(p))).astype(int)
    out_dir = out_root / arch
    out_dir.mkdir(parents=True, exist_ok=True)
    title = (f"{arch} — ZERO-SHOT on unseen domain  |  "
             f"RMSE {m['rmse']:.3f} °C, SSIM {m['ssim']:.4f}")
    _panel([c[i] for i in idx], [p[i] for i in idx], [t[i] for i in idx],
           title, out_dir / f"{arch}_transfer_panel.png")
    print(f"[transfer] {arch:10s} ssim={m['ssim']:.4f} rmse={m['rmse']:.3f} "
          f"bias={m['bias']:+.3f} -> {out_dir}")
    return m


def main(archs: list[str], n_panel: int, out_root: str,
         source_reports: str) -> None:
    dcfg = load_config("data")
    tcfg = load_config("train")
    zarr = dcfg["dataset"]["out_zarr"]
    norm = str(Path(zarr).parent / "norm_stats.json")
    device = resolve_device(tcfg["device"])
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

    full = xr.open_zarr(zarr, consolidated=True)
    test = full.sel(time=full["split"] == "test")
    truth = test["target"].sel(channel_out="tmp").values
    coarse = test["input"].sel(channel_in="coarse_tmp").values
    dem = test["input"].sel(channel_in="dem").values

    report: dict = {"domain": dcfg["domain"]["name"],
                    "n_test": int(test.sizes["time"]), "methods": {}}

    # Bilinear floor on the TARGET domain.
    report["methods"]["bilinear"] = _metrics(coarse, truth, coarse)

    # Lapse-rate baseline with the SOURCE-domain coefficient, so it transfers
    # under exactly the same rules as the network weights.
    src_swin = Path(source_reports.format(arch="swin"))
    if src_swin.exists():
        lapse = json.loads(src_swin.read_text())["lapse_temp"]
        a, b = lapse["a_per_dem_unit"], lapse["b_offset"]
        report["source_lapse"] = {"a_per_dem_unit": a, "b_offset": b}
        report["methods"]["lapse_source_coeff"] = _metrics(
            coarse + a * dem + b, truth, coarse)

    for arch in archs:
        m = run_arch(arch, zarr, norm, device, n_panel, root)
        if m:
            report["methods"][arch] = m

    out = root / "transfer_report.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"\n{'method':<22}{'SSIM':>8}{'sp_corr':>9}{'resid':>8}"
          f"{'RMSE':>8}{'MAE':>8}{'bias':>8}")
    print("-" * 71)
    for name, m in sorted(report["methods"].items(),
                          key=lambda kv: -kv[1]["ssim"]):
        print(f"{name:<22}{m['ssim']:>8.4f}{m['spatial_corr']:>9.4f}"
              f"{m['resid_corr_vs_bilinear']:>8.3f}{m['rmse']:>8.3f}"
              f"{m['mae']:>8.3f}{m['bias']:>+8.3f}")
    print(f"\n[transfer] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+",
                    default=["restormer", "segformer", "swin"])
    ap.add_argument("--n-panel", type=int, default=4)
    ap.add_argument("--out", default="image_outputs/austin")
    ap.add_argument("--source-report", default="outputs/eval_report_{arch}.json",
                    help="source-domain report, for the transferred lapse coeff")
    args = ap.parse_args()
    main(args.archs, args.n_panel, args.out, args.source_report)
