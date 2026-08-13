"""Export coarse / prediction / truth / |error| panels per architecture to disk.

Same figure the trainer streams to wandb, regenerated from the saved
checkpoints so the images exist as files rather than only inside a run.
Written to ``image_outputs/<arch>/``.

Samples are taken at fixed indices spread across the split (not random and not
the first N days) so every architecture is shown on the SAME days and the
panels are directly comparable side by side.

Usage:
    python -m evaluation.export_panels
    python -m evaluation.export_panels --split test --n 4 --archs swin restormer
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import load_config
from models.model import load_checkpoint
from training.dataset import DownscaleDataset
from training.metrics import rmse, ssim
from training.run_all import ALL_ARCHS, ckpt_path
from training.train import resolve_device


def _panel(coarse, pred, truth, title: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(coarse)
    # Shared colour scale across coarse/pred/truth so the eye compares fields,
    # not colour maps; |error| gets its own.
    allv = np.concatenate([np.stack([c, p, t]).ravel()
                           for c, p, t in zip(coarse, pred, truth)])
    vmin, vmax = np.nanpercentile(allv, [2, 98])
    emax = max(np.nanpercentile(np.abs(p - t), 98)
               for p, t in zip(pred, truth)) + 1e-6

    labels = ["coarse input (POWER)", "prediction", "truth (ERA5-Land)", "|error|"]
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.1 * n), squeeze=False)
    for i in range(n):
        panels = [coarse[i], pred[i], truth[i], np.abs(pred[i] - truth[i])]
        for j, img in enumerate(panels):
            ax = axes[i][j]
            if j < 3:
                im = ax.imshow(img, cmap="RdBu_r", vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(img, cmap="magma", vmin=0, vmax=emax)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(labels[j], fontsize=11)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        axes[i][0].set_ylabel(f"sample {i}", fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def export(arch: str, zarr: str, norm: str, split: str, n: int,
           out_root: Path, device) -> dict | None:
    ckpt = ckpt_path(arch)
    if not ckpt.exists():
        print(f"[panels] {arch}: no checkpoint at {ckpt}, skipping")
        return None

    model, mcfg = load_checkpoint(ckpt, device, load_config("model"))

    ds = DownscaleDataset(zarr, norm, split, None)
    if len(ds) == 0:
        print(f"[panels] {arch}: split '{split}' is empty")
        return None
    nz = ds.nz
    ci = ds.in_names.index("coarse_tmp")
    ti = ds.target_index("tmp")

    # Fixed, evenly spread indices -> identical days for every architecture.
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    coarse, pred, truth = [], [], []
    for k in idx:
        x, y, _ = ds[int(k)]
        out = model(x.unsqueeze(0).to(device), {"temp"})
        coarse.append(nz.inverse("in::coarse_tmp", x[ci].numpy()))
        pred.append(nz.inverse("out::tmp", out["temp"][0, 0].cpu().numpy()))
        truth.append(nz.inverse("out::tmp", y[ti].numpy()))

    p = np.stack(pred)
    t = np.stack(truth)
    stats = {"rmse": rmse(p, t), "ssim": ssim(p, t)}

    out_dir = out_root / arch
    out_dir.mkdir(parents=True, exist_ok=True)
    title = (f"{arch} — {split} split, day indices "
             f"{', '.join(str(int(i)) for i in idx)}  |  "
             f"panel RMSE {stats['rmse']:.3f} °C, SSIM {stats['ssim']:.4f}")
    _panel(coarse, p, t, title, out_dir / f"{arch}_{split}_panel.png")
    print(f"[panels] {arch}: wrote {out_dir}/{arch}_{split}_panel.png "
          f"(rmse {stats['rmse']:.3f}, ssim {stats['ssim']:.4f})")
    return stats


def main(archs: list[str], split: str, n: int, out_root: str,
         zarr: str | None) -> None:
    dcfg = load_config("data")
    tcfg = load_config("train")
    zarr = zarr or dcfg["dataset"]["out_zarr"]
    norm = str(Path(zarr).parent / "norm_stats.json")
    device = resolve_device(tcfg["device"])
    root = Path(out_root)

    for arch in archs:
        export(arch, zarr, norm, split, n, root, device)
    print(f"[panels] done -> {root}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=ALL_ARCHS)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="image_outputs")
    ap.add_argument("--zarr", default=None, help="override dataset path")
    args = ap.parse_args()
    main(args.archs, args.split, args.n, args.out, args.zarr)
