"""Render downscaling prediction panels for wandb (or disk).

Builds a grid: one row per sample, columns = [coarse input, prediction, truth,
|error|], in denormalized physical units with a shared colorbar per field so
the eye can compare. Returns a matplotlib Figure; the trainer wraps it in
wandb.Image.
"""
from __future__ import annotations

import numpy as np
import torch


def _denorm(nz, key, arr):
    return nz.inverse(key, arr)


@torch.no_grad()
def prediction_panel(model, ds, tasks, device, n: int = 4,
                     field: str = "tmp"):
    """Figure comparing coarse input / prediction / truth / error for n samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    head_key, coarse_name, cmap = "temp", f"coarse_{field}", "RdBu_r"
    nz = ds.nz
    ci = ds.in_names.index(coarse_name)
    oi = ds.target_index(field)

    n = min(n, len(ds))
    rows = []
    model.eval()
    for k in range(n):
        x, y, *_ = ds[k]
        out = model(x.unsqueeze(0).to(device), tasks)
        if head_key not in out:
            return None
        coarse = _denorm(nz, f"in::{coarse_name}", x[ci].numpy())
        pred = _denorm(nz, f"out::{field}", out[head_key][0, 0].cpu().numpy())
        truth = _denorm(nz, f"out::{field}", y[oi].numpy())
        rows.append((coarse, pred, truth))

    # Shared color scale across field columns; error gets its own.
    allvals = np.concatenate([np.stack([r[0], r[1], r[2]]).ravel() for r in rows])
    vmin, vmax = np.nanpercentile(allvals, [2, 98])
    emax = max(np.nanpercentile(np.abs(r[1] - r[2]), 98) for r in rows) + 1e-6

    titles = ["coarse input", "prediction", "truth", "|error|"]
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n), squeeze=False)
    for i, (coarse, pred, truth) in enumerate(rows):
        panels = [coarse, pred, truth, np.abs(pred - truth)]
        for j, img in enumerate(panels):
            ax = axes[i][j]
            if j < 3:
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(img, cmap="magma", vmin=0, vmax=emax)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{field} downscaling — sample {field} fields")
    fig.tight_layout()
    return fig
