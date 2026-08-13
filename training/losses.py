"""Loss functions.

Temperature: pixel loss + (1 - SSIM), the SSIM term rewarding spatial structure.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# SSIM (single-channel, Gaussian window)
# --------------------------------------------------------------------------- #
def _gaussian_window(ws: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(ws, device=device, dtype=dtype) - ws // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = torch.outer(g, g)
    return w.view(1, 1, ws, ws)


def ssim(pred: torch.Tensor, target: torch.Tensor, ws: int = 11,
         sigma: float = 1.5, data_range: float = 1.0) -> torch.Tensor:
    """Mean SSIM over a batch of (B, 1, H, W) tensors."""
    win = _gaussian_window(ws, sigma, pred.device, pred.dtype)
    pad = ws // 2
    mu_p = F.conv2d(pred, win, padding=pad)
    mu_t = F.conv2d(target, win, padding=pad)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    sig_p = F.conv2d(pred * pred, win, padding=pad) - mu_p2
    sig_t = F.conv2d(target * target, win, padding=pad) - mu_t2
    sig_pt = F.conv2d(pred * target, win, padding=pad) - mu_pt
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mu_pt + c1) * (2 * sig_pt + c2)) / \
        ((mu_p2 + mu_t2 + c1) * (sig_p + sig_t + c2))
    return s.mean()


def temp_loss(pred: torch.Tensor, target: torch.Tensor,
              ssim_weight: float = 0.2, base: str = "l1",
              mask: torch.Tensor | None = None) -> torch.Tensor:
    """Pixel loss + a structural term.

    ``base='l1'`` by default rather than MSE. Both push toward the conditional
    mean, but MSE penalises large errors quadratically and so blurs harder;
    L1 preserves edges better, which matters here because the whole point of
    downscaling is fine-scale structure. The SSIM term pushes further in the
    same direction — it is a metric we report, so optimizing it directly is
    deliberate, and that overlap is stated in RESULTS.md.

    ``mask`` (1 = score this cell, 0 = ignore) excludes ocean, where ERA5-Land
    has no truth. Masked cells are zero-filled upstream, so without this the
    network would be trained to predict zeros over water.
    """
    if mask is not None:
        m = mask if mask.dim() == pred.dim() else mask.unsqueeze(1)
        denom = m.sum().clamp_min(1.0)
        diff = (pred - target) * m
        pix = diff.abs().sum() / denom if base == "l1" \
            else (diff ** 2).sum() / denom
        # SSIM over masked input is ill-defined per-window; apply it to the
        # masked fields, which is exact wherever a window is fully on land and
        # a mild approximation only in the coastline windows.
        struct = 1.0 - ssim(pred * m, target * m)
    else:
        pix = F.l1_loss(pred, target) if base == "l1" \
            else F.mse_loss(pred, target)
        struct = 1.0 - ssim(pred, target)
    return pix + ssim_weight * struct

