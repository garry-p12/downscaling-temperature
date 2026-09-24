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
    deliberate, and that overlap is stated in README.md.

    ``base='amse'`` swaps the pixel term for the spectrally adjusted MSE below,
    which attacks the same blurring at its root rather than by preferring a
    gentler norm. Note that all three are PIXEL-space or SPECTRAL fidelity
    terms occupying the same slot: amse replaces l1, it does not stack with it.

    ``mask`` (1 = score this cell, 0 = ignore) excludes ocean, where ERA5-Land
    has no truth. Masked cells are zero-filled upstream, so without this the
    network would be trained to predict zeros over water.
    """
    if mask is not None:
        m = mask if mask.dim() == pred.dim() else mask.unsqueeze(1)
        denom = m.sum().clamp_min(1.0)
        diff = (pred - target) * m
        if base == "amse":
            pix = amse(pred, target, mask=m)
        elif base == "l1":
            pix = diff.abs().sum() / denom
        else:
            pix = (diff ** 2).sum() / denom
        # SSIM over masked input is ill-defined per-window; apply it to the
        # masked fields, which is exact wherever a window is fully on land and
        # a mild approximation only in the coastline windows.
        struct = 1.0 - ssim(pred * m, target * m)
    else:
        if base == "amse":
            pix = amse(pred, target)
        else:
            pix = F.l1_loss(pred, target) if base == "l1" \
                else F.mse_loss(pred, target)
        struct = 1.0 - ssim(pred, target)
    return pix + ssim_weight * struct



# --------------------------------------------------------------------------- #
# AMSE — spectrally adjusted mean squared error
# --------------------------------------------------------------------------- #
# Subich et al. 2025, "Fixing the Double Penalty in Data-Driven Weather
# Forecasting Through a Modified Spherical Harmonic Loss Function" (ICML).
#
# WHY THIS EXISTS
#   Under MSE, if a scale is only partially predictable (correlation rho < 1),
#   the loss-optimal amplitude is sigma = rho < 1. The model therefore MINIMISES
#   ITS LOSS BY SHRINKING the amplitude of anything it cannot predict, rather
#   than by predicting it. Measured on this project's Austin holdout, the V2
#   DeepSD model's amplitude ratio tracks its coherence down the whole spectrum:
#
#       wavelength   amplitude ratio   coherence
#          495 km         0.958          0.985
#          165 km         0.800          0.568
#           99 km         0.629          0.408
#           55 km         0.472          0.247
#           38 km         0.421          0.108
#
#   By the paper's criterion (amplitude ratio below sqrt(0.75)) the effective
#   resolution is ~165 km on a 10 km grid — a 16x gap. L1 smooths less than MSE
#   but still smooths (paper appendix B.5), which is where the residual gap
#   between the two columns above comes from.
#
# THE FIX
#   Write MSE in spectral space and note that the decorrelation term is coupled
#   to the amplitudes through their geometric mean:
#       MSE = sum_k (sqrt(PSDx) - sqrt(PSDy))^2 + 2 sqrt(PSDx PSDy)(1 - Coh)
#   Shrinking PSDx therefore buys a discount on the decorrelation penalty.
#   Replacing that coupling with max() removes the incentive while keeping the
#   loss zero iff x == y:
#       AMSE = sum_k (sqrt(PSDx) - sqrt(PSDy))^2 + 2 max(PSDx, PSDy)(1 - Coh)
#   It is parameter-free and a drop-in for MSE.
#
# ADAPTATION TO THIS PROJECT
#   The paper uses spherical harmonics on a global lat/lon grid. This is a
#   limited-area 10 km Albers grid trained on 48x48 patches, so the multiscale
#   decomposition is a 2D DFT with radial binning by |k| — the paper's section
#   4.2 anticipates exactly this substitution. Two consequences:
#     * The patch is NOT periodic, so a raw FFT leaks the edge discontinuity
#       into high wavenumbers and would read as false sharpness. A Hann window
#       suppresses that. It also down-weights patch edges in the gradient;
#       with random patch origins every pixel is an edge sometimes, so this
#       averages out over training.
#     * Ocean is zero-filled. The same mask is applied to both fields, so the
#       leakage it creates is common-mode and largely cancels in the ratio and
#       the coherence.

_BIN_CACHE: dict = {}


def _radial_bins(h: int, w: int, device, dtype):
    """(bin index per Fourier mode, n_bins). Cached — it depends only on shape."""
    key = (h, w, str(device))
    if key not in _BIN_CACHE:
        ky = torch.fft.fftfreq(h, device=device).view(-1, 1)
        kx = torch.fft.fftfreq(w, device=device).view(1, -1)
        kr = torch.sqrt(ky ** 2 + kx ** 2)
        nb = max(2, min(h, w) // 2)
        idx = (kr / (kr.max() + 1e-12) * nb).long().clamp_(max=nb - 1)
        _BIN_CACHE[key] = (idx.reshape(-1), nb)
    return _BIN_CACHE[key]


def _hann2d(h: int, w: int, device, dtype):
    key = ("hann", h, w, str(device), str(dtype))
    if key not in _BIN_CACHE:
        wy = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
        wx = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
        _BIN_CACHE[key] = torch.outer(wy, wx).view(1, 1, h, w)
    return _BIN_CACHE[key]


def amse(pred: torch.Tensor, target: torch.Tensor,
         mask: torch.Tensor | None = None, window: bool = True,
         pool_batch: bool = True, eps: float = 1e-10) -> torch.Tensor:
    """Spectrally adjusted MSE over a batch of (B, 1, H, W) tensors.

    Normalised to sit on the same scale as ``F.mse_loss`` so the existing loss
    weights and learning rate transfer without retuning.

    ``pool_batch`` averages each bin's spectral statistics across the batch
    before forming the max() and the coherence. This matters at our patch size.
    The paper computes its spectra over a full global field, where every
    wavenumber bin holds many modes; a 48x48 patch holds few, so the per-bin
    PSD estimate is noisy, and max(PSDx, PSDy) then picks the larger of two
    noisy numbers — an upward bias that is worst exactly where the amplitudes
    are correct. Measured on a synthetic sweep at correlation 0.5, the loss
    minimum sits at amplitude 0.90 per-sample and 0.95 pooled, against the
    ideal of 1.0 and against 0.50 for plain MSE. So pooling recovers most of
    the remaining bias, and what is left is conservative: it under-sharpens
    slightly rather than over-sharpening into noise.
    """
    b, _, h, w = pred.shape
    if mask is not None:
        m = mask if mask.dim() == pred.dim() else mask.unsqueeze(1)
        pred, target = pred * m, target * m

    if window:
        win = _hann2d(h, w, pred.device, pred.dtype)
        pred, target = pred * win, target * win
        wnorm = (win ** 2).mean()
    else:
        wnorm = pred.new_ones(())

    # Full fft2 rather than rfft2: the Hermitian half-spectrum needs per-column
    # doubling to satisfy Parseval, and getting that subtly wrong would bias the
    # high-wavenumber bins — the exact thing this loss is meant to measure.
    fp = torch.fft.fft2(pred.squeeze(1))
    ft = torch.fft.fft2(target.squeeze(1))
    idx, nb = _radial_bins(h, w, pred.device, pred.dtype)

    pp = (fp.real ** 2 + fp.imag ** 2).reshape(b, -1)
    tt = (ft.real ** 2 + ft.imag ** 2).reshape(b, -1)
    pt = (fp.real * ft.real + fp.imag * ft.imag).reshape(b, -1)

    z = pred.new_zeros((b, nb))
    ix = idx.unsqueeze(0).expand(b, -1)
    psd_p = z.scatter_add(1, ix, pp)
    psd_t = z.clone().scatter_add(1, ix, tt)
    cross = z.clone().scatter_add(1, ix, pt)
    if pool_batch:
        psd_p = psd_p.mean(0, keepdim=True)
        psd_t = psd_t.mean(0, keepdim=True)
        cross = cross.mean(0, keepdim=True)

    coh = cross / torch.sqrt(psd_p * psd_t + eps)
    amp = (torch.sqrt(psd_p + eps) - torch.sqrt(psd_t + eps)) ** 2
    decoh = 2.0 * torch.maximum(psd_p, psd_t) * (1.0 - coh)

    # Parseval: sum_ij r^2 == (1/HW) sum_k |F_k|^2, and mse_loss divides by HW
    # again, hence (HW)^2. Dividing out the window energy keeps the magnitude
    # comparable to an unwindowed MSE.
    return (amp + decoh).sum(1).mean() / ((h * w) ** 2 * wnorm)


# --------------------------------------------------------------------------- #
# Error-budget decomposed loss
# --------------------------------------------------------------------------- #
# This is the idea taken from Subich et al. 2025, rather than their loss.
#
# Their contribution is not the max() trick; it is the observation that a single
# scalar loss CONFLATES error modes with different fixability, and then spends
# the model's capacity on whichever mode is cheapest to reduce — which is not
# the one you care about. Their split is amplitude vs phase, per scale, because
# that is what matters for a forecast. Ours is different, and README section 5.4
# already measured it:
#
#     spatially-uniform daily offset   50.8% of holdout residual variance
#     time-invariant spatial pattern    2.8%
#     space-time remainder             46.4%
#
# Half the error we score is one number per day applied to the whole region — a
# product-level disagreement between POWER and ERA5-Land. No spatial model can
# fix it, but the loss keeps charging for it, so gradient that could be buying
# spatial structure is spent chasing a bias instead. Section 5.4 also showed the
# offset is separately predictable from its own lag-1 (out-of-sample R^2 +0.221,
# better than anything spatial), so it belongs in a different model entirely.
#
# This loss makes that split explicit and lets the offset term be down-weighted:
#
#     loss = spatial(pred - mu_p, target - mu_t)  +  offset_weight * |mu_p - mu_t|
#
# offset_weight = 1.0 is close to the existing behaviour; 0.0 tells the model to
# ignore the daily bias completely and spend everything on pattern, leaving the
# offset to be restored at inference by the AR(1) term.
#
# WHY THE MEAN IS TAKEN OVER THE WHOLE FIELD, NOT A PATCH
#   Measured on the training split: the offset has sd 0.463 degC, while the
#   difference between a 48x48 patch mean and the true domain mean has sd 0.376
#   — 81% of the signal. Subtracting a PATCH mean would therefore remove mostly
#   real spatial structure rather than the bias. This loss is only meaningful on
#   full fields (patch_size: null), which costs 1.01x the pixels per epoch since
#   there are correspondingly fewer steps.


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Per-sample mean over the mask, shaped to broadcast back onto (B,1,H,W)."""
    dims = (1, 2, 3)
    return (x * m).sum(dims, keepdim=True) / m.sum(dims, keepdim=True).clamp_min(1.0)


def decomposed_loss(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor | None = None,
                    offset_weight: float = 1.0, ssim_weight: float = 0.2,
                    base: str = "l1") -> torch.Tensor:
    """Pixel/structure loss on the de-meaned field + a separate offset term."""
    if mask is None:
        mask = torch.ones_like(pred)
    m = mask if mask.dim() == pred.dim() else mask.unsqueeze(1)

    mu_p, mu_t = _masked_mean(pred, m), _masked_mean(target, m)
    offset = (mu_p - mu_t).abs().mean()

    pc, tc = (pred - mu_p) * m, (target - mu_t) * m
    denom = m.sum().clamp_min(1.0)
    if base == "amse":
        spatial = amse(pc, tc, mask=m)
    elif base == "l1":
        spatial = (pc - tc).abs().sum() / denom
    else:
        spatial = ((pc - tc) ** 2).sum() / denom
    struct = 1.0 - ssim(pc, tc)
    return spatial + ssim_weight * struct + offset_weight * offset
