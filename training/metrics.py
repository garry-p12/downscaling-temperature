"""Evaluation metrics (numpy). Operate on denormalized physical fields.

RMSE, MAE, bias, spatial correlation, SSIM, extreme bias, and a
radially-averaged power spectrum for the over-smoothing check.
"""
from __future__ import annotations

import numpy as np


def rmse(pred, target) -> float:
    return float(np.sqrt(np.nanmean((pred - target) ** 2)))


def mae(pred, target) -> float:
    return float(np.nanmean(np.abs(pred - target)))


def bias(pred, target) -> float:
    return float(np.nanmean(pred - target))


def spatial_corr(pred, target) -> float:
    """Pearson correlation over space, averaged across samples.

    NaN-aware throughout. Masked cells (ocean, and the spatial holdout during
    validation) arrive as NaN, and a plain ``.mean()`` propagates a single NaN
    across the whole sample — which silently returns NaN for the entire metric
    rather than scoring the valid cells.
    """
    p = pred.reshape(pred.shape[0], -1).astype("float64")
    t = target.reshape(target.shape[0], -1).astype("float64")
    valid = np.isfinite(p) & np.isfinite(t)
    p = np.where(valid, p, np.nan)
    t = np.where(valid, t, np.nan)
    p = p - np.nanmean(p, axis=1, keepdims=True)
    t = t - np.nanmean(t, axis=1, keepdims=True)
    num = np.nansum(p * t, axis=1)
    den = np.sqrt(np.nansum(p ** 2, axis=1) * np.nansum(t ** 2, axis=1)) + 1e-12
    corr = num / den
    corr[valid.sum(axis=1) < 2] = np.nan          # too few cells to correlate
    return float(np.nanmean(corr)) if np.isfinite(corr).any() else float("nan")


def residual_spatial_corr(pred, target, baseline) -> float:
    """Spatial correlation of the fields *after* removing a baseline.

    corr(pred - baseline, target - baseline). Subtracting a strong baseline
    (bilinear, or lapse-rate) strips the large-scale structure that inflates
    raw spatial_corr toward ~0.99 for free, leaving only the fine-scale skill
    the model is actually responsible for. This is the discriminating number.
    """
    return spatial_corr(pred - baseline, target - baseline)


def power_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged 2-D power spectrum of a single (H, W) field.

    Comparing model vs. truth spectra reveals a blurred field: a deterministic
    regressor loses power at high wavenumbers (fine scales).
    """
    f = np.fft.fftshift(np.fft.fft2(field - np.nanmean(field)))
    psd2d = np.abs(f) ** 2
    h, w = field.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), psd2d.ravel())
    nr = np.bincount(r.ravel())
    radial = tbin / np.maximum(nr, 1)
    k = np.arange(len(radial))
    return k, radial


def _as_2d_stack(a) -> np.ndarray:
    """(N, H, W) from either (N, H, W) or (N, 1, H, W)."""
    a = np.asarray(a)
    if a.ndim == 4:
        if a.shape[1] != 1:
            raise ValueError(f"expected a single channel, got shape {a.shape}")
        a = a[:, 0]
    if a.ndim != 3:
        raise ValueError(f"expected (N, H, W), got shape {a.shape}")
    return a


def ssim(pred, target, ws: int = 11, sigma: float = 1.5,
         data_range: float | None = None) -> float:
    """Mean structural similarity over a batch of 2-D fields.

    Structure-aware in a way RMSE is not: SSIM compares local means, variances
    and covariance, so a field that is numerically close but has lost its
    texture scores poorly even when its RMSE is good.

    ``data_range`` defaults to the range of the TARGET field per sample. That
    matters for temperature: SSIM is scale-dependent, and a fixed range would
    make winter days (wide spatial spread) and summer days score
    incomparably. Normalizing by each day's own truth range keeps samples
    commensurate.

    Mirrors the Gaussian-window formulation in training/losses.py so the
    reported value is consistent with the training objective.
    """
    from scipy.ndimage import gaussian_filter

    p, t = _as_2d_stack(pred), _as_2d_stack(target)
    trunc = ((ws - 1) / 2) / sigma          # scipy window == ws pixels wide
    g = lambda a: gaussian_filter(a, sigma, truncate=trunc, mode="nearest")  # noqa: E731

    out = []
    for pi, ti in zip(p, t):
        pi = pi.astype("float64")
        ti = ti.astype("float64")
        dr = float(np.nanmax(ti) - np.nanmin(ti)) if data_range is None \
            else float(data_range)
        if dr <= 0:
            continue
        c1, c2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
        mu_p, mu_t = g(pi), g(ti)
        mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
        sig_p = g(pi * pi) - mu_p2
        sig_t = g(ti * ti) - mu_t2
        sig_pt = g(pi * ti) - mu_pt
        s = ((2 * mu_pt + c1) * (2 * sig_pt + c2)) / \
            ((mu_p2 + mu_t2 + c1) * (sig_p + sig_t + c2))
        out.append(float(np.nanmean(s)))
    return float(np.mean(out)) if out else float("nan")


def extreme_bias(pred, target, q=(5, 95)) -> dict:
    """Bias at distribution tails, in degC.

    RMSE can look healthy while extremes are badly smoothed — an MSE/L1-trained
    model hedges toward the conditional mean, which compresses both tails.
    Heat waves and cold snaps are usually what a downscaled product is FOR, so
    tail bias is reported alongside the mean-based metrics.
    """
    out = {}
    for qq in q:
        pq = float(np.nanpercentile(pred, qq))
        tq = float(np.nanpercentile(target, qq))
        out[f"p{qq}_bias"] = pq - tq
        out[f"p{qq}_pred"] = pq
        out[f"p{qq}_truth"] = tq
    return out


def temperature_metrics(pred, target) -> dict:
    m = dict(rmse=rmse(pred, target), mae=mae(pred, target),
             bias=bias(pred, target), spatial_corr=spatial_corr(pred, target),
             ssim=ssim(pred, target))
    m.update(extreme_bias(pred, target))
    return m

