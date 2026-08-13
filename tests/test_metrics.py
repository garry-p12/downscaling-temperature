"""Metrics. These decide the reported result, so their edge cases matter more
than the model's."""
from __future__ import annotations

import numpy as np
import pytest

from training.metrics import (
    bias,
    extreme_bias,
    mae,
    power_spectrum,
    residual_spatial_corr,
    rmse,
    spatial_corr,
    ssim,
    temperature_metrics,
)


def test_perfect_prediction_scores_perfectly(rng):
    x = rng.normal(size=(4, 16, 16))
    assert rmse(x, x) == pytest.approx(0.0)
    assert mae(x, x) == pytest.approx(0.0)
    assert bias(x, x) == pytest.approx(0.0)
    assert spatial_corr(x, x) == pytest.approx(1.0, abs=1e-6)
    assert ssim(x, x) == pytest.approx(1.0, abs=1e-3)


def test_rmse_matches_the_closed_form():
    pred = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    truth = np.zeros_like(pred)
    assert rmse(pred, truth) == pytest.approx(np.sqrt(30 / 4))


def test_bias_is_signed_and_mae_is_not():
    pred = np.array([[[1.0, -3.0]]])
    truth = np.zeros_like(pred)
    assert bias(pred, truth) == pytest.approx(-1.0)
    assert mae(pred, truth) == pytest.approx(2.0)


def test_nans_are_skipped_not_propagated():
    """Ocean cells are NaN; a single one must not blank the whole metric."""
    pred = np.array([[[1.0, np.nan], [1.0, 1.0]]])
    truth = np.zeros_like(pred)
    assert rmse(pred, truth) == pytest.approx(1.0)


def test_constant_offset_leaves_spatial_corr_intact(rng):
    """spatial_corr must measure pattern, not level — that is why bias is
    reported separately."""
    truth = rng.normal(size=(3, 20, 20))
    assert spatial_corr(truth + 5.0, truth) == pytest.approx(1.0, abs=1e-6)


def test_residual_corr_is_harsher_than_raw_corr(rng):
    """The headline claim in RESULTS.md rests on this: removing the baseline
    strips the large-scale structure that inflates raw correlation."""
    truth = rng.normal(size=(6, 24, 24))
    base = truth + 0.5 * rng.normal(size=truth.shape)      # a decent baseline
    pred = base + 0.5 * rng.normal(size=truth.shape)       # no extra skill
    assert residual_spatial_corr(pred, truth, base) < spatial_corr(pred, truth)


def test_ssim_penalizes_blur(rng):
    """The over-smoothing check: a blurred field must not beat the sharp one."""
    from scipy.ndimage import gaussian_filter

    truth = rng.normal(size=(2, 32, 32))
    blurred = gaussian_filter(truth, sigma=(0, 2, 2))
    assert ssim(blurred, truth) < ssim(truth, truth)


def test_extreme_bias_reports_both_tails(rng):
    m = extreme_bias(rng.normal(size=(4, 16, 16)), rng.normal(size=(4, 16, 16)))
    assert m and all(np.isfinite(v) for v in m.values())


def test_power_spectrum_of_a_blurred_field_loses_high_wavenumber_power(rng):
    from scipy.ndimage import gaussian_filter

    field = rng.normal(size=(32, 32))
    k, ps_sharp = power_spectrum(field)
    _, ps_blur = power_spectrum(gaussian_filter(field, sigma=2))
    assert len(k) == len(ps_sharp) == len(ps_blur)
    hi = slice(len(k) // 2, None)
    assert np.nansum(ps_blur[hi]) < np.nansum(ps_sharp[hi])


def test_temperature_metrics_reports_the_full_block(rng):
    m = temperature_metrics(rng.normal(size=(3, 12, 12)),
                            rng.normal(size=(3, 12, 12)))
    for key in ("rmse", "mae", "bias", "spatial_corr", "ssim"):
        assert key in m and np.isfinite(m[key])
