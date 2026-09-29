"""Tests for the pieces the map figures rest their claims on.

Every number printed on budget_*.png, coherence_*.png and improvement_*.png
comes out of one of four functions. None of them is complicated, and that is
exactly why they are worth pinning: a sign error in `decompose` or a factor of
sqrt(2) in `sigma_for_cutoff` would not crash, would not look wrong on a map,
and would change what the paper claims.

No store, no checkpoint, no GPU — everything here is synthetic and offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.fields import cellwise_corr, decompose, highpass  # noqa: E402
from scripts.make_coherence_panels import sigma_for_cutoff  # noqa: E402


def _mask(ny=12, nx=15):
    m = np.zeros((ny, nx), bool)
    m[2:10, 3:13] = True
    return m


# --------------------------------------------------------------------------- #
# decompose — the 5.4 error budget
# --------------------------------------------------------------------------- #

def test_decompose_recovers_a_known_construction():
    """Build resid from known mu, s and eps; check all three come back exactly.

    The construction has to satisfy the same constraints the decomposition
    imposes, or no method could separate the terms: s is mean-zero over the
    mask, and eps is mean-zero both across the mask on every day and through
    time in every cell. Double-centring eps gives both properties exactly, so
    this is an identity test, not an approximation test.
    """
    rng = np.random.default_rng(0)
    m = _mask()
    T = 400
    mu = rng.normal(0, 0.6, T)
    s = rng.normal(0, 0.15, m.shape)
    s -= s[m].mean()

    e = rng.normal(0, 0.5, (T, *m.shape))
    e -= e[:, m].mean(1)[:, None, None]            # zero spatial mean each day
    e -= np.where(m, e.mean(0), 0.0)[None]         # zero time mean each cell
    assert np.abs(e[:, m].mean(1)).max() < 1e-12
    assert np.abs(e[:, m].mean(0)).max() < 1e-12

    resid = mu[:, None, None] + s[None] + e
    got_mu, got_s, got_eps, sh = decompose(resid, m)

    assert np.allclose(got_mu, mu, atol=1e-10)
    assert np.allclose(got_s[m], s[m], atol=1e-10)
    assert np.allclose(got_eps[:, m], e[:, m], atol=1e-10)

    tot = mu.var() + s[m].var() + e[:, m].var()
    assert sh["offset"] == pytest.approx(mu.var() / tot)
    assert sh["static"] == pytest.approx(s[m].var() / tot)
    assert sh["remainder"] == pytest.approx(e[:, m].var() / tot)
    # sd 0.6 / 0.15 / 0.5 puts the offset term just over half, as in 5.4.
    assert 0.48 < sh["offset"] < 0.58
    assert sh["static"] < 0.06


def test_decompose_shares_sum_to_one():
    rng = np.random.default_rng(1)
    m = _mask()
    resid = rng.normal(0, 1, (120, *m.shape))
    _, _, _, sh = decompose(resid, m)
    assert sh["offset"] + sh["static"] + sh["remainder"] == pytest.approx(1.0)


def test_decompose_terms_are_orthogonal():
    """The three terms must not share variance, or the shares double count."""
    rng = np.random.default_rng(2)
    m = _mask()
    resid = rng.normal(0, 1, (200, *m.shape)) + rng.normal(0, 1, (200, 1, 1))
    mu, s, eps, _ = decompose(resid, m)

    # eps has zero spatial mean on every day and zero time mean in every cell.
    assert np.allclose(eps[:, m].mean(1), 0, atol=1e-10)
    assert np.allclose(eps[:, m].mean(0), 0, atol=1e-10)
    # mu and s are therefore uncorrelated with it by construction.
    assert np.allclose(s[m].mean(), 0, atol=1e-10)


def test_decompose_mu_is_exactly_the_masked_daily_mean():
    rng = np.random.default_rng(3)
    m = _mask()
    resid = rng.normal(0, 1, (50, *m.shape))
    mu, _, _, _ = decompose(resid, m)
    assert np.allclose(mu, resid[:, m].mean(1))


def test_decompose_ignores_cells_outside_the_mask():
    """Garbage off-mask must not move any quoted number."""
    rng = np.random.default_rng(4)
    m = _mask()
    resid = rng.normal(0, 1, (80, *m.shape))
    a = decompose(resid, m)[3]
    resid[:, ~m] = 1e6
    b = decompose(resid, m)[3]
    assert a == pytest.approx(b)


# --------------------------------------------------------------------------- #
# sigma_for_cutoff / highpass — the 5.5 and 5.8 filter
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("lam_km", [40.0, 60.0, 120.0])
def test_highpass_passes_half_the_amplitude_at_the_stated_wavelength(lam_km):
    """A wave of exactly lam_km must come through at half amplitude.

    This is the claim the figure's caption makes. A factor of sqrt(2) here —
    the difference between half amplitude and half power — would quietly move
    the stated resolution of every panel.
    """
    dx = 10.0
    n = 512                                   # long, so edge handling is moot
    x = np.arange(n) * dx
    wave = np.sin(2 * np.pi * x / lam_km)
    field = np.tile(wave, (48, 1))            # constant down-column
    out = highpass(field, sigma_for_cutoff(lam_km, dx))
    ratio = out[24, n // 4:-n // 4].std() / wave[n // 4:-n // 4].std()
    assert ratio == pytest.approx(0.5, abs=0.02)


def test_highpass_attenuates_long_waves_more_than_short_ones():
    dx, n = 10.0, 512
    x = np.arange(n) * dx
    sig = sigma_for_cutoff(60.0, dx)
    ratios = []
    for lam in (300.0, 60.0, 25.0):
        f = np.tile(np.sin(2 * np.pi * x / lam), (32, 1))
        ratios.append(highpass(f, sig)[16].std() / f[16].std())
    assert ratios[0] < ratios[1] < ratios[2]
    assert ratios[0] < 0.1 and ratios[2] > 0.9


def test_highpass_kills_a_constant_field():
    f = np.full((30, 40), 7.5)
    assert np.abs(highpass(f, 1.5)).max() < 1e-5


def test_highpass_2d_and_3d_agree():
    rng = np.random.default_rng(5)
    stack = rng.normal(0, 1, (4, 20, 25))
    got = highpass(stack, 1.12)
    for k in range(4):
        assert np.allclose(got[k], highpass(stack[k], 1.12), atol=1e-6)


def test_highpass_edges_do_not_manufacture_signal():
    """'nearest' edges: a linear ramp must not light up the rim.

    Zero padding would put a large step at the boundary and the high-pass would
    read it as real fine-scale structure — a bright frame on every panel.
    """
    ramp = np.tile(np.linspace(0, 20, 60), (40, 1))
    out = np.abs(highpass(ramp, 1.12))
    assert out[:, :2].max() < 0.5
    assert out[:, -2:].max() < 0.5


# --------------------------------------------------------------------------- #
# cellwise_corr — the coherence maps
# --------------------------------------------------------------------------- #

def test_cellwise_corr_is_one_for_identical_stacks():
    rng = np.random.default_rng(6)
    m = _mask()
    a = rng.normal(0, 1, (60, *m.shape))
    r = cellwise_corr(a, a, m)
    assert np.allclose(r[m], 1.0, atol=1e-6)


def test_cellwise_corr_is_minus_one_for_negated_stacks():
    rng = np.random.default_rng(7)
    m = _mask()
    a = rng.normal(0, 1, (60, *m.shape))
    assert np.allclose(cellwise_corr(-a, a, m)[m], -1.0, atol=1e-6)


def test_cellwise_corr_is_nan_off_the_mask():
    rng = np.random.default_rng(8)
    m = _mask()
    a = rng.normal(0, 1, (40, *m.shape))
    assert np.isnan(cellwise_corr(a, a, m)[~m]).all()


def test_cellwise_corr_is_invariant_to_per_cell_offset_and_scale():
    """It must measure phase, not amplitude — that is the whole point of
    separating coherence from the amplitude ratio."""
    rng = np.random.default_rng(9)
    m = _mask()
    a = rng.normal(0, 1, (80, *m.shape))
    b = 3.7 * a + 12.0
    assert np.allclose(cellwise_corr(b, a, m)[m], 1.0, atol=1e-6)


def test_cellwise_corr_matches_numpy_on_one_cell():
    rng = np.random.default_rng(10)
    m = _mask()
    a = rng.normal(0, 1, (100, *m.shape))
    b = rng.normal(0, 1, (100, *m.shape))
    got = cellwise_corr(a, b, m)
    i, j = 4, 6
    assert m[i, j]
    assert got[i, j] == pytest.approx(np.corrcoef(a[:, i, j], b[:, i, j])[0, 1])


def test_cellwise_corr_returns_nan_on_a_constant_cell():
    """Zero variance has no correlation; it must be NaN, not 0 or a crash."""
    m = _mask()
    a = np.random.default_rng(11).normal(0, 1, (30, *m.shape))
    b = a.copy()
    b[:, 4, 6] = 2.0
    assert np.isnan(cellwise_corr(a, b, m)[4, 6])
