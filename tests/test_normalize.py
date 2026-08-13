"""Normalizer: round-trip exactness, NaN handling, persistence."""
from __future__ import annotations

import numpy as np
import pytest

from common.normalize import Normalizer


def test_transform_standardizes_to_zero_mean_unit_std(rng):
    v = rng.normal(12.0, 3.0, size=5000)
    nz = Normalizer()
    nz.fit("t", v)
    z = nz.transform("t", v)
    assert abs(float(z.mean())) < 1e-6
    assert abs(float(z.std()) - 1.0) < 1e-4


def test_inverse_round_trips(rng):
    v = rng.normal(12.0, 3.0, size=1000)
    nz = Normalizer()
    nz.fit("t", v)
    assert np.allclose(nz.inverse("t", nz.transform("t", v)), v, atol=1e-6)


def test_log1p_round_trips_on_skewed_nonnegative_data(rng):
    v = rng.exponential(2.0, size=1000)
    nz = Normalizer()
    nz.fit("p", v, log1p=True)
    assert np.allclose(nz.inverse("p", nz.transform("p", v)), v, atol=1e-5)


def test_fit_ignores_nans():
    """Ocean cells are NaN in ERA5-Land; they must not poison the stats."""
    clean = np.array([1.0, 2.0, 3.0, 4.0])
    dirty = np.array([1.0, np.nan, 2.0, 3.0, np.inf, 4.0])
    a, b = Normalizer(), Normalizer()
    a.fit("t", clean)
    b.fit("t", dirty)
    assert a.stats["t"]["mean"] == pytest.approx(b.stats["t"]["mean"])
    assert a.stats["t"]["std"] == pytest.approx(b.stats["t"]["std"])


def test_zero_variance_channel_does_not_divide_by_zero():
    nz = Normalizer()
    nz.fit("const", np.full(100, 5.0))
    assert np.all(np.isfinite(nz.transform("const", np.full(10, 5.0))))


def test_season_keys_are_independent():
    nz = Normalizer()
    nz.fit("t", np.array([0.0, 10.0]), season="djf")
    nz.fit("t", np.array([20.0, 30.0]), season="jja")
    assert nz.stats["t__djf"]["mean"] != nz.stats["t__jja"]["mean"]


def test_save_load_preserves_stats(tmp_path, rng):
    v = rng.normal(size=100)
    nz = Normalizer()
    nz.fit("t", v)
    path = tmp_path / "norm_stats.json"
    nz.save(path)
    assert Normalizer.load(path).stats == nz.stats


def test_transform_on_unfitted_variable_raises():
    """Silently normalizing with the wrong constants is worse than crashing."""
    with pytest.raises(KeyError):
        Normalizer().transform("never_fit", np.zeros(3))
