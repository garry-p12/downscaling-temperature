"""Tests for the resolution summaries.

The old `effective_resolution` returned the centre of the first band below the
amplitude floor, which made it blind: on a 37x35 crop the bands above 60 km are
495, 165, 99 and 71 km, so four arms with visibly different curves all scored
exactly 165. These pin the replacement — that it interpolates, that it is not
quantised to band centres, and that the coherence variant reports "never
crossed" rather than inventing a number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.spectral import (AMP_FLOOR, coherence_resolution,   # noqa: E402
                                 effective_resolution, radial_spectrum,
                                 summarise_spectrum)


def rows(pairs, coh=None):
    """Build a curve from (wavelength, amplitude) pairs, coarse first."""
    return [{"wavelength_km": w, "amplitude_ratio": a,
             "coherence": (coh[i] if coh else 1.0)}
            for i, (w, a) in enumerate(pairs)]


# --------------------------------------------------------------------------- #
# the bug this replaced
# --------------------------------------------------------------------------- #

def test_interpolates_rather_than_snapping_to_a_band_centre():
    """The real case: 0.88 at 495 km, 0.78 at 165 km, floor 0.866.

    Banded mode returns 165 — the band centre. The crossing is actually just
    below 495, because the curve only dips under the floor a little way down.
    """
    r = rows([(495, 0.88), (165, 0.78), (99, 0.55), (71, 0.46)])
    banded = effective_resolution(r, interpolate=False)
    interp = effective_resolution(r)
    assert banded == 165
    assert 165 < interp < 495
    assert interp == pytest.approx(405, abs=25)


def test_distinguishes_curves_the_banded_version_cannot():
    """Two arms that both cross between the same pair of bands must differ."""
    a = rows([(495, 0.88), (165, 0.78)])     # crosses just below 495
    b = rows([(495, 0.95), (165, 0.70)])     # stays high longer, crosses lower
    assert effective_resolution(a, interpolate=False) == \
           effective_resolution(b, interpolate=False)          # the old blindness
    assert effective_resolution(b) < effective_resolution(a)   # the fix


def test_crossing_is_exact_when_a_band_sits_on_the_floor():
    r = rows([(400, 0.90), (200, AMP_FLOOR), (100, 0.50)])
    assert effective_resolution(r) == pytest.approx(200, rel=1e-6)


# --------------------------------------------------------------------------- #
# behaviour at the edges
# --------------------------------------------------------------------------- #

def test_none_when_never_below_the_floor():
    assert effective_resolution(rows([(400, 0.99), (200, 0.95), (100, 0.90)])) is None


def test_first_crossing_wins_despite_a_later_rise():
    """Amplitude rises again near Nyquist where the model injects noise; the
    summary must not report that as recovered skill."""
    r = rows([(400, 0.95), (200, 0.60), (100, 0.55), (50, 0.95), (25, 1.10)])
    v = effective_resolution(r)
    assert 200 < v < 400


def test_coherence_resolution_reports_none_when_it_never_drops():
    """The AORC-trained arms never fall below 0.5 in the measured range. That
    is a lower bound on skill and must not be rendered as a number."""
    r = rows([(400, .9), (200, .9), (100, .9)], coh=[0.82, 0.75, 0.70])
    assert coherence_resolution(r) is None


def test_coherence_resolution_interpolates():
    r = rows([(400, .9), (200, .9), (100, .9)], coh=[0.80, 0.60, 0.40])
    v = coherence_resolution(r, level=0.5)
    assert 100 < v < 200


# --------------------------------------------------------------------------- #
# the summary that replaces the single number
# --------------------------------------------------------------------------- #

def test_summary_carries_both_crossings_and_the_curve():
    r = rows([(400, 0.95), (200, 0.70), (55, 0.40), (38, 0.30)],
             coh=[0.8, 0.6, 0.3, 0.2])
    s = summarise_spectrum(r)
    assert set(s) == {"eff_res_km", "eff_res_km_banded", "coh_res_km",
                      "amp_below_60km", "coh_below_60km", "curve"}
    assert len(s["curve"]) == len(r)
    # band means use only the sub-60 km entries
    assert s["amp_below_60km"] == pytest.approx((0.40 + 0.30) / 2)
    assert s["coh_below_60km"] == pytest.approx((0.3 + 0.2) / 2)


# --------------------------------------------------------------------------- #
# end to end on a field with known content
# --------------------------------------------------------------------------- #

def test_on_a_smoothed_field_the_resolution_is_coarser_than_the_grid():
    """A deliberately blurred copy must score worse than an exact copy."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    truth = rng.normal(0, 1, (40, 48, 48)).astype("float32")
    truth = np.stack([gaussian_filter(t, 1.0) for t in truth])      # give it structure
    blur = np.stack([gaussian_filter(t, 2.5) for t in truth])

    # dx_km is explicit because radial_spectrum no longer defaults it: the
    # old 10.0 default was "whatever grid the project was on" and reported
    # every wavelength 2x too coarse the moment that changed.
    exact = effective_resolution(radial_spectrum(truth, truth, dx_km=10.0))
    smooth = effective_resolution(radial_spectrum(blur, truth, dx_km=10.0))
    assert exact is None                       # identical fields never cross
    assert smooth is not None and smooth > 0   # the blurred one does
