"""AMSE, the spectrally adjusted MSE.

The point of this loss is a behavioural claim, not an algebraic one: under MSE a
model minimises its loss by shrinking the amplitude of whatever it cannot
predict, and AMSE removes that incentive. These tests check the claim directly
rather than checking that some numbers come out of a function.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from training.losses import _radial_bins, amse, temp_loss  # noqa: E402


def _partially_correlated(target, noise, rho, sigma):
    """A prediction with correlation rho to target, scaled to amplitude sigma."""
    return sigma * (rho * target + (1 - rho ** 2) ** 0.5 * noise)


# --------------------------------------------------------------------------- #
# Algebraic properties the paper requires
# --------------------------------------------------------------------------- #
def test_zero_iff_identical():
    x = torch.randn(4, 1, 32, 32)
    assert amse(x, x).item() == pytest.approx(0.0, abs=1e-9)
    assert amse(x, torch.randn_like(x)).item() > 0.0


def test_parseval_normalisation_reproduces_mse():
    """Rebuild plain MSE through the same spectral machinery. If this drifts,
    the normalisation is wrong and every AMSE magnitude is off by a constant."""
    torch.manual_seed(0)
    p, t = torch.randn(4, 1, 32, 32), torch.randn(4, 1, 32, 32)
    b, _, h, w = p.shape
    fp, ft = torch.fft.fft2(p.squeeze(1)), torch.fft.fft2(t.squeeze(1))
    idx, nb = _radial_bins(h, w, p.device, p.dtype)
    pp = (fp.real ** 2 + fp.imag ** 2).reshape(b, -1)
    tt = (ft.real ** 2 + ft.imag ** 2).reshape(b, -1)
    pt = (fp.real * ft.real + fp.imag * ft.imag).reshape(b, -1)
    z = p.new_zeros((b, nb))
    ix = idx.unsqueeze(0).expand(b, -1)
    P = z.scatter_add(1, ix, pp)
    T = z.clone().scatter_add(1, ix, tt)
    C = z.clone().scatter_add(1, ix, pt)
    spectral = ((P + T - 2 * C).sum(1).mean()) / ((h * w) ** 2)
    assert spectral.item() == pytest.approx(F.mse_loss(p, t).item(), rel=1e-5)


def test_gradients_are_finite():
    x = torch.randn(4, 1, 32, 32, requires_grad=True)
    amse(x, torch.randn(4, 1, 32, 32)).backward()
    assert torch.isfinite(x.grad).all()


def test_gradient_is_finite_on_an_exact_match():
    """sqrt(PSD) and the coherence denominator both vanish at x == y; without
    an epsilon this is where NaN appears, and it would appear mid-training."""
    x = torch.randn(2, 1, 16, 16)
    p = x.clone().requires_grad_(True)
    amse(p, x).backward()
    assert torch.isfinite(p.grad).all()


# --------------------------------------------------------------------------- #
# The behavioural claim
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rho", [0.3, 0.5])
def test_mse_is_minimised_by_shrinking_amplitude(rho):
    """The defect being fixed. MSE prefers amplitude rho, not 1."""
    torch.manual_seed(0)
    t, n = torch.randn(64, 1, 32, 32), torch.randn(64, 1, 32, 32)
    losses = {s: F.mse_loss(_partially_correlated(t, n, rho, s), t).item()
              for s in (rho, 1.0)}
    assert losses[rho] < losses[1.0]


@pytest.mark.parametrize("rho", [0.3, 0.5, 0.8])
def test_amse_prefers_correct_amplitude_over_the_mse_optimum(rho):
    """The fix. AMSE must rank a correctly-scaled prediction ABOVE the shrunken
    one that MSE prefers — this is the entire reason the loss exists."""
    torch.manual_seed(0)
    t, n = torch.randn(64, 1, 32, 32), torch.randn(64, 1, 32, 32)
    at_one = amse(_partially_correlated(t, n, rho, 1.0), t, window=False).item()
    shrunk = amse(_partially_correlated(t, n, rho, rho), t, window=False).item()
    assert at_one < shrunk


def test_amse_optimum_is_near_unit_amplitude():
    """Finite patches leave a small residual bias (see the docstring): the
    optimum lands near 0.95 rather than 1.0. Guard that it stays near 1 and
    never collapses back toward rho."""
    torch.manual_seed(0)
    rho = 0.5
    t, n = torch.randn(64, 1, 48, 48), torch.randn(64, 1, 48, 48)
    grid = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05]
    best = min(grid, key=lambda s: amse(_partially_correlated(t, n, rho, s), t,
                                        window=False).item())
    assert best >= 0.9, f"AMSE optimum collapsed to {best}, near the MSE optimum {rho}"


# --------------------------------------------------------------------------- #
# Integration with the training loss
# --------------------------------------------------------------------------- #
def test_temp_loss_accepts_amse_base_with_and_without_a_mask():
    p, t = torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32)
    m = torch.ones(2, 1, 32, 32)
    m[:, :, :8] = 0.0
    for mask in (None, m):
        v = temp_loss(p, t, ssim_weight=0.2, base="amse", mask=mask)
        assert torch.isfinite(v) and v.item() > 0


def test_amse_masks_the_same_cells_the_other_losses_do():
    """Ocean is zero-filled upstream. If the mask were ignored the loss would
    reward predicting zeros over water."""
    torch.manual_seed(0)
    p, t = torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32)
    m = torch.ones(2, 1, 32, 32)
    m[:, :, :16] = 0.0
    a = amse(p, t, mask=m)
    p2 = p.clone()
    p2[:, :, :16] = 99.0            # garbage, but only where mask == 0
    assert amse(p2, t, mask=m).item() == pytest.approx(a.item(), rel=1e-6)


def test_window_suppresses_edge_leakage():
    """A field with a step at the patch edge leaks power into high wavenumbers
    when unwindowed. The window must reduce the loss attributed to that."""
    t = torch.zeros(4, 1, 32, 32)
    p = torch.zeros(4, 1, 32, 32)
    p[:, :, :, 0] = 5.0             # a hard edge, only at the boundary
    assert amse(p, t, window=True).item() < amse(p, t, window=False).item()


# --------------------------------------------------------------------------- #
# The error-budget decomposed loss
# --------------------------------------------------------------------------- #
from training.losses import decomposed_loss  # noqa: E402


def test_decomposed_loss_is_zero_on_an_exact_match():
    x = torch.randn(2, 1, 40, 40)
    m = torch.ones_like(x)
    assert decomposed_loss(x, x, m, offset_weight=1.0).item() == pytest.approx(0.0, abs=1e-6)


def test_a_pure_uniform_bias_is_charged_only_to_the_offset_term():
    """The whole point: a spatially uniform error is a separable mode, so it
    must land entirely in the offset term and be scalable independently."""
    x = torch.randn(2, 1, 40, 40)
    m = torch.ones_like(x)
    y = x + 0.5
    assert decomposed_loss(x, y, m, offset_weight=1.0,
                           ssim_weight=0.0).item() == pytest.approx(0.5, abs=1e-4)
    assert decomposed_loss(x, y, m, offset_weight=0.0,
                           ssim_weight=0.0).item() == pytest.approx(0.0, abs=1e-5)


def test_offset_weight_does_not_change_the_spatial_term():
    """Turning the bias term off must leave the pattern term untouched,
    otherwise the two modes are not actually separated."""
    torch.manual_seed(0)
    p, t = torch.randn(2, 1, 40, 40), torch.randn(2, 1, 40, 40)
    m = torch.ones_like(p)
    a = decomposed_loss(p, t, m, offset_weight=1.0, ssim_weight=0.0).item()
    b = decomposed_loss(p, t, m, offset_weight=0.0, ssim_weight=0.0).item()
    mu = ((p * m).sum((1, 2, 3)) / m.sum((1, 2, 3))
          - (t * m).sum((1, 2, 3)) / m.sum((1, 2, 3))).abs().mean().item()
    assert a - b == pytest.approx(mu, rel=1e-4)


def test_decomposed_loss_respects_the_mask():
    torch.manual_seed(0)
    p, t = torch.randn(2, 1, 40, 40), torch.randn(2, 1, 40, 40)
    m = torch.ones_like(p)
    m[:, :, :10] = 0.0
    before = decomposed_loss(p, t, m, offset_weight=0.5).item()
    p2 = p.clone()
    p2[:, :, :10] = 42.0                      # garbage outside the mask only
    assert decomposed_loss(p2, t, m, offset_weight=0.5).item() == pytest.approx(
        before, rel=1e-5)


def test_gradients_flow_with_the_offset_term_disabled():
    """offset_weight=0 must still train the spatial pattern, not zero out."""
    p = torch.randn(2, 1, 40, 40, requires_grad=True)
    t = torch.randn(2, 1, 40, 40)
    decomposed_loss(p, t, torch.ones_like(t), offset_weight=0.0).backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
