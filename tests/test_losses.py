"""Training losses, in particular the land/holdout mask — a silent bug there
trains the network to predict zeros over water."""
from __future__ import annotations

import pytest
import torch

from training.losses import ssim, temp_loss


def test_ssim_of_identical_fields_is_one():
    x = torch.randn(2, 1, 32, 32)
    assert ssim(x, x).item() == pytest.approx(1.0, abs=1e-3)


def test_ssim_is_symmetric():
    a, b = torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32)
    assert ssim(a, b).item() == pytest.approx(ssim(b, a).item(), abs=1e-6)


def test_temp_loss_is_minimal_at_the_truth():
    y = torch.randn(2, 1, 32, 32)
    assert temp_loss(y, y).item() < temp_loss(y + 0.5, y).item()


def test_l1_and_mse_bases_differ_on_large_errors():
    y = torch.zeros(1, 1, 16, 16)
    pred = torch.full_like(y, 4.0)
    l1 = temp_loss(pred, y, ssim_weight=0.0, base="l1").item()
    mse = temp_loss(pred, y, ssim_weight=0.0, base="mse").item()
    assert mse > l1, "MSE must penalise the large error harder — that is why L1 is the default"


def test_mask_excludes_cells_from_the_pixel_term():
    """Errors in masked (ocean / holdout) cells must not reach the gradient."""
    y = torch.zeros(1, 1, 16, 16)
    pred = torch.zeros_like(y)
    pred[..., :8, :] = 10.0           # huge error, entirely in the masked half
    mask = torch.ones_like(y)
    mask[..., :8, :] = 0.0
    assert temp_loss(pred, y, ssim_weight=0.0, mask=mask).item() == pytest.approx(0.0)


def test_mask_normalizes_by_valid_cell_count_not_total():
    y = torch.zeros(1, 1, 16, 16)
    pred = torch.ones_like(y)
    mask = torch.zeros_like(y)
    mask[..., :4, :] = 1.0            # a quarter of the field is valid
    # Mean |error| over VALID cells is 1.0; dividing by the full count gives 0.25.
    assert temp_loss(pred, y, ssim_weight=0.0, mask=mask).item() == pytest.approx(1.0)


def test_mask_accepts_an_unsqueezed_channel_dim():
    """The Dataset yields (B, H, W) masks; the loss must handle both ranks."""
    y = torch.randn(2, 1, 8, 8)
    pred = torch.randn(2, 1, 8, 8)
    m3 = torch.ones(2, 8, 8)
    m4 = m3.unsqueeze(1)
    assert temp_loss(pred, y, mask=m3).item() == pytest.approx(
        temp_loss(pred, y, mask=m4).item(), abs=1e-6)


def test_all_masked_out_does_not_produce_nan():
    y = torch.randn(1, 1, 8, 8)
    loss = temp_loss(torch.randn(1, 1, 8, 8), y, mask=torch.zeros_like(y))
    assert torch.isfinite(loss)


def test_ssim_weight_actually_changes_the_loss():
    y = torch.randn(1, 1, 32, 32)
    pred = y + 0.3 * torch.randn_like(y)
    assert temp_loss(pred, y, ssim_weight=0.0).item() != \
        pytest.approx(temp_loss(pred, y, ssim_weight=0.5).item())


def test_loss_is_differentiable():
    pred = torch.randn(1, 1, 16, 16, requires_grad=True)
    temp_loss(pred, torch.randn(1, 1, 16, 16)).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
