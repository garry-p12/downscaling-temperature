"""A deliberately linear corrector — the headroom probe, not a contender.

WHY THIS EXISTS
    README 5.1 found ten architectures statistically tied (between-model spread
    1.22x the between-seed sd), and 5.6 found the only thing that ever beat the
    noise floor was a LOSS change. Before designing a new architecture it is
    worth answering a prior question: how much of what DeepSD achieves is
    actually due to its NONLINEARITY?

    This model is the control for that question. It has exactly the same shape
    as the task — a residual correction to the interpolated coarse field, with
    a spatial receptive field over the static covariates — but no activation
    anywhere, so the whole map from inputs to output is affine. Compare its
    per-wavenumber coherence (evaluation/spectral.py) against DeepSD's:

      linear ~= DeepSD   the nonlinear capacity is not buying anything, and a
                         fancier architecture is unlikely to either. The signal
                         that remains is not a modelling-capacity problem.
      linear <<  DeepSD  the network is extracting real nonlinear structure and
                         more architecture may pay.

    Reported per SCALE rather than as one number, because the answer differs by
    scale: coherence is 0.985 at 495 km (interpolation already solves it) and
    under 0.2 below 55 km (little to extract). The interesting band is the
    99-165 km middle, and this probe sizes the headroom there.

KERNEL SIZE
    Matched to DeepSD's effective receptive field rather than to 1. A 1x1 probe
    would answer a different and less interesting question (per-pixel linear),
    and would understate what linear can do. DeepSD's 3 stacked 9-1-5 blocks
    give a receptive field of 3*(8+4)+1 = 37 cells; ``k=13`` captures the bulk
    of it at 13*13*C parameters, which stays small enough to fit without
    regularisation games.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """Affine residual corrector: est = coarse + W * statics (+ b). No ReLU."""

    def __init__(self, in_channels: int, k: int = 13, coarse_index: int = 0):
        super().__init__()
        self.coarse_index = coarse_index
        # Every channel INCLUDING the coarse field: a linear model may also want
        # to reweight or spatially smooth the estimate it starts from.
        self.w = nn.Conv2d(in_channels, 1, k, padding=k // 2,
                           padding_mode="reflect")
        nn.init.zeros_(self.w.weight)      # start as the identity corrector, so
        nn.init.zeros_(self.w.bias)        # epoch 0 == the interpolation baseline

    def forward(self, x: torch.Tensor, tasks: set[str] | None = None) -> dict:
        del tasks
        est = x[:, self.coarse_index:self.coarse_index + 1]
        return {"temp": est + self.w(x)}
