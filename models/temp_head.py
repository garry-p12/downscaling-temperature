"""Temperature head: pixel-shuffle upsampling decoder -> 10 km temperature.

Consumes the shared backbone feature map at patch-embed resolution and
upsamples by ``patch_size`` back to the full target grid. Trained with
MSE + SSIM (see training/losses.py). Predicts a residual on top of the
bilinearly-upsampled coarse temperature, which the model adds back — the
network only has to learn the fine-scale correction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PixelShuffleUpsampler(nn.Module):
    """Upsample by an integer factor via conv + PixelShuffle blocks (x2 each)."""

    def __init__(self, dim: int, factor: int, hidden: int = 64):
        super().__init__()
        pm = "reflect"   # avoid zero-pad boundary artifacts in the decoder
        layers = [nn.Conv2d(dim, hidden, 3, padding=1, padding_mode=pm), nn.GELU()]
        f = factor
        while f > 1:
            assert f % 2 == 0, "patch_size must be a power of 2"
            layers += [nn.Conv2d(hidden, hidden * 4, 3, padding=1, padding_mode=pm),
                       nn.PixelShuffle(2), nn.GELU()]
            f //= 2
        self.body = nn.Sequential(*layers)
        self.out_hidden = hidden

    def forward(self, x):
        return self.body(x)


class TempHead(nn.Module):
    def __init__(self, dim: int, patch_size: int, out_channels: int = 1,
                 hidden: int = 64):
        super().__init__()
        self.up = PixelShuffleUpsampler(dim, patch_size, hidden)
        self.out = nn.Conv2d(hidden, out_channels, 3, padding=1, padding_mode="reflect")

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.out(self.up(feat))
