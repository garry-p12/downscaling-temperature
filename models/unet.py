"""Plain convolutional U-Net downscaler — the architecture ablation.

Same inputs, same loss, same splits as the Swin model; the only thing removed
is windowed self-attention. If this matches the transformer, the attention is
not earning its parameters and the simpler model should ship.

Operates at target resolution throughout (the coarse predictor is already
bilinearly interpolated onto the 10 km grid), so this is an encoder-decoder
corrector, not a super-resolution stack.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convs with GroupNorm + GELU. Reflect padding, as in the Swin
    decoder, so boundary cells are not fed artificial zeros."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(min(8, cout), cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(min(8, cout), cout), nn.GELU(),
        )

    def forward(self, x):
        return self.body(x)


class UNetDownscaler(nn.Module):
    def __init__(self, in_channels: int, base: int = 96, depth: int = 4,
                 out_channels: int = 1):
        super().__init__()
        dims = [base * (2 ** i) for i in range(depth)]
        self.depth = depth
        self.divisor = 2 ** (depth - 1)

        self.enc = nn.ModuleList()
        cin = in_channels
        for d in dims:
            self.enc.append(ConvBlock(cin, d))
            cin = d
        self.pool = nn.MaxPool2d(2)

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(dims[i], dims[i - 1], 2, stride=2))
            self.dec.append(ConvBlock(2 * dims[i - 1], dims[i - 1]))

        self.head = nn.Conv2d(base, out_channels, 3, padding=1,
                              padding_mode="reflect")

    def forward(self, x: torch.Tensor, tasks: set[str] | None = None) -> dict:
        del tasks                      # temperature-only; kept for API parity
        H, W = x.shape[-2:]
        ph, pw = (-H) % self.divisor, (-W) % self.divisor
        if ph or pw:
            mode = "reflect" if (ph < H and pw < W) else "replicate"
            x = F.pad(x, (0, pw, 0, ph), mode=mode)

        skips = []
        for i, block in enumerate(self.enc):
            x = block(x)
            if i < self.depth - 1:
                skips.append(x)
                x = self.pool(x)

        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            # Odd sizes make the transpose-conv output miss the skip by a row.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = dec(torch.cat([x, skip], dim=1))

        out = self.head(x)
        return {"temp": out[..., :H, :W]}
