"""DeepSD-style stacked SRCNN — the canonical deep-learning downscaling baseline.

After Vandal et al., "DeepSD: Generating High Resolution Climate Change
Projections through Single Image Super-Resolution" (KDD 2017): a stack of
SRCNN blocks, each refining the previous estimate, with elevation re-injected
at every stage so terrain information is available at all scales.

Adaptation, stated plainly: the original stacks progressive 2x super-resolution
steps from a genuinely coarse raster. Here every input already sits on the
10 km grid (this pipeline is an SR-style *corrector*), so the stack refines in
place rather than upsampling. The defining features are kept — shallow SRCNN
blocks, residual refinement, elevation re-injected per stage — and the model
stays deliberately tiny (~100k params vs the Swin's 41M). That size gap is the
point of the comparison, not a handicap to apologize for.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SRCNNBlock(nn.Module):
    """Classic 9-1-5 SRCNN: patch extraction, nonlinear mapping, reconstruction."""

    def __init__(self, in_channels: int, n1: int = 64, n2: int = 32):
        super().__init__()
        pm = "reflect"
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, n1, 9, padding=4, padding_mode=pm), nn.ReLU(True),
            nn.Conv2d(n1, n2, 1), nn.ReLU(True),
            nn.Conv2d(n2, 1, 5, padding=2, padding_mode=pm),
        )

    def forward(self, x):
        return self.body(x)


class DeepSD(nn.Module):
    """``n_stages`` SRCNN blocks; each sees [current estimate, statics]."""

    def __init__(self, in_channels: int, n_stages: int = 3, n1: int = 64,
                 n2: int = 32, coarse_index: int = 0):
        super().__init__()
        self.coarse_index = coarse_index
        self.n_static = in_channels - 1        # every channel but coarse_tmp
        self.stages = nn.ModuleList(
            [SRCNNBlock(1 + self.n_static, n1, n2) for _ in range(n_stages)])

    def forward(self, x: torch.Tensor, tasks: set[str] | None = None) -> dict:
        del tasks
        idx = self.coarse_index
        est = x[:, idx:idx + 1]
        statics = torch.cat([x[:, :idx], x[:, idx + 1:]], dim=1)
        for stage in self.stages:
            # Residual refinement: each stage corrects the running estimate.
            est = est + stage(torch.cat([est, statics], dim=1))
        return {"temp": est}
