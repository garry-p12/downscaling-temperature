"""Shared fixtures.

The suite is deliberately offline and data-free: it must pass on a fresh clone
with no Zarr store, no checkpoints, no CDS key and no GPU. Anything that needs
a real dataset belongs in data/sanity_check.py, which is a gate on the sweep,
not a unit test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def tiny_model_cfg() -> dict:
    """A Swin config small enough to instantiate in a unit test.

    Same keys as configs/model.yaml, ~1000x fewer parameters.
    """
    return {
        "arch": "swin",
        "in_channels": 4,
        "patch_size": 2,
        "encoder": {
            "embed_dim": 24,
            "depths": [1, 1, 1, 1],
            "num_heads": [1, 1, 1, 1],
            "window_size": 4,
            "mlp_ratio": 2.0,
            "drop_path": 0.0,
        },
        "temp_head": {"enabled": True, "out_channels": 1},
    }


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
