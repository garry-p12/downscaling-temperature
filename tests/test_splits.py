"""Temporal split assignment — the guard against train/test leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import load_config
from data.build_dataset import assign_splits

CFG = {"train": ["2019-01-01", "2021-12-31"],
       "val": ["2022-01-01", "2022-12-31"],
       "test": ["2023-01-01", "2023-12-31"]}


def test_each_day_lands_in_exactly_one_split():
    times = pd.date_range("2019-01-01", "2023-12-31", freq="D")
    split = assign_splits(times, CFG)
    assert set(np.unique(split)) == {"train", "val", "test"}
    assert len(split) == len(times)


def test_boundaries_are_inclusive_on_both_ends():
    times = pd.DatetimeIndex(["2019-01-01", "2021-12-31", "2022-01-01",
                              "2022-12-31", "2023-01-01", "2023-12-31"])
    assert list(assign_splits(times, CFG)) == [
        "train", "train", "val", "val", "test", "test"]


def test_days_outside_every_window_are_marked_none():
    times = pd.DatetimeIndex(["2018-06-01", "2020-06-01", "2030-06-01"])
    assert list(assign_splits(times, CFG)) == ["none", "train", "none"]


def test_splits_are_contiguous_blocks_not_interleaved():
    """Random day-level splits leak: daily fields are strongly autocorrelated."""
    times = pd.date_range("2019-01-01", "2023-12-31", freq="D")
    split = assign_splits(times, CFG)
    changes = int((split[1:] != split[:-1]).sum())
    assert changes == 2, "splits must form 3 contiguous runs"


def test_shipped_config_covers_the_full_span_with_no_gap():
    cfg = load_config("configs/data_southcentral.yaml")
    times = pd.date_range(cfg["time"]["train"][0], cfg["time"]["test"][1], freq="D")
    assert "none" not in set(np.unique(assign_splits(times, cfg["time"])))
