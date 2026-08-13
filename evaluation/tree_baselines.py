"""Per-pixel tree baselines: gradient boosting and random forest.

The sharpest ablation available. These see ONE CELL AT A TIME — no neighbours,
no convolution, no attention — so they answer: how much of the network's skill
is a nonlinear *local* mapping from (coarse temperature, elevation, land cover,
season, position) to truth, and how much genuinely requires spatial reasoning?

If gradient boosting lands close to the Swin's RMSE, the "the model learns
terrain shape" claim is much weaker than it looks, because terrain shape is
exactly what a per-pixel model cannot see.

Random forest is included separately because it is the workhorse of the
statistical-downscaling literature and reviewers expect it.

Features per cell: the four input channels, day-of-year encoded as
sin/cos (so December and January are adjacent, not maximally distant), and
projected x/y. Position is included deliberately — it lets the tree memorize
per-cell corrections, which is the fairest version of this baseline and a
useful upper bound on how much of the task is memorization.

Usage:
    python -m evaluation.tree_baselines
    python -m evaluation.tree_baselines --model gbm --max-train 500000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from common import load_config
from training.metrics import residual_spatial_corr, temperature_metrics


def build_features(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (time, y, x) into rows of per-cell features + target."""
    inp = ds["input"].values                          # (T, C, Y, X)
    tgt = ds["target"].sel(channel_out="tmp").values  # (T, Y, X)
    T, C, Y, X = inp.shape

    doy = pd.DatetimeIndex(ds["time"].values).dayofyear.values.astype("float32")
    ang = 2 * np.pi * doy / 365.25
    yy, xx = np.meshgrid(ds["y"].values, ds["x"].values, indexing="ij")

    feats = [inp[:, c].reshape(T, -1) for c in range(C)]
    feats.append(np.repeat(np.sin(ang)[:, None], Y * X, axis=1))
    feats.append(np.repeat(np.cos(ang)[:, None], Y * X, axis=1))
    feats.append(np.repeat(yy.ravel()[None, :], T, axis=0))
    feats.append(np.repeat(xx.ravel()[None, :], T, axis=0))

    Xf = np.stack(feats, axis=-1).reshape(-1, len(feats)).astype("float32")
    yf = tgt.reshape(-1).astype("float32")
    return Xf, yf


def feature_names(ds: xr.Dataset) -> list[str]:
    return ds["channel_in"].values.tolist() + ["doy_sin", "doy_cos", "y", "x"]


def fit_predict(kind: str, ds: xr.Dataset, max_train: int, seed: int,
                n_jobs: int = -1):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

    train = ds.sel(time=ds["split"] == "train")
    test = ds.sel(time=ds["split"] == "test")
    Xtr, ytr = build_features(train)
    Xte, _ = build_features(test)

    rng = np.random.RandomState(seed)
    if max_train and len(Xtr) > max_train:
        # Random forests on millions of rows are memory-bound, not accuracy-
        # bound; subsampling costs little and keeps the run tractable.
        sel = rng.choice(len(Xtr), max_train, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
    print(f"[{kind}] fitting on {len(Xtr):,} rows x {Xtr.shape[1]} features")

    if kind == "gbm":
        model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.06, max_depth=None,
            max_leaf_nodes=63, early_stopping=True, validation_fraction=0.1,
            random_state=seed)
    elif kind == "rf":
        model = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, max_features="sqrt",
            n_jobs=n_jobs, random_state=seed)
    else:
        raise ValueError(f"unknown kind {kind!r}; expected gbm | rf")

    model.fit(Xtr, ytr)
    pred = model.predict(Xte).reshape(test.sizes["time"], test.sizes["y"],
                                      test.sizes["x"])
    return model, pred, test


def main(models: list[str], max_train: int, seed: int) -> None:
    cfg = load_config("data")
    ds = xr.open_zarr(cfg["dataset"]["out_zarr"], consolidated=True)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    for kind in models:
        model, pred, test = fit_predict(kind, ds, max_train, seed)
        truth = test["target"].sel(channel_out="tmp").values
        bilin = test["input"].sel(channel_in="coarse_tmp").values

        report = {
            "arch": kind,
            "n_test": int(test.sizes["time"]),
            "params_M": None,          # tree ensembles: report size differently
            "model_temp": temperature_metrics(pred, truth),
        }
        report["model_temp"]["resid_corr_vs_bilinear"] = \
            residual_spatial_corr(pred, truth, bilin)
        if hasattr(model, "feature_importances_"):
            report["feature_importance"] = dict(zip(
                feature_names(ds),
                [float(v) for v in model.feature_importances_]))

        path = out_dir / f"eval_report_{kind}.json"
        path.write_text(json.dumps(report, indent=2))
        m = report["model_temp"]
        print(f"[{kind}] rmse={m['rmse']:.3f} mae={m['mae']:.3f} "
              f"bias={m['bias']:+.3f} resid_corr={m['resid_corr_vs_bilinear']:.3f}")
        print(f"[{kind}] wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="both", choices=["gbm", "rf", "both"])
    ap.add_argument("--max-train", type=int, default=1_000_000,
                    help="subsample training rows (0 = use all)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    kinds = ["gbm", "rf"] if args.model == "both" else [args.model]
    main(kinds, args.max_train, args.seed)
