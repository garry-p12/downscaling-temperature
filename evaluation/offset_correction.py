"""Correct the spatially-uniform daily offset — the half of the error no
spatial model can reach.

README 5.4 decomposed the holdout residual and found 50.8% of its variance is a
single number per day applied to the whole region: a product-level disagreement
between POWER and ERA5-Land. Every architecture and every covariate set in this
project competed for the 2.8% that static spatial information can address while
that term sat untouched. This estimates it and subtracts it.

THREE ESTIMATORS, DELIBERATELY SEPARATED BY WHAT THEY ASSUME
    spatial   The offset is measured over the TRAINING region on the SAME day
              and applied to the holdout. Uses no holdout truth whatsoever and
              needs no time-series model, so it is the only one that is a fair
              comparison against an uncorrected model. It is also a direct test
              of the uniformity claim: it can only work if the offset really is
              shared across the domain.
    ar1/ar3   Fit on the holdout's OWN offset history and predicted one step
              ahead. This is 5.4's framing, and it assumes yesterday's truth is
              available IN THE REGION BEING PREDICTED — realistic for delayed
              reanalysis, but it is strictly more information than the
              uncorrected model gets, so the two are not directly comparable.
              Reported because 5.4 quotes it, labelled because it is not free.
    oracle    The true holdout offset. Not a method — the ceiling this family
              of corrections is working toward.

Coefficients are fit on the TRAIN split only and applied to test.

Usage:
    python -m evaluation.offset_correction --archs v2_landcov_s1337 of0_s1337
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common import load_config                       # noqa: E402
from common.normalize import Normalizer              # noqa: E402
from data.build_dataset import NO_NORMALIZE          # noqa: E402
from models.model import load_checkpoint             # noqa: E402
from training.dataset import holdout_bounds          # noqa: E402


def predict_all(ckpt, ds, nz, names_stored, device="cpu"):
    """Model output in ANOMALY space for every day in the store."""
    model, mcfg = load_checkpoint(ckpt, device, load_config("model"))
    names = list(mcfg.get("use_channels") or names_stored)
    idx = [names_stored.index(n) for n in names]
    n = ds.sizes["time"]
    out = np.empty((n, ds.sizes["y"], ds.sizes["x"]), "float32")
    for k in range(n):
        x = ds["input"].isel(time=k).values[idx].copy()
        for c, nm in enumerate(names):
            if nm not in NO_NORMALIZE:
                x[c] = nz.transform(f"in::{nm}", x[c])
        np.nan_to_num(x, copy=False)
        with torch.no_grad():
            out[k] = model(torch.from_numpy(x).unsqueeze(0).to(device),
                           {"temp"})["temp"][0, 0].cpu().numpy()
    return out


def subregion_means(resid, mask, ny, nx, k=3):
    """Mean residual in each of k x k blocks of the TRAINING region.

    One domain-wide mean throws away the fact that the bias is not perfectly
    uniform (measured train<->holdout offset correlation is only ~0.6). Letting
    a fit weight sub-regions separately lets it lean on whichever parts of the
    domain actually track the held-out one.
    """
    out = []
    ys = np.linspace(0, ny, k + 1).astype(int)
    xs = np.linspace(0, nx, k + 1).astype(int)
    for a in range(k):
        for b in range(k):
            m = np.zeros_like(mask)
            m[ys[a]:ys[a + 1], xs[b]:xs[b + 1]] = True
            m &= mask
            out.append(resid[:, m].mean(1) if m.sum() >= 20
                       else np.zeros(resid.shape[0]))
    return np.stack(out, 1)


def ridge_fit_predict(X, y, idx_fit, alpha=1.0):
    """Ridge fit on idx_fit only, prediction everywhere. Columns standardised
    using the FITTING rows so test statistics never touch the fit."""
    mu, sd = X[idx_fit].mean(0), X[idx_fit].std(0) + 1e-9
    Z = (X - mu) / sd
    Zf = np.c_[Z[idx_fit], np.ones(len(idx_fit))]
    A = Zf.T @ Zf + alpha * np.eye(Zf.shape[1])
    A[-1, -1] -= alpha                       # do not penalise the intercept
    beta = np.linalg.solve(A, Zf.T @ y[idx_fit])
    return np.c_[Z, np.ones(len(Z))] @ beta


def fit_ar(series, order, idx_fit):
    """Least-squares AR(order) on the fitting indices; returns coefficients."""
    X, y = [], []
    for t in idx_fit:
        if t - order < 0:
            continue
        X.append([series[t - l] for l in range(1, order + 1)] + [1.0])
        y.append(series[t])
    return np.linalg.lstsq(np.asarray(X), np.asarray(y), rcond=None)[0]


def main(archs, zarr, device, out):
    cfg = load_config("data")
    ds = xr.open_zarr(zarr, consolidated=True)
    nz = Normalizer.load(str(Path(zarr).parent / "norm_stats.json"))
    stored = ds["channel_in"].values.tolist()
    i0, i1, j0, j1 = holdout_bounds(ds, cfg["holdout"])

    split = ds["split"].values
    is_tr, is_te = split == "train", split == "test"
    land = ds["input"].values[0, stored.index("land_mask")] > 0.5
    hold = np.zeros_like(land); hold[i0:i1, j0:j1] = True
    m_hold, m_train = hold & land, (~hold) & land        # disjoint by construction
    truth_a = np.nan_to_num(ds["target"].values[:, 0])

    print(f"train-region cells {int(m_train.sum())}, holdout cells "
          f"{int(m_hold.sum())}, days {len(split)} "
          f"({is_tr.sum()} train / {is_te.sum()} test)\n")

    report = {}
    for arch in archs:
        ck = REPO / "checkpoints" / arch / "last.pt"
        if not ck.exists():
            print(f"{arch}: no last.pt, skipping"); continue
        pred_a = predict_all(ck, ds, nz, stored, device)
        pred_a = nz.inverse("out::tmp", pred_a)
        resid = truth_a - pred_a                       # what the model still owes

        off_hold = resid[:, m_hold].mean(1)            # the thing to predict
        off_train = resid[:, m_train].mean(1)          # what we may legitimately see

        def rmse(corr):
            # The residual is truth - pred in anomaly space; the climatology
            # cancels in the difference, so this is already degC.
            return float(np.sqrt(np.mean((resid[is_te][:, m_hold]
                                          - corr[is_te, None]) ** 2)))

        zero = np.zeros(len(split))
        res = {"uncorrected": rmse(zero),
               "spatial (train-region offset, same day)": rmse(off_train),
               "oracle (true holdout offset)": rmse(off_hold)}

        tr_idx = np.where(is_tr)[0]

        # --- combined estimators. Every feature below is observable without
        # --- any held-out truth, so these stay comparable to an uncorrected
        # --- model. The AR variants further down are not, and are labelled.
        ny, nx = land.shape
        lag = lambda a, n: np.r_[np.full(n, a[:n].mean()), a[:-n]]  # noqa: E731
        sub = subregion_means(resid, m_train, ny, nx, k=3)
        doy_ = pd.DatetimeIndex(ds["time"].values).dayofyear.values
        season = np.c_[np.sin(2 * np.pi * doy_ / 365.25),
                       np.cos(2 * np.pi * doy_ / 365.25)]
        coarse_mean = ds["input"].values[:, stored.index("coarse_tmp")][:, m_hold].mean(1)

        feats = {"spatial + lag1 + season":
                 np.c_[off_train, lag(off_train, 1), season]}
        # Sweep the block count. More blocks = more freedom to weight the parts
        # of the domain that actually track the holdout, but also more chance to
        # fit the train split's noise; the ridge is fit on train days only and
        # scored on test, so the sweep shows where that turns over.
        for kk_ in (3, 6):
            feats[f"{kk_ * kk_} sub-region means (k={kk_})"] = \
                subregion_means(resid, m_train, ny, nx, k=kk_)
        feats["sub-regions k=6 + lag1 + season"] = np.c_[
            subregion_means(resid, m_train, ny, nx, k=6), lag(off_train, 1), season]
        for nm, X in feats.items():
            res[f"FAIR: {nm}"] = rmse(ridge_fit_predict(X, off_hold, tr_idx))

        for p in (1, 3):
            beta = fit_ar(off_hold, p, tr_idx)
            pred_off = np.zeros(len(split))
            for t in range(p, len(split)):
                pred_off[t] = beta[-1] + sum(beta[l - 1] * off_hold[t - l]
                                             for l in range(1, p + 1))
            res[f"ar({p}) on holdout history [uses lagged holdout truth]"] = rmse(pred_off)

        base = res["uncorrected"]
        r = np.corrcoef(off_train[is_te], off_hold[is_te])[0, 1]
        print(f"=== {arch}")
        print(f"  corr(train-region offset, holdout offset) on test days: {r:.4f}")
        print(f"  holdout offset sd {off_hold[is_te].std():.4f} degC\n")
        print(f"  {'estimator':<48}{'RMSE':>9}{'vs uncorr':>12}")
        for k2, v in res.items():
            print(f"  {k2:<48}{v:9.4f}{(v - base) / base * 100:+11.1f}%")
        print()
        report[arch] = {"metrics": res, "train_holdout_offset_corr": float(r),
                        "holdout_offset_sd": float(off_hold[is_te].std())}

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", required=True)
    ap.add_argument("--zarr", default="data_store_sc/super/dataset.zarr")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/offset_correction.json")
    a = ap.parse_args()
    main(a.archs, a.zarr, a.device, a.out)
