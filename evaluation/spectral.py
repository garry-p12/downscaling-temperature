"""Effective resolution: how fine can this model actually resolve?

A 10 km grid does not mean 10 km detail. Trained under a pixel loss, a network
minimises its loss by SHRINKING the amplitude of any scale it cannot predict
rather than by predicting it (Subich et al. 2025, ICML). Nothing in RMSE, SSIM
or residual correlation reports that directly, so a model can be systematically
blurry and still look fine on every published metric.

This module measures it. For each radial wavenumber it reports

    amplitude ratio   sqrt(PSD_model / PSD_truth)   1.0 = correct variance
    coherence         normalised cross-spectrum      1.0 = perfectly in phase

and defines EFFECTIVE RESOLUTION as the wavelength at which the amplitude ratio
falls below sqrt(0.75), i.e. where a quarter of the per-wavenumber energy has
been lost. That is the paper's criterion.

Two adaptations from the paper, which works on a global sphere:
  * Spherical harmonics become a 2D DFT with radial binning, valid because the
    target grid is equal-area so Parseval holds exactly.
  * The domain is NOT periodic. A raw FFT leaks the boundary discontinuity into
    high wavenumbers and reads as false sharpness, so a Hann window is applied.

Usage:
    python -m evaluation.spectral --archs v2_landcov_s1337 restormer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common import load_config                       # noqa: E402
from common.normalize import Normalizer              # noqa: E402
from data.build_dataset import NO_NORMALIZE          # noqa: E402
from models.model import load_checkpoint             # noqa: E402
from training.dataset import holdout_bounds          # noqa: E402

AMP_FLOOR = np.sqrt(0.75)     # the paper's effective-resolution threshold


def radial_spectrum(pred: np.ndarray, truth: np.ndarray,
                    dx_km: float = 10.0, window: bool = True) -> list[dict]:
    """Per-wavenumber amplitude ratio and coherence for (T, H, W) stacks."""
    _, ny, nx = pred.shape
    w = np.outer(np.hanning(ny), np.hanning(nx)) if window else 1.0
    fp, ft = np.fft.rfft2(pred * w), np.fft.rfft2(truth * w)

    ky = np.fft.fftfreq(ny, dx_km)[:, None]
    kx = np.fft.rfftfreq(nx, dx_km)[None, :]
    kr = np.sqrt(ky ** 2 + kx ** 2)
    nb = max(2, min(ny, nx) // 2)
    edges = np.linspace(0, kr.max(), nb + 1)

    out = []
    for b in range(nb):
        m = (kr >= edges[b]) & (kr < edges[b + 1])
        if not m.any():
            continue
        pp = (np.abs(fp) ** 2)[:, m].sum(1).mean()
        tt = (np.abs(ft) ** 2)[:, m].sum(1).mean()
        ct = np.real(fp * np.conj(ft))[:, m].sum(1).mean()
        k = 0.5 * (edges[b] + edges[b + 1])
        if k <= 0 or tt <= 0:
            continue
        out.append({"wavelength_km": float(1.0 / k),
                    "amplitude_ratio": float(np.sqrt(pp / tt)),
                    "coherence": float(ct / np.sqrt(pp * tt))})
    return out


def effective_resolution(rows: list[dict]) -> float | None:
    """Coarsest wavelength at which amplitude ratio is still above the floor.

    Scanned from coarse to fine, stopping at the FIRST crossing. Scanning for
    the last one would be wrong: the amplitude ratio rises again near the
    Nyquist corner, but with coherence ~0 that is uncorrelated noise, not
    recovered signal — the paper calls this the noise-based regime.
    """
    for r in sorted(rows, key=lambda r: -r["wavelength_km"]):
        if r["amplitude_ratio"] < AMP_FLOOR:
            return r["wavelength_km"]
    return None


def predict_holdout(ckpt: Path, ds, nz, idx_map, box, device="cpu"):
    """(pred, truth) over the holdout crop for the test split, anomaly space."""
    model, mcfg = load_checkpoint(ckpt, device, load_config("model"))
    stored = ds["channel_in"].values.tolist()
    names = list(mcfg.get("use_channels") or stored)
    missing = [n for n in names if n not in stored]
    if missing:
        raise KeyError(f"{ckpt}: store lacks {missing}")
    idx = [stored.index(n) for n in names]
    i0, i1, j0, j1 = box
    P, T = [], []
    for k in idx_map:
        x = ds["input"].isel(time=int(k)).values[idx].copy()
        for c, nm in enumerate(names):
            if nm not in NO_NORMALIZE:
                x[c] = nz.transform(f"in::{nm}", x[c])
        np.nan_to_num(x, copy=False)
        with torch.no_grad():
            o = model(torch.from_numpy(x).unsqueeze(0).to(device),
                      {"temp"})["temp"][0, 0].cpu().numpy()
        y = np.nan_to_num(nz.transform("out::tmp",
                                       ds["target"].isel(time=int(k)).values[0]))
        P.append(o[i0:i1, j0:j1])
        T.append(y[i0:i1, j0:j1])
    return np.asarray(P), np.asarray(T)


def main(archs: list[str], zarr: str, out: str | None, device: str,
         which: str = "best") -> None:
    ds = xr.open_zarr(zarr, consolidated=True)
    nz = Normalizer.load(str(Path(zarr).parent / "norm_stats.json"))
    cfg = load_config("configs/data_southcentral.yaml")
    box = holdout_bounds(ds, cfg["holdout"])
    test = np.where(ds["split"].values == "test")[0]
    print(f"[spectral] {len(test)} test days, holdout crop "
          f"{box[1] - box[0]}x{box[3] - box[2]} cells, using {which}.pt")
    if which == "best":
        print("[spectral] NOTE: best.pt is selected on validation RMSE, and RMSE "
              "is minimised\n           by blurring — the very thing measured "
              "here. Prefer --ckpt last when\n           comparing a loss that "
              "is meant to sharpen.\n")

    report = {}
    for arch in archs:
        ck = REPO / "checkpoints" / arch / f"{which}.pt"
        if not ck.exists():
            print(f"[spectral] {arch}: no checkpoint, skipping")
            continue
        try:
            P, T = predict_holdout(ck, ds, nz, test, box, device)
        except KeyError as e:
            print(f"[spectral] {arch}: {e}")
            continue
        rows = radial_spectrum(P, T)
        eff = effective_resolution(rows)
        report[arch] = {"effective_resolution_km": eff, "spectrum": rows}
        print(f"=== {arch}")
        print(f"{'wavelength km':>14} {'amp ratio':>10} {'coherence':>10}")
        for r in rows:
            flag = "  <- below floor" if r["amplitude_ratio"] < AMP_FLOOR else ""
            print(f"{r['wavelength_km']:14.0f} {r['amplitude_ratio']:10.3f} "
                  f"{r['coherence']:10.3f}{flag}")
        print(f"  effective resolution: "
              f"{'%.0f km' % eff if eff else 'finer than the grid'}"
              f"   (grid is 10 km)\n")

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2))
        print(f"[spectral] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", required=True)
    ap.add_argument("--zarr", default="data_store_sc/super/dataset.zarr")
    ap.add_argument("--out", default="outputs/spectral_report.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default="best", choices=["best", "last"],
                    help="which checkpoint to score. 'best' is chosen on val "
                         "RMSE, which favours the blurriest epoch; use 'last' "
                         "to judge a sharpening loss on equal terms.")
    a = ap.parse_args()
    main(a.archs, a.zarr, a.out, a.device, a.ckpt)
