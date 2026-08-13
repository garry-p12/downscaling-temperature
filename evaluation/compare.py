"""Collect every eval report into one comparison table.

Reads outputs/eval_report_*.json (one per method) and prints a ranked table
plus a markdown version for pasting into notes. Parameter count is reported
next to RMSE deliberately: "1.34 degC at 41M params" versus "1.51 degC at
0.1M" is a result about cost-effectiveness that RMSE alone hides.

The non-ML baselines (bilinear, lapse, BCSD) are lifted from whichever report
has them — they are identical across runs by construction, since every method
is scored on the same test split by the same code.

Usage:
    python -m evaluation.compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROWS = [
    ("bilinear_temp", "Bilinear (POWER as-is)", None),
    ("lapse_temp", "Lapse-rate", None),
    ("bcsd_temp", "BCSD (quantile map)", None),
]


def spectral_ratio(report: dict, k_lo: int = 12) -> float | None:
    """Model/truth power ratio at fine scales (k >= k_lo, roughly <40 km).

    Reported next to RMSE because the two disagree: an MSE-trained regressor
    can win RMSE by predicting the conditional mean, which destroys variance
    at exactly the scales downscaling exists to produce. 1.0 = matches truth,
    <1 = over-smoothed. Ranking on RMSE alone hides this entirely.
    """
    spec = report.get("power_spectrum_tmp")
    if not spec:
        return None
    m = np.asarray(spec["model"][k_lo:])
    t = np.asarray(spec["truth"][k_lo:])
    if t.sum() <= 0:
        return None
    return float(m.sum() / t.sum())


def load_reports(out_dir: Path) -> dict[str, dict]:
    reports = {}
    for path in sorted(out_dir.glob("eval_report_*.json")):
        arch = path.stem.replace("eval_report_", "")
        reports[arch] = json.loads(path.read_text())
    return reports


def main(out_dir: str) -> None:
    d = Path(out_dir)
    reports = load_reports(d)
    if not reports:
        raise SystemExit(f"No eval_report_*.json in {d}. Run evaluate.py / "
                         f"tree_baselines.py first.")

    rows = []
    # Non-ML baselines: take them from any report that carries them.
    donor = next((r for r in reports.values() if "bilinear_temp" in r), None)
    if donor:
        for key, label, _ in ROWS:
            if key in donor:
                m = donor[key]
                rows.append((label, m.get("ssim"), m.get("spatial_corr"), None,
                             None, m["rmse"], m["mae"], m["bias"], None, True))

    for arch, rep in reports.items():
        m = rep.get("model_temp")
        if not m:
            continue
        rows.append((arch, m.get("ssim"), m.get("spatial_corr"),
                     m.get("resid_corr_vs_lapse") or
                     m.get("resid_corr_vs_bilinear"),
                     spectral_ratio(rep), m["rmse"], m["mae"], m["bias"],
                     rep.get("params_M"), False))

    # Ranked by SSIM (structure), not RMSE: a model can win RMSE by blurring,
    # and SSIM is the metric that refuses to reward that.
    rows.sort(key=lambda r: (r[1] is None, -(r[1] or 0)))
    best_baseline = min((r[5] for r in rows if r[9]), default=None)

    hdr = (f"{'method':<24}{'SSIM':>8}{'sp_corr':>9}{'resid':>8}{'spec':>7}"
           f"{'RMSE':>8}{'MAE':>8}{'bias':>8}{'params_M':>10}{'skill':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    md = ["| Method | SSIM | Spatial corr | Resid corr | Spectral ratio | "
          "RMSE °C | MAE | Bias | Params (M) | Skill vs best baseline |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for label, ss, sc, rc, sr, rmse, mae, bias, pm, _is_base in rows:
        skill = best_baseline / rmse if best_baseline else float("nan")
        f = lambda v, n=3: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
        print(f"{label:<24}{f(ss, 4):>8}{f(sc, 4):>9}{f(rc):>8}{f(sr, 2):>7}"
              f"{rmse:>8.3f}{mae:>8.3f}{bias:>+8.3f}{f(pm, 2):>10}{skill:>7.2f}")
        md.append(f"| {label} | {f(ss, 4)} | {f(sc, 4)} | {f(rc)} | {f(sr, 2)} | "
                  f"{rmse:.3f} | {mae:.3f} | {bias:+.3f} | {f(pm, 2)} | "
                  f"{skill:.2f} |")

    print("\nRANKED BY SSIM — structural similarity (local mean, variance and")
    print("      covariance). Unlike RMSE it cannot be won by over-smoothing:")
    print("      a blurred field scores ~0.2 even when its RMSE looks good.")
    print("sp_corr = raw spatial correlation. Sits near 0.95+ for free via the")
    print("      large-scale field, so treat it as a floor, not a ranking.")
    print("resid = spatial corr after removing the lapse baseline — the")
    print("      fine-scale skill the model is actually responsible for.")
    print("spec  = model/truth power ratio at fine scales (k>=12, ~<40 km).")
    print("        1.0 matches truth; <1 is over-smoothed.")
    print("skill = best_non_ML_baseline_rmse / method_rmse  (>1 is better)\n")

    md_path = d / "comparison.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"[compare] wrote {md_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()
    main(args.out_dir)
