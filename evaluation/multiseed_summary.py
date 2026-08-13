"""Decide whether the architecture ranking is real or run-to-run noise.

RESULTS.md §3 claims nine of ten architectures are tied. That rests on one
training run each plus bootstrap CIs — but a bootstrap over days measures
SAMPLING variability (which days landed in the test year), not TRAINING
variability (which random initialization the optimizer got). Only the second
can tell you whether "model A beats model B by 0.03 degC" survives a rerun.

This reads a multi-seed holdout report and answers one question:

    Is the spread BETWEEN SEEDS of one architecture larger than the spread
    BETWEEN ARCHITECTURES of their means?

If yes, the ranking is noise and architecture selection is settled. If some
architecture separates by more than a few seed-standard-deviations from the
rest, that is a real effect worth reporting.

Usage:
    python -m evaluation.multiseed_summary \\
        --report image_outputs/multiseed/holdout_report.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("rmse", "ssim", "resid_corr_vs_interp", "bias")


def main(report: str, metric: str) -> None:
    data = json.loads(Path(report).read_text())
    methods = data["methods"]

    # keys look like "deepsd_s1337"; the baseline row has no seed suffix
    by_arch: dict[str, dict[int, dict]] = defaultdict(dict)
    baseline = None
    for name, m in methods.items():
        hit = re.match(r"^(.*)_s(\d+)$", name)
        if hit:
            by_arch[hit.group(1)][int(hit.group(2))] = m
        elif "interp" in name.lower():
            baseline = m

    if not by_arch:
        raise SystemExit(f"No seed-suffixed entries in {report}")

    rows = []
    for arch, seeds in by_arch.items():
        v = np.array([seeds[s][metric] for s in sorted(seeds)])
        rows.append((arch, v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0,
                     v.min(), v.max(), len(v)))
    lower_is_better = metric in ("rmse", "bias")
    rows.sort(key=lambda r: r[1] if lower_is_better else -r[1])

    print(f"\n=== {metric} across seeds  (n={rows[0][5]} per arch)")
    print(f"{'arch':<14}{'mean':>9}{'sd':>8}{'min':>9}{'max':>9}{'seeds':>7}")
    print("-" * 56)
    for a, mu, sd, lo, hi, n in rows:
        print(f"{a:<14}{mu:>9.4f}{sd:>8.4f}{lo:>9.4f}{hi:>9.4f}{n:>7}")
    if baseline:
        print(f"{'(interp POWER)':<14}{baseline[metric]:>9.4f}")

    means = np.array([r[1] for r in rows])
    sds = np.array([r[2] for r in rows])
    between_model = means.max() - means.min()
    typical_seed_sd = float(np.median(sds))
    print(f"\nbetween-architecture spread (max-min of means): {between_model:.4f}")
    print(f"typical between-seed sd (median):               {typical_seed_sd:.4f}")

    if typical_seed_sd <= 0:
        print("\nOnly one seed per architecture — cannot separate the two.")
        return

    ratio = between_model / typical_seed_sd
    print(f"ratio: {ratio:.2f}x")
    if ratio < 2:
        print("\nVERDICT: architecture differences are WITHIN seed noise.")
        print("  The ranking is not meaningful; select on cost, not accuracy.")
    else:
        print("\nVERDICT: at least one architecture separates beyond seed noise.")
        # Which ones are distinguishable from the best?
        best_mu, best_sd = rows[0][1], max(rows[0][2], 1e-9)
        print(f"\n  distance from best ({rows[0][0]}) in pooled seed-sd units:")
        for a, mu, sd, *_ in rows[1:]:
            pooled = np.sqrt((best_sd ** 2 + max(sd, 1e-9) ** 2) / 2)
            d = abs(mu - best_mu) / pooled
            verdict = "DISTINGUISHABLE" if d >= 2 else "tied"
            print(f"    {a:<14}{d:>6.2f} sd   {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="image_outputs/multiseed/holdout_report.json")
    ap.add_argument("--metric", default="rmse", choices=METRICS)
    args = ap.parse_args()
    main(args.report, args.metric)
