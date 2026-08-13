"""Train + evaluate several architectures sequentially, then build the table.

Sequential, not parallel, on purpose: these runs share one GPU/MPS device, and
overlapping them makes wall-clock timings meaningless as a reported metric.

Each architecture gets its own wandb run (named after the arch), its own
checkpoint directory, and its own eval report. A failure in one model is logged
and the rest continue — a single bad architecture should not cost the sweep.

Usage:
    python -m training.run_all --archs swinir maxvit convnext --epochs 30
    python -m training.run_all --all --epochs 60
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ALL_ARCHS = ["swin", "unet", "deepsd", "swinir", "restormer", "segformer", "vit",
             "maxvit", "convnext", "edsr", "esrt", "swinir_light"]

# Cheap models first (the user's "baselines before transformers" ordering):
# every one of these fits in ~2.5 h total on the South-Central domain.
TIER_A = ["deepsd", "edsr", "vit", "esrt", "segformer", "swinir_light"]


def ckpt_path(arch: str) -> Path:
    base = Path("checkpoints")
    return (base / "best.pt") if arch == "swin" else (base / arch / "best.pt")


def run(cmd: list[str]) -> bool:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode == 0


def main(archs: list[str], epochs: int, skip_eval: bool,
         skip_existing: bool = False) -> None:
    py = sys.executable
    timings: dict[str, float] = {}
    failed: list[str] = []

    for arch in archs:
        # Restartability: a sweep is many hours, and an interrupted machine
        # should not cost the architectures that already finished. A report on
        # disk means that arch completed training AND evaluation.
        report = Path("outputs") / f"eval_report_{arch}.json"
        if skip_existing and report.exists():
            print(f"\n=== {arch}: {report} exists, skipping", flush=True)
            continue

        print(f"\n{'=' * 70}\n=== {arch} ({epochs} epochs)\n{'=' * 70}", flush=True)
        t0 = time.time()
        if not run([py, "-u", "-m", "training.train",
                    "--arch", arch, "--epochs", str(epochs)]):
            print(f"[run_all] {arch} TRAINING FAILED — continuing")
            failed.append(arch)
            continue
        timings[arch] = time.time() - t0

        if skip_eval:
            continue
        ckpt = ckpt_path(arch)
        if not ckpt.exists():
            print(f"[run_all] {arch}: no checkpoint at {ckpt}, skipping eval")
            failed.append(arch)
            continue
        if not run([py, "-u", "-m", "evaluation.evaluate", "--ckpt", str(ckpt)]):
            print(f"[run_all] {arch} EVAL FAILED — continuing")
            failed.append(arch)

    print(f"\n{'=' * 70}\n=== summary\n{'=' * 70}")
    for arch, secs in timings.items():
        print(f"{arch:12s} train {secs / 60:6.1f} min")
    if failed:
        print(f"FAILED: {', '.join(failed)}")

    if not skip_eval:
        run([py, "-u", "-m", "evaluation.compare"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=None)
    ap.add_argument("--all", action="store_true", help=f"run {ALL_ARCHS}")
    ap.add_argument("--tier-a", action="store_true", help=f"run {TIER_A}")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip archs that already have an eval report "
                         "(resume an interrupted sweep)")
    args = ap.parse_args()
    archs = TIER_A if args.tier_a else (
        ALL_ARCHS if args.all else (args.archs or ALL_ARCHS))
    main(archs, args.epochs, args.skip_eval, args.skip_existing)
