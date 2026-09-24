#!/bin/bash
# HEADROOM PROBE: how much of DeepSD's skill actually needs a nonlinearity?
#
# Before designing a new architecture (spectral front end, FNO/AFNO-style
# token mixing), answer the prior question. README 5.1 found ten architectures
# statistically tied and 5.6 found the only thing to beat the noise floor was a
# LOSS change. If an AFFINE model with a comparable receptive field matches
# DeepSD per scale, then nonlinear capacity is not the binding constraint and a
# fancier architecture is unlikely to be either.
#
# models/linear_probe.py is that control: residual correction to the
# interpolated field through a single 13x13 conv over all channels, no
# activation anywhere, initialised to the identity so epoch 0 IS the
# interpolation baseline. ~2.2k parameters against DeepSD's 210k.
#
# Protocol is identical to the published v2_landcov arm -- same channels, same
# patched sampling, same l1+0.2*(1-SSIM) loss, same 20 epochs, same five seeds
# -- so the ONLY difference is the nonlinearity.
#
# READ THE RESULT PER SCALE, NOT AS ONE NUMBER. Coherence is 0.985 at 495 km
# (interpolation already solves it) and under 0.2 below 55 km (little to
# extract). The band that decides whether more architecture is worth building
# is 99-165 km, where coherence is 0.43-0.57.

set -uo pipefail

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
ZARR="${ZARR:-data_store_sc/super/dataset.zarr}"
PRECISION="${PRECISION:-fp32}"
EPOCHS="${EPOCHS:-20}"
OUT="${OUT:-image_outputs/linear_probe}"

V2=$(grep -A2 '^V2="' slurm/covariate_ablation_body.sh \
     | tr '\n' ' ' | tr -d '\\' | sed 's/.*V2="//; s/".*//')
NCH=$(echo $V2 | wc -w)
[ "$NCH" -ne 13 ] && { echo "parsed $NCH channels, expected 13"; exit 1; }
echo "V2 channels ($NCH): $V2"

read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-1337 7 42 2024 31337}"
RUNS=()
for seed in "${SEEDS[@]}"; do
    tag="lin_s${seed}"; RUNS+=("$tag")
    [ -f "checkpoints/${tag}/last.pt" ] && { echo "=== $tag done, skipping"; continue; }
    echo "=============== ${tag}  $(date +%H:%M:%S)"
    "$PYTHON" -u -m training.train --arch linear --epochs "$EPOCHS" --seed "$seed" \
        --run-name "$tag" --zarr "$ZARR" --precision "$PRECISION" \
        --use-channels $V2 || echo "[probe] $tag FAILED, continuing"
done

DONE=()
for t in "${RUNS[@]}"; do [ -f "checkpoints/$t/last.pt" ] && DONE+=("$t"); done
for s in "${SEEDS[@]}"; do
    [ -f "checkpoints/v2_landcov_s${s}/last.pt" ] && DONE+=("v2_landcov_s${s}")
    [ -f "checkpoints/of0_s${s}/last.pt" ] && DONE+=("of0_s${s}")
done

if [ ${#DONE[@]} -gt 0 ]; then
    echo "=============== holdout eval  $(date +%H:%M:%S)"
    "$PYTHON" -u -m evaluation.holdout_eval --archs "${DONE[@]}" --n-boot 500 \
        --device "$DEVICE" --out "$OUT" || true
    echo "=============== spectra  $(date +%H:%M:%S)"
    "$PYTHON" -u -m evaluation.spectral --archs "${DONE[@]}" --zarr "$ZARR" \
        --device "$DEVICE" --ckpt last --out "${OUT}/spectral_report_last.json" || true
fi
echo "=== done $(date +%H:%M:%S)"
