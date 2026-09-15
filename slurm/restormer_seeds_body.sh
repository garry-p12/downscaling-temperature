#!/bin/bash
# Is Restormer's 0.7528 real, or seed luck?
#
# The single-seed transformer sweep put Restormer 0.0307 below DeepSD, which is
# 2.3x DeepSD's seed sd. That looks decisive and is not: the original multi-seed
# study found Restormer has the LARGEST seed spread of any architecture
# (sd 0.0251, range 0.745-0.800), so one draw at 0.753 sits inside its own
# historical range. The same trap produced the original table's fake DeepSD win.
#
# Five seeds on the winning land-cover channel set, same protocol, evaluated
# against the existing DeepSD five-seed runs in one pass.
set -uo pipefail
PYTHON="${PYTHON:-python}"; DEVICE="${DEVICE:-cuda}"
ZARR="${ZARR:-data_store_sc/super/dataset.zarr}"
PRECISION="${PRECISION:-fp32}"; EPOCHS="${EPOCHS:-20}"
OUT="${OUT:-image_outputs/restormer_seeds}"
SEEDS=(1337 7 42 2024 31337)

CHANNELS="coarse_tmp dem coastal_dist land_mask doy_sin doy_cos \
          lc_water lc_developed lc_forest lc_shrub lc_herbaceous lc_cultivated lc_wetland"

RUNS=()
for seed in "${SEEDS[@]}"; do
    tag="rs_restormer_s${seed}"; RUNS+=("$tag")
    [ -f "checkpoints/${tag}/last.pt" ] && { echo "=== ${tag}: done, skipping"; continue; }
    echo "=============== ${tag}  $(date +%H:%M:%S)"
    "$PYTHON" -u -m training.train --arch restormer --epochs "$EPOCHS" --seed "$seed" \
        --run-name "$tag" --zarr "$ZARR" --precision "$PRECISION" \
        --use-channels $CHANNELS || echo "[rs] ${tag} FAILED, continuing"
done

DONE=()
for t in "${RUNS[@]}"; do [ -f "checkpoints/${t}/last.pt" ] && DONE+=("$t"); done
for s in "${SEEDS[@]}"; do [ -f "checkpoints/v2_landcov_s${s}/last.pt" ] && DONE+=("v2_landcov_s${s}"); done
[ ${#DONE[@]} -gt 0 ] && {
    echo "=============== holdout eval  $(date +%H:%M:%S)"
    "$PYTHON" -u -m evaluation.holdout_eval --archs "${DONE[@]}" \
        --n-boot 500 --device "$DEVICE" --out "$OUT" || true; }
echo "=== done $(date +%H:%M:%S)"
