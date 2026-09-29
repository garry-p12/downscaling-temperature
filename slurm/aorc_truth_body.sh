#!/bin/bash
# Train the SAME model against AORC instead of ERA5-Land.
#
# This is the direct test of "the bottleneck is the target, not the model".
# Architecture, predictors, schedule and seeds are identical to the published
# v2_landcov arm; only the truth product differs. Both are then scored against
# NOAA ISD thermometers — a judge neither product controls.
#
# Prediction under the thesis: the AORC-trained model should be closer to the
# stations. If it is not, the ceiling belongs to the TASK rather than to the
# choice of target, which is a different and equally reportable result.
#
# Coverage differs between the two builds (AORC is CONUS-only; ~74% finite
# south of 29 N), so the scored-cell count is printed and must be reported
# beside any comparison.

set -uo pipefail
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-20}"
PRECISION="${PRECISION:-fp32}"
ZARR="data_store_sc/aorc/dataset.zarr"

V2=$(grep -A2 '^V2="' slurm/covariate_ablation_body.sh \
     | tr '\n' ' ' | tr -d '\\' | sed 's/.*V2="//; s/".*//')
NCH=$(echo $V2 | wc -w)
[ "$NCH" -ne 13 ] && { echo "parsed $NCH channels, expected 13"; exit 1; }

if [ ! -d "$ZARR" ]; then
    echo "=============== building the AORC-truth store $(date +%H:%M:%S)"
    DOWNSCALE_CONFIG_data=configs/data_sc_aorc.yaml \
        "$PYTHON" -u -m data.build_dataset || { echo "BUILD FAILED"; exit 1; }
fi

export DOWNSCALE_CONFIG_data=configs/data_sc_aorc.yaml
# ${VAR-default}, not ${VAR:-default}: an explicitly EMPTY SEEDS_OVERRIDE means
# "build the store and train nothing", which the colon form would silently turn
# into a full five-seed run.
read -r -a SEEDS <<< "${SEEDS_OVERRIDE-1337 7 42 2024 31337}"
[ ${#SEEDS[@]} -eq 0 ] && echo "=== SEEDS_OVERRIDE empty: store only, no training"
for seed in "${SEEDS[@]}"; do
    tag="aorc_s${seed}"
    [ -f "checkpoints/${tag}/last.pt" ] && { echo "=== $tag done, skipping"; continue; }
    echo "=============== ${tag} $(date +%H:%M:%S)"
    "$PYTHON" -u -m training.train --arch deepsd --epochs "$EPOCHS" --seed "$seed" \
        --run-name "$tag" --zarr "$ZARR" --precision "$PRECISION" \
        --use-channels $V2 || echo "[aorc] $tag FAILED, continuing"
done
echo "=== done $(date +%H:%M:%S)"
