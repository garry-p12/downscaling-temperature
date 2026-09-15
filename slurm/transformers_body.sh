#!/bin/bash
# Do the covariate findings generalise beyond DeepSD?
#
# The ablation was run on DeepSD (0.13 M) because it is cheap and statistically
# tied with everything larger. That is a sound reason to use it for a DATA
# question, but it leaves one thing untested: whether the land-cover-composition
# SSIM gain is a property of the DATA or a quirk of a 3-layer SRCNN stack. This
# trains the attention architectures on the SAME winning channel set.
#
# Ordered cheapest-first on purpose. The gh-dev queue caps at 2 h and the sweep
# is restartable, so a job that runs out of wall time has still banked the cheap
# models, and resubmitting continues from there rather than starting over.
#
# Caller sets PYTHON, DEVICE, ZARR, PRECISION.

set -uo pipefail

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
ZARR="${ZARR:-data_store_sc/super/dataset.zarr}"
PRECISION="${PRECISION:-fp32}"     # never bf16 on GPU: halves SSIM (RESULTS s8)
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-1337}"
OUT="${OUT:-image_outputs/transformers_covariates}"

# The winning covariate set from the ablation: land-cover composition replacing
# the majority class and urban_frac, coastal_dist retained (dropping it flipped
# the hot-tail bias and cost residual correlation).
CHANNELS="coarse_tmp dem coastal_dist land_mask doy_sin doy_cos \
          lc_water lc_developed lc_forest lc_shrub lc_herbaceous lc_cultivated lc_wetland"

# Ascending parameter count, from the published table.
ARCHS=(esrt swinir_light maxvit segformer vit restormer swin)

RUNS=()
for arch in "${ARCHS[@]}"; do
    tag="tf_${arch}"
    RUNS+=("$tag")
    if [ -f "checkpoints/${tag}/last.pt" ]; then
        echo "=== ${tag}: done already, skipping"
        continue
    fi
    echo "=============== ${tag}  $(date +%H:%M:%S)"
    "$PYTHON" -u -m training.train \
        --arch "$arch" --epochs "$EPOCHS" --seed "$SEED" \
        --run-name "$tag" --zarr "$ZARR" --precision "$PRECISION" \
        --use-channels $CHANNELS \
        || echo "[tf] ${tag} FAILED, continuing"
done

# Evaluate whatever finished, including the DeepSD reference on the same
# channels so the comparison is like-for-like rather than against a number
# from a different report.
DONE=()
for tag in "${RUNS[@]}"; do
    [ -f "checkpoints/${tag}/last.pt" ] && DONE+=("$tag")
done
for s in 1337 7 42 2024 31337; do
    [ -f "checkpoints/v2_landcov_s${s}/last.pt" ] && DONE+=("v2_landcov_s${s}")
done

if [ ${#DONE[@]} -gt 0 ]; then
    echo "=============== holdout eval  $(date +%H:%M:%S)"
    "$PYTHON" -u -m evaluation.holdout_eval \
        --archs "${DONE[@]}" --n-boot 500 --device "$DEVICE" --out "$OUT" || true
fi
echo "=== done $(date +%H:%M:%S)"
