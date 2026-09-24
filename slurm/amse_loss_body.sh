#!/bin/bash
# Does the loss, not the data, cap the fine-scale skill?
#
# Measured on the holdout with evaluation/spectral.py, the published V2 model's
# amplitude ratio tracks its coherence down the whole spectrum (0.42 at 38 km
# against a coherence of 0.11), giving an effective resolution of ~165 km on a
# 10 km grid. That is what a pixel loss asks for: when a scale has correlation
# rho < 1, the loss-optimal amplitude is rho, so the model minimises loss by
# SHRINKING what it cannot predict. AMSE (Subich et al. 2025, ICML) removes the
# incentive by decoupling the amplitude and decorrelation terms in spectral
# space. training/losses.py adapts it from spherical harmonics to a windowed 2D
# DFT, which is what a limited-area grid allows.
#
# ONE VARIABLE CHANGES. Same architecture, channels, epochs, precision and
# seeds as the published v2_landcov arm -- only --pixel-loss differs. The
# baseline arm is NOT retrained: checkpoints/v2_landcov_s* already exist and
# retraining them would risk moving the very numbers we are comparing against.
#
# PRE-REGISTERED, BEFORE ANY RUN. Sharpening is expected to make RMSE slightly
# WORSE: loss-optimal smoothing minimises RMSE by construction, and README 5.4
# shows RMSE is dominated by a spatially uniform daily offset no spatial model
# can reach. So RMSE is NOT the success criterion. The claim is judged on:
#   1. amplitude ratio at 38-99 km moving toward 1.0 (primary)
#   2. SSIM and residual correlation vs the published between-seed sd
#      (0.0039 and 0.0092 respectively)
#   3. RMSE not degrading by more than the seed sd, 0.0097 degC (a guard)
# A win on 1 with 2 flat and 3 held is still a real result, and is what the
# paper's own appendix B.3 predicts is UNLIKELY for 2 m temperature -- it finds
# elevation already anchors the fine scales. That is the point of running it.

set -uo pipefail

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
ZARR="${ZARR:-data_store_sc/super/dataset.zarr}"
PRECISION="${PRECISION:-fp32}"     # never bf16 on GPU: halves SSIM (README 9)
EPOCHS="${EPOCHS:-20}"
OUT="${OUT:-image_outputs/amse_loss}"

SEEDS=(1337 7 42 2024 31337)

# Parsed from the ablation script so the two arms cannot drift apart.
V2=$(grep -A2 '^V2="' slurm/covariate_ablation_body.sh \
     | tr '\n' ' ' | tr -d '\\' | sed 's/.*V2="//; s/".*//')
NCH=$(echo $V2 | wc -w)
if [ "$NCH" -ne 13 ]; then
    echo "parsed $NCH channels from covariate_ablation_body.sh, expected 13: $V2"
    echo "the ablation definition changed shape - fix this parser rather than"
    echo "training an arm that does not match the baseline it is compared to."
    exit 1
fi
echo "V2 channels ($NCH): $V2"

RUNS=()
for seed in "${SEEDS[@]}"; do
    tag="v2_amse_s${seed}"
    RUNS+=("$tag")
    if [ -f "checkpoints/${tag}/last.pt" ]; then
        echo "=== ${tag}: done already, skipping"
        continue
    fi
    echo "=============== ${tag}  $(date +%H:%M:%S)"
    "$PYTHON" -u -m training.train \
        --arch deepsd --epochs "$EPOCHS" --seed "$seed" \
        --run-name "$tag" --zarr "$ZARR" --precision "$PRECISION" \
        --pixel-loss amse --use-channels $V2 \
        || echo "[amse] ${tag} FAILED, continuing"
done

# Evaluate both arms together so the comparison is like-for-like rather than
# against a number copied out of an older report.
DONE=()
for tag in "${RUNS[@]}"; do
    [ -f "checkpoints/${tag}/last.pt" ] && DONE+=("$tag")
done
for s in "${SEEDS[@]}"; do
    [ -f "checkpoints/v2_landcov_s${s}/last.pt" ] && DONE+=("v2_landcov_s${s}")
done

if [ ${#DONE[@]} -gt 0 ]; then
    echo "=============== holdout eval  $(date +%H:%M:%S)"
    "$PYTHON" -u -m evaluation.holdout_eval \
        --archs "${DONE[@]}" --n-boot 500 --device "$DEVICE" --out "$OUT" || true
    # Both checkpoints on purpose. best.pt is selected on validation RMSE, and
    # RMSE is minimised by blurring -- selecting on it while testing a loss
    # designed to stop blurring would quietly decide the experiment. last.pt is
    # the honest comparison here; best.pt is reported for continuity with every
    # other number in the repo.
    for W in last best; do
        echo "=============== spectra (${W}.pt)  $(date +%H:%M:%S)"
        "$PYTHON" -u -m evaluation.spectral \
            --archs "${DONE[@]}" --zarr "$ZARR" --device "$DEVICE" --ckpt "$W" \
            --out "${OUT}/spectral_report_${W}.json" || true
    done
fi
echo "=== done $(date +%H:%M:%S)"
