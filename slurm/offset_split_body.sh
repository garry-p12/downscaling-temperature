#!/bin/bash
# Does splitting the loss along OUR error budget beat one scalar loss?
#
# THE IDEA, TAKEN FROM Subich et al. 2025 (ICML) RATHER THAN THEIR LOSS.
# Their contribution is not the max() trick. It is that one scalar loss
# conflates error modes with different fixability, so the model spends its
# capacity on whichever mode is cheapest to reduce rather than the one that
# matters. Their split is amplitude vs phase per scale, because that is what a
# forecast needs. Ours is measured in README 5.4 and is completely different:
#
#     spatially-uniform daily offset   50.8% of holdout residual variance
#     time-invariant spatial pattern    2.8%
#     space-time remainder             46.4%
#
# Half of what we score is one number per day applied to the whole region, a
# product-level disagreement between POWER and ERA5-Land that no spatial model
# can fix -- and 5.4 showed it is separately predictable from its own lag-1
# (out-of-sample R^2 +0.221, better than anything spatial). Yet the loss keeps
# charging for it, so gradient that could buy spatial structure buys nothing.
#
# training/losses.py decomposed_loss makes the split explicit:
#     loss = spatial(pred - mu_p, target - mu_t) + offset_weight * |mu_p - mu_t|
#
# ARMS. The offset arm alone would confound two changes at once, so the
# full-field control is not optional:
#   of1_*  offset_weight 1.0, full fields  -- isolates "full fields" by itself
#   of0_*  offset_weight 0.0, full fields  -- the hypothesis
#   baseline: the published v2_landcov_s*, patched, single-term loss (not
#             retrained; retraining it would move the numbers being compared to)
#
# Full fields are REQUIRED, not a free choice. Measured on the training split,
# the offset has sd 0.463 degC while the gap between a 48x48 patch mean and the
# true domain mean has sd 0.376 -- 81% of the signal. Splitting on a patch mean
# would remove real spatial structure instead of the bias. It costs nothing:
# there are 4x fewer steps of 4x the pixels.
#
# PRE-REGISTERED, BEFORE ANY RUN. of0 has no daily offset by construction, so
# its RAW RMSE MUST get worse -- that is the arm working as designed, not
# failing. It would be restored at inference by the AR(1) term of 5.4. Judge on
# SPATIAL skill only:
#   1. residual correlation and SSIM vs the published seed sd (0.0092, 0.0039)
#   2. amplitude ratio and coherence at 22-99 km (evaluation/spectral.py)
#   3. of1 vs baseline isolates the full-field change; of0 vs of1 isolates the
#      offset change. Reporting of0 against the baseline alone would be wrong.

set -uo pipefail

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
ZARR="${ZARR:-data_store_sc/super/dataset.zarr}"
PRECISION="${PRECISION:-fp32}"     # never bf16 on GPU: halves SSIM (README 9)
EPOCHS="${EPOCHS:-20}"
OUT="${OUT:-image_outputs/offset_split}"
WITH_AMSE="${WITH_AMSE:-0}"        # 1 also runs the paper's loss as a reference

V2=$(grep -A2 '^V2="' slurm/covariate_ablation_body.sh \
     | tr '\n' ' ' | tr -d '\\' | sed 's/.*V2="//; s/".*//')
NCH=$(echo $V2 | wc -w)
if [ "$NCH" -ne 13 ]; then
    echo "parsed $NCH channels, expected 13: $V2"
    echo "fix this parser rather than training an arm that does not match the"
    echo "baseline it is compared against."
    exit 1
fi
echo "V2 channels ($NCH): $V2"

# Overridable so a smoke test can run one seed for one epoch before a real
# submission spends a queue slot on five.
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-1337 7 42 2024 31337}"
RUNS=()

run_arm () {                       # $1 tag prefix, $2... extra train flags
    local prefix="$1"; shift
    for seed in "${SEEDS[@]}"; do
        local tag="${prefix}_s${seed}"
        RUNS+=("$tag")
        if [ -f "checkpoints/${tag}/last.pt" ]; then
            echo "=== ${tag}: done already, skipping"; continue
        fi
        echo "=============== ${tag}  $(date +%H:%M:%S)"
        "$PYTHON" -u -m training.train \
            --arch deepsd --epochs "$EPOCHS" --seed "$seed" \
            --run-name "$tag" --zarr "$ZARR" --precision "$PRECISION" \
            --use-channels $V2 "$@" \
            || echo "[offset] ${tag} FAILED, continuing"
    done
}

# ARMS selects which arms this invocation trains. Splitting the experiment
# across several SHORT jobs is what lets Slurm backfill it into gaps: a 20 min
# request slots into holes a 2 h request never fits. The arms are independent,
# so they need no ordering between them, and every arm skips any seed whose
# last.pt already exists — so the jobs are idempotent and a wall-time kill
# costs only the run that was in flight.
# ${ARMS-...} not ${ARMS:-...}: an explicitly EMPTY ARMS means "train nothing,
# just evaluate", and the colon form would silently turn that into a full
# retrain of both arms.
ARMS="${ARMS-of1 of0}"
# Full-field batches with worker processes deadlock here: job 1018551 sat at
# 0% GPU and 0.4% CPU for 13 minutes without logging one step, while the same
# code with in-process loading runs fine. The store is small and local, so
# workers buy nothing anyway.
NW="${NW---num-workers 0}"
DO_EVAL="${DO_EVAL:-1}"

for arm in $ARMS; do
    case "$arm" in
        of1) run_arm of1 --patch-size 0 --offset-weight 1.0 $NW ;;
        of0) run_arm of0 --patch-size 0 --offset-weight 0.0 $NW ;;
        amse) run_arm v2_amse --pixel-loss amse ;;
        *) echo "unknown arm '$arm'"; exit 1 ;;
    esac
done
[ "$WITH_AMSE" = "1" ] && run_arm v2_amse --pixel-loss amse

# An eval-only job still needs the full arm list to score, so rebuild it from
# the arms the experiment defines rather than from what this job happened to
# train.
if [ "$DO_EVAL" = "1" ]; then
    RUNS=()
    for prefix in of1 of0 v2_amse; do
        for seed in "${SEEDS[@]}"; do RUNS+=("${prefix}_s${seed}"); done
    done
fi

DONE=()
for tag in "${RUNS[@]}"; do
    [ -f "checkpoints/${tag}/last.pt" ] && DONE+=("$tag")
done
for s in "${SEEDS[@]}"; do
    [ -f "checkpoints/v2_landcov_s${s}/last.pt" ] && DONE+=("v2_landcov_s${s}")
done

if [ "$DO_EVAL" != "1" ]; then
    echo "=== DO_EVAL=0, trained only: ${#DONE[@]} checkpoints present"
elif [ ${#DONE[@]} -gt 0 ]; then
    echo "=============== holdout eval  $(date +%H:%M:%S)"
    "$PYTHON" -u -m evaluation.holdout_eval \
        --archs "${DONE[@]}" --n-boot 500 --device "$DEVICE" --out "$OUT" || true
    # last.pt, not best.pt: best is selected on validation RMSE, and of0 is
    # designed to sacrifice RMSE, so selecting on it would discard the arm.
    for W in last best; do
        echo "=============== spectra (${W}.pt)  $(date +%H:%M:%S)"
        "$PYTHON" -u -m evaluation.spectral \
            --archs "${DONE[@]}" --zarr "$ZARR" --device "$DEVICE" --ckpt "$W" \
            --out "${OUT}/spectral_report_${W}.json" || true
    done
fi
echo "=== done $(date +%H:%M:%S)"
