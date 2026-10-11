#!/bin/bash
# Three follow-ups on the AORC target, which the station panel established as
# the right one (AORC beats ERA5-Land at 44 of 44 stations over five years).
#
# ARM A — ARCHITECTURE, RE-ASKED
#   Ten architectures tied earlier. That null was measured against ERA5-Land,
#   which carries 26% of AORC's sub-60 km amplitude at ~zero coherence: there
#   was no fine structure for extra capacity to capture, so the test could not
#   have come out any other way. Against a target that does resolve those
#   scales the question is open again, and this is the cheapest way to ask it.
#
# ARM B — SPECTRAL LOSS, RE-ASKED
#   The AMSE loss rewards matching the target's power spectrum. On ERA5-Land it
#   did not transfer, which is what should happen when the fine-scale content
#   it is asked to match is itself uncorrelated with reality. On AORC there is
#   something genuine to match.
#
# ARM C — TEMPORAL CONTEXT
#   The model sees one day. Anomalies persist and advect, so the coarse field
#   on t-2..t+2 carries information about t. The lags are of the INPUT, never
#   of the target, so nothing is leaked.
#
# All arms share the AORC target, the intersected land mask and the same seeds,
# so a difference is the architecture, the loss or the channels — never the
# data. Arm C needs a different store; its baseline is the deepsd arm of A on
# the same seeds, and the two are reported side by side.

set -uo pipefail
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-20}"
PRECISION="${PRECISION:-fp32}"          # bf16 halves SSIM on this loss (README 9)
export DOWNSCALE_CONFIG_data="${DOWNSCALE_CONFIG_data:-configs/data_sc_aorc.yaml}"
export DOWNSCALE_MAX_WORKERS="${DOWNSCALE_MAX_WORKERS:-8}"

# NUM_WORKERS=0 IS NOT A PERFORMANCE CHOICE. With the default worker count,
# every job on the wide stores hung on the first dataloader batch and was
# killed at its wall limit having produced nothing — thirteen jobs, no epochs,
# no error message. Do not "restore" workers without reproducing that probe.
export NUM_WORKERS="${NUM_WORKERS:-0}"

BASE_STORE="data_store_sc/aorc/dataset.zarr"
CTX_STORE="data_store_sc/aorc_ctx/dataset.zarr"

ARCHS="${ARCHS:-deepsd unet convnext swinir}"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE-1337 7 42}"

# Validity, not existence: a killed build leaves a directory that passes
# `[ -d ... ]` and then trains on half a store.
store_ok () {
    "$PYTHON" -m data.check_store "$1" --days 1826 2>/dev/null
}

if ! store_ok "$CTX_STORE"; then
    echo "=============== building the temporal-context store $(date +%H:%M:%S)"
    # make_context_store writes with mode="w", which replaces any partial
    # store in place. No recursive delete is needed, and not doing one keeps
    # a typo in CTX_STORE from being destructive.
    "$PYTHON" -u -m data.make_context_store --src "$BASE_STORE" \
        --out "$CTX_STORE" --lags 1 2 || { echo "CTX BUILD FAILED"; exit 1; }
    store_ok "$CTX_STORE" || { echo "CTX STORE INVALID AFTER BUILD"; exit 1; }
fi

# One flat work list so a shard is a stride over it and slices are disjoint.
WORK=()
for a in $ARCHS; do
    for s in "${SEEDS[@]}"; do WORK+=("arch-${a} $a $BASE_STORE $s l1"); done
done
for s in "${SEEDS[@]}"; do WORK+=("amse deepsd $BASE_STORE $s amse"); done
for s in "${SEEDS[@]}"; do WORK+=("ctx deepsd $CTX_STORE $s l1"); done

SHARD="${SHARD:-0}"; NSHARDS="${NSHARDS:-1}"
echo "=== ${#WORK[@]} runs total; this is shard $SHARD of $NSHARDS"

for i in "${!WORK[@]}"; do
    [ $(( i % NSHARDS )) -eq "$SHARD" ] || continue
    set -- ${WORK[$i]}
    tag=$1 arch=$2 store=$3 seed=$4 loss=$5
    name="${tag}_s${seed}"
    if [ -f "checkpoints/$name/last.pt" ]; then
        echo "--- $name already done, skipping"; continue
    fi

    # Claim the run. The same shard may be queued on more than one partition,
    # and without a claim two jobs would write one checkpoint directory. A lock
    # older than 45 minutes belonged to a killed job and is taken over.
    LOCK="checkpoints/$name/.running"
    if [ -f "$LOCK" ] && [ -n "$(find "$LOCK" -mmin -45 2>/dev/null)" ]; then
        echo "--- $name is claimed by $(cat "$LOCK" 2>/dev/null), skipping"; continue
    fi
    mkdir -p "checkpoints/$name"
    echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)" > "$LOCK"
    trap 'rm -f "$LOCK"' EXIT

    echo "=============== [$i] $name  arch=$arch loss=$loss  $(date +%H:%M:%S)"
    "$PYTHON" -u -m training.train \
        --arch "$arch" --epochs "$EPOCHS" --seed "$seed" \
        --precision "$PRECISION" --zarr "$store" \
        --pixel-loss "$loss" --num-workers "$NUM_WORKERS" \
        --run-name "$name" || echo "[followup] $name FAILED, continuing"
    rm -f "$LOCK"
    trap - EXIT
done

echo "=============== shard $SHARD done $(date)"
ls -d checkpoints/{arch-*,amse,ctx}_s* 2>/dev/null | sed 's|checkpoints/||' | tr '\n' ' '; echo
