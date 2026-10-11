#!/bin/bash
# Target product x weather-modulated terrain: a 2 x 2, five seeds per cell.
#
# TWO QUESTIONS AT ONCE, DELIBERATELY CROSSED
#   (1) Does the TARGET limit the downscaling? Measured over the held-out box,
#       ERA5-Land carries only 26% of AORC's sub-60 km amplitude at essentially
#       zero coherence with it. A model cannot learn structure its target does
#       not contain, so part of the 165 km effective resolution may be
#       inherited rather than a property of the model.
#   (2) Do weather-modulated terrain channels reach the space-time remainder?
#       The 5.4 budget caps static maps at 2.8%; only time-varying predictors
#       can address the other 46.4%. The dynamic fields already gave the
#       largest predictor effect in the project, but as bare extra channels.
#       The ix_* products hand the interaction over directly.
#
# Crossing them matters: interactions may only pay off against a target that
# actually resolves the scales they act on, in which case an ERA5-Land-only
# test would report a null for the wrong reason.
#
# BOTH STORES SHARE BIT-IDENTICAL INPUTS AND ONE INTERSECTED LAND MASK, so a
# difference between arms is the target or the channels, never the cells.
#
# The two truth products are kept in separate stores and separate run names on
# purpose; nothing in this script mixes them.
#
# SHARDING. Set SHARD=i NSHARDS=n to run only every n-th item of the work list.
# Shards are disjoint by construction, so several may run at once WITHOUT
# racing on a checkpoint directory. They must not race on a store build either,
# which is why BUILD_ONLY=1 exists: build once in its own job, then release the
# shards against it with --dependency=afterok.

set -uo pipefail
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-20}"
PRECISION="${PRECISION:-fp32}"     # bf16 halves SSIM on this loss (README 9)
export DOWNSCALE_CONFIG_data=configs/data_sc_dyn.yaml
export DOWNSCALE_MAX_WORKERS="${DOWNSCALE_MAX_WORKERS:-8}"

# NUM_WORKERS=0 IS NOT A PERFORMANCE CHOICE, IT IS THE ONLY SETTING THAT RUNS.
# With the default worker count every job on these 34-channel stores hung on
# the first `for x, y, m in train_ld` and was killed at its wall limit having
# produced zero batches — thirteen jobs, no epochs, no error message. With
# --num-workers 0 the same command trains normally. The 21-channel stores did
# not trip it, so the deadlock scales with per-sample size; do not "restore"
# workers without reproducing that probe first.
export NUM_WORKERS="${NUM_WORKERS:-0}"

# The 25 channels the dyn arm already uses; the ix arm adds nine more.
DYN25="coarse_tmp coarse_wind coarse_rh coarse_dewpt coarse_swdown dem \
landcover coastal_dist urban_frac land_mask doy_sin doy_cos elev_std \
slope_mean tpi northness eastness lc_water lc_developed lc_barren lc_forest \
lc_shrub lc_herbaceous lc_cultivated lc_wetland"
IX9="coarse_dewdep ix_tpi_wind ix_elevstd_wind ix_dem_dewdep ix_dem_swdown \
ix_urban_wind ix_coast_wind ix_forest_swdown ix_north_swdown"
IX34="$DYN25 $IX9"

[ "$(echo $DYN25 | wc -w)" -ne 25 ] && { echo "DYN25 is not 25 channels"; exit 1; }
[ "$(echo $IX34  | wc -w)" -ne 34 ] && { echo "IX34 is not 34 channels";  exit 1; }

# --- stores -----------------------------------------------------------------
# Validity, not existence. A killed build leaves a directory that passes
# `[ -d ... ]` and then trains on half a store; that is exactly what happened
# to ix_aorc, which was reported "already built" with no consolidated metadata.
store_ok () {
    "$PYTHON" -m data.check_store "data_store_sc/$1/dataset.zarr" \
        --channels 34 --days 1826 \
        --require ix_tpi_wind ix_north_swdown coarse_dewdep 2>/dev/null
}

build () {   # build <name> <flag> <value>
    local name=$1; shift
    if store_ok "$name"; then
        echo "=== $name already built and valid"; return 0
    fi
    echo "=== $name absent or incomplete, rebuilding from scratch"
    rm -rf "data_store_sc/$name"
    echo "=============== building $name $(date +%H:%M:%S)"
    "$PYTHON" -u -m data.make_interaction_store \
        --src data_store_sc/dyn/dataset.zarr "$@" \
        --out "data_store_sc/$name/dataset.zarr" || { echo "BUILD $name FAILED"; exit 1; }
}
if [ "${BUILD_ONLY:-0}" = "1" ] || [ "${SHARD:-0}" = "0" ] && [ "${NSHARDS:-1}" = "1" ]; then
    build ix_era  --mask-from  data_store_sc/aorc/dataset.zarr
    build ix_aorc --truth-from data_store_sc/aorc/dataset.zarr
else
    # A shard must never build: two shards building the same store at once
    # would interleave writes into one zarr and corrupt it silently.
    for s in ix_era ix_aorc; do
        store_ok "$s" || {
            echo "data_store_sc/$s is missing or incomplete, and a shard does not build."
            echo "Run the BUILD_ONLY=1 job first."; exit 1; }
    done
    echo "=== both stores present and valid"
fi
[ "${BUILD_ONLY:-0}" = "1" ] && { echo "=== BUILD_ONLY: stores done, training nothing"; exit 0; }

# --- training ---------------------------------------------------------------
# ${VAR-default}, not ${VAR:-default}: an explicitly EMPTY SEEDS_OVERRIDE means
# "build the stores and train nothing", which the colon form would silently
# turn into a full five-seed run.
read -r -a SEEDS <<< "${SEEDS_OVERRIDE-1337 7 42 2024 31337}"
[ ${#SEEDS[@]} -eq 0 ] && { echo "=== SEEDS_OVERRIDE empty: stores only"; exit 0; }

# --- work list, then shard it ----------------------------------------------
# One flat list of (tag, store, channel-set) x seed, so a shard is a simple
# stride over it and the slices are provably disjoint.
WORK=()
for spec in "dynera ix_era DYN25" "ixera ix_era IX34" \
            "dynaorc ix_aorc DYN25" "ixaorc ix_aorc IX34"; do
    set -- $spec
    for seed in "${SEEDS[@]}"; do WORK+=("$1 $2 $3 $seed"); done
done

SHARD="${SHARD:-0}"; NSHARDS="${NSHARDS:-1}"
echo "=== ${#WORK[@]} runs total; this is shard $SHARD of $NSHARDS"

for i in "${!WORK[@]}"; do
    [ $(( i % NSHARDS )) -eq "$SHARD" ] || continue
    set -- ${WORK[$i]}
    tag=$1 store=$2 chanset=$3 seed=$4
    name="${tag}_s${seed}"
    [ "$chanset" = "IX34" ] && chans="$IX34" || chans="$DYN25"

    if [ -f "checkpoints/$name/last.pt" ]; then
        echo "--- $name already done, skipping"; continue
    fi
    # Claim the run before starting it. The same shard may be queued on more
    # than one partition at once (gh is 170 deep, gh-dev is 12, so both get a
    # copy and whichever runs first wins). Without a claim, two jobs would
    # write one checkpoint directory. A lock older than STALE_MIN is assumed to
    # belong to a job that was killed and is taken over.
    LOCK="checkpoints/$name/.running"
    STALE_MIN="${STALE_MIN:-45}"
    if [ -f "$LOCK" ]; then
        if [ -n "$(find "$LOCK" -mmin "-$STALE_MIN" 2>/dev/null)" ]; then
            echo "--- $name is claimed by $(cat "$LOCK" 2>/dev/null), skipping"; continue
        fi
        echo "--- $name lock is stale, taking over from $(cat "$LOCK" 2>/dev/null)"
    fi
    mkdir -p "checkpoints/$name"
    echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)" > "$LOCK"
    trap 'rm -f "$LOCK"' EXIT
    echo "=============== [$i] $name  ($store, $chanset)  $(date +%H:%M:%S)"
    "$PYTHON" -u -m training.train \
        --arch deepsd --epochs "$EPOCHS" --seed "$seed" \
        --precision "$PRECISION" \
        --zarr "data_store_sc/$store/dataset.zarr" \
        --run-name "$name" \
        --num-workers "${NUM_WORKERS:-0}" \
        --use-channels $chans || echo "[ix] $name FAILED, continuing"
    rm -f "$LOCK"
    trap - EXIT
done

echo "=============== shard $SHARD done $(date)"
ls -d checkpoints/{dynera,ixera,dynaorc,ixaorc}_s* 2>/dev/null | sed 's|checkpoints/||' | tr '\n' ' '; echo
