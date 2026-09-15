#!/bin/bash
# Covariate ablation: the experiment itself, independent of where it runs.
#
# Sourced by slurm/vista_covariate_ablation.sh and runnable directly on a
# laptop. ONE definition of the channel lists on purpose — the whole experiment
# rests on the variants differing only in the channels named here, so a second
# copy that could drift would be a correctness hazard, not just duplication.
#
# Caller must have set PYTHON (interpreter) and DEVICE (cuda|mps|cpu) and must
# export DOWNSCALE_CONFIG_data=configs/data_sc_super.yaml.
#
# Restartable: any run whose checkpoint already exists is skipped.

set -uo pipefail

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cpu}"
ZARR="${ZARR:-data_store_sc/super/dataset.zarr}"
EPOCHS="${EPOCHS:-20}"
OUT="${OUT:-image_outputs/covariate_ablation}"
# fp32 on GPU, always. bf16 autocast halves SSIM on this loss (RESULTS.md s8),
# and it engages ONLY on CUDA -- so a GPU sweep silently differs from a laptop
# one unless this is pinned. The old Tier-B script did it by sed-ing the YAML.
PRECISION="${PRECISION:-fp32}"
SEEDS=(1337 7 42 2024 31337)

# --------------------------------------------------------------------------- #
# Variants. Channel lists are written out in full rather than composed, so the
# log is an unambiguous record of exactly what each run saw.
# --------------------------------------------------------------------------- #
BASE="coarse_tmp dem landcover coastal_dist urban_frac land_mask doy_sin doy_cos"

# V1 adds terrain SHAPE. slope_mean is deliberately excluded: r = 0.87 with
# elev_std, and a near-duplicate channel spends detection budget for nothing.
V1="$BASE elev_std tpi northness eastness"

# V2 replaces the majority class AND the hand-rolled urban fraction with a full
# composition. urban_frac must go: r = 0.996 with lc_developed, so keeping it
# would make this a test of a duplicated channel. lc_barren excluded (sd 0.008).
V2="coarse_tmp dem coastal_dist land_mask doy_sin doy_cos \
    lc_water lc_developed lc_forest lc_shrub lc_herbaceous lc_cultivated lc_wetland"

# V3 drops the coast. READ THE RESULT CAREFULLY: coastal_dist is r = 0.740 with
# dem here, so "no effect" means "dem already carried it", not "marine air does
# not matter".
V3="coarse_tmp dem landcover urban_frac land_mask doy_sin doy_cos"

# V4 isolates elev_std from the rest of the terrain block. V1 bundled four
# channels and came out flat-to-slightly-worse; this asks whether sub-grid
# relief on its own does anything, or whether the aspect pair (visibly noisy
# over flat terrain) was diluting it.
V4="$BASE elev_std"

run_variant () {
    local name="$1" channels="$2" seed tag
    for seed in "${SEEDS[@]}"; do
        tag="${name}_s${seed}"
        if [ -f "checkpoints/${tag}/last.pt" ]; then
            echo "=== ${tag}: checkpoint exists, skipping"
            continue
        fi
        echo "=============== ${tag}  $(date +%H:%M:%S)"
        "$PYTHON" -u -m training.train \
            --arch deepsd --epochs "$EPOCHS" --seed "$seed" \
            --run-name "$tag" --zarr "$ZARR" --precision "$PRECISION" \
            --use-channels $channels \
            || echo "[covabl] ${tag} FAILED, continuing"
    done
}

VARIANTS=(v0_base v1_terrain v2_landcov v3_nocoast v4_elevstd)

run_variant v0_base    "$BASE"
run_variant v1_terrain "$V1"
run_variant v2_landcov "$V2"
run_variant v3_nocoast "$V3"
run_variant v4_elevstd "$V4"

# --------------------------------------------------------------------------- #
# One evaluation pass over every run, so all variants share an identical
# protocol and a single bootstrap draw.
# --------------------------------------------------------------------------- #
ALL_RUNS=()
for v in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do ALL_RUNS+=("${v}_s${seed}"); done
done

echo "=============== holdout eval  $(date +%H:%M:%S)"
"$PYTHON" -u -m evaluation.holdout_eval \
    --archs "${ALL_RUNS[@]}" --n-boot 500 --device "$DEVICE" --out "$OUT" || true

echo "=== done $(date +%H:%M:%S)"
