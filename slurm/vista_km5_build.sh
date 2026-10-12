#!/bin/bash
#SBATCH -J km5build
#SBATCH -o logs/km5build_%j.out
#SBATCH -e logs/km5build_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A ATM23014
#
# Build the 5 km AORC store, then run the feasibility probe against it.
# Trains nothing.
#
# WHY THIS IS A COMPUTE JOB AND NOT A LOGIN-NODE COMMAND.
# data/build_dataset.py stacks the whole (T, C, y, x) input array in memory
# before writing. At 10 km with 21 channels that is 1.4 GB; at 5 km it is
# 5.7 GB, with roughly a second copy live during the stack. A login node will
# either refuse it or take the node down for everyone else on it.
#
# WHY 90 MINUTES. The AORC download is the long pole and it streams from S3.
# A short request backfills into the gaps ahead of larger reservations;
# padding it to "safe" six hours means sitting behind a 340-deep queue.
# Every step below is resumable, so an overrun costs a resubmit, not a redo.
set -uo pipefail
cd "${WORK}/downscaling"
mkdir -p logs outputs/km5
echo "=== node $(hostname)  $(date)"

export PYTHON="${WORK}/miniforge3/envs/dsc/bin/python"
export DOWNSCALE_MAX_WORKERS=8
CFG=configs/data_sc_5km.yaml
STORE=data_store_sc/km5/dataset.zarr

# ---- 1. AORC 2019-2023 -----------------------------------------------------
# All five years are already here (verified 1,826 days, 2019-01-01..2023-12-31),
# so this is a no-op that re-checks rather than re-downloads: download_aorc
# skips any year whose store already covers the window, and writes through a
# .partial rename so an interrupted year can never be mistaken for a complete
# one. Kept in the script so the job is self-contained if the raw store is ever
# cleaned up.
echo "--- [1/3] AORC 2019-2023 (expected: all five skipped)"
$PYTHON -m data.download_aorc --start 2019-01-01 --end 2023-12-31 || {
    echo "!!! AORC check failed; nothing downstream can be trusted"; exit 1; }

# ---- 2. the 5 km store -----------------------------------------------------
# check_store tests VALIDITY, not existence. A bare [ -d ] once reported a
# half-written store as "already built" and every result downstream of it was
# quietly wrong, which is why this guard is a python call and not a test.
echo "--- [2/3] 5 km store"
if $PYTHON -m data.check_store "$STORE" --channels 21 --days 1826 >/dev/null 2>&1; then
    echo "    $STORE already valid, skipping build"
else
    DOWNSCALE_CONFIG_data=$CFG $PYTHON -m data.build_dataset || {
        echo "!!! build failed"; exit 1; }
    $PYTHON -m data.check_store "$STORE" --channels 21 --days 1826 || {
        echo "!!! store built but did not validate"; exit 1; }
fi

# ---- 3. the probe ----------------------------------------------------------
# Writes outputs/km5/probe.json: the three sanity checks, the 5 km error
# budget, and the bicubic control that any native 5 km model has to beat.
echo "--- [3/3] probe"
$PYTHON scripts/probe_5km.py --arms aorc_s1337 aorc_s7 aorc_s42 --device cuda

# The station panel carries grid indices, so it is only valid on the grid it
# was built for. Regenerate it for 5 km under its own name; the 10 km panel
# stays where it is.
echo "--- station panel for the 5 km grid"
$PYTHON -m evaluation.station_panel --build-panel --store "$STORE"

echo "=== done $(date)"
