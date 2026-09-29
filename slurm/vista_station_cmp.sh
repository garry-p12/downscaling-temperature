#!/bin/bash
#SBATCH -J stncmp
#SBATCH -o logs/stncmp_%j.out
#SBATCH -e logs/stncmp_%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A ATM23014
#
# The comparison the whole AORC chain exists for: two models differing ONLY in
# which product they were trained toward, both scored against NOAA ISD
# thermometers — a judge neither product controls. Identical predictors,
# architecture, schedule, holdout and scored cells.
#
# Runs on a COMPUTE node, not the login node. Reading the time:1-chunked AORC
# store spawns a zstd decode thread per chunk, and login nodes cap user threads
# ("RuntimeError: can't start new thread"). ISD records are pre-fetched on the
# login node because compute nodes have no outbound network.
#
# Each arm is a separate invocation: they need different stores, climatologies
# and normalizers, and mixing those would silently compare mismatched units.
set -uo pipefail
cd "${WORK}/downscaling"
source "${WORK}/miniforge3/etc/profile.d/conda.sh"
conda activate dsc
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

for st in KAUS_Austin_Bergstrom KATT_Austin_Executive; do
    echo "################## ${st} — AORC-trained"
    DOWNSCALE_CONFIG_data=configs/data_sc_aorc.yaml \
        python -u -m evaluation.station_check \
            --archs aorc_s1337 aorc_s7 aorc_s42 \
            --year 2023 --station "$st" || echo "ARM FAILED"
    echo "################## ${st} — ERA5-Land-trained"
    DOWNSCALE_CONFIG_data=configs/data_sc_super.yaml \
        python -u -m evaluation.station_check \
            --archs v2_landcov_s1337 v2_landcov_s42 v2_landcov_s7 \
            --year 2023 --station "$st" || echo "ARM FAILED"
done
echo "=== done $(date +%H:%M:%S)"
