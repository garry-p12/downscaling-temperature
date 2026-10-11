#!/bin/bash
#SBATCH -J ixbuild
#SBATCH -o logs/ixbuild_%j.out
#SBATCH -e logs/ixbuild_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:50:00
#SBATCH -A ATM23014
#
# Build both interaction stores and train nothing. Deliberately separate from
# the training shards: two shards writing one zarr at once would interleave
# and corrupt it, so exactly one job builds and the shards depend on it.
#
# 50 minutes rather than a padded few hours, because a short request is what
# makes a job backfillable into the gaps ahead of larger reservations.
set -uo pipefail
cd "${WORK}/downscaling"
mkdir -p logs
echo "=== node $(hostname)  $(date)"
export PYTHON="${WORK}/miniforge3/envs/dsc/bin/python"
export DOWNSCALE_MAX_WORKERS=8
export BUILD_ONLY=1
bash slurm/interaction_target_body.sh
