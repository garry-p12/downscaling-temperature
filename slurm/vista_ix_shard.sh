#!/bin/bash
#SBATCH -J ixshard
#SBATCH -o logs/ixshard_%j.out
#SBATCH -e logs/ixshard_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:45:00
#SBATCH -A ATM23014
#
# One disjoint slice of the 20-run work list. SHARD/NSHARDS come from the
# environment (sbatch --export), so the same script serves every shard.
#
# `gh`, not `gh-dev`: the qdevelopment QoS allows only ONE running job per
# user (MaxJobsPU=1), so gh-dev cannot give parallelism at all. qgh allows 20
# concurrent. 1h45 keeps each job small enough to backfill.
#
# Shards never build stores — they exit if one is missing. Submit with
# --dependency=afterok on the build job.
set -uo pipefail
cd "${WORK}/downscaling"
mkdir -p logs
echo "=== node $(hostname)  shard ${SHARD}/${NSHARDS}  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export PYTHON="${WORK}/miniforge3/envs/dsc/bin/python"
export DOWNSCALE_MAX_WORKERS=8
bash slurm/interaction_target_body.sh
