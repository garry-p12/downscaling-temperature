#!/bin/bash
#SBATCH -J aofu
#SBATCH -o logs/aofu_%j.out
#SBATCH -e logs/aofu_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:15:00
#SBATCH -A ATM23014
#
# One disjoint slice of the 20-run work list. SHARD/NSHARDS come from the
# environment (sbatch --export), so the same script serves every shard.
#
# Small-shard variant, for when the queue is too deep to wait on a big job.
#
# SLURM's backfill scheduler starts a job early only if it fits entirely in the
# gap before a higher-priority reservation. A 1h45 request fits almost no gaps
# on a 170-deep queue; a 40-minute one fits many. Two runs at ~10 min each plus
# torch startup sits comfortably inside 40 minutes, and qgh permits 20
# concurrent jobs, so ten of these can run at once where three could not.
#
# Safe to overlap with the larger shards: every run is claimed by a lock file
# before it starts and skipped if already complete, so two different
# partitionings of the same 20-run list cannot collide.
#
# Shards never build stores — they exit if one is missing. Submit with
# --dependency=afterok on the build job.
set -uo pipefail
cd "${WORK}/downscaling"
mkdir -p logs
echo "=== node $(hostname)  shard ${SHARD:-0}/${NSHARDS:-1}  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export PYTHON="${WORK}/miniforge3/envs/dsc/bin/python"
export DOWNSCALE_MAX_WORKERS=8
bash slurm/aorc_followup_body.sh
