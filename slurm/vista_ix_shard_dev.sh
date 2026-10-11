#!/bin/bash
#SBATCH -J ixdev
#SBATCH -o logs/ixdev_%j.out
#SBATCH -e logs/ixdev_%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014
#
# One disjoint slice of the 20-run work list. SHARD/NSHARDS come from the
# environment (sbatch --export), so the same script serves every shard.
#
# gh-dev variant, submitted when `gh` gets too deep to wait on. qdevelopment
# runs only ONE job per user at a time, so three of these execute in sequence
# rather than in parallel — but a queue of 12 beats a queue of 170, and three
# serial 70-minute shards still land hours before a gh slot opens.
#
# The 2 h cap is a hard kill, which is safe here only because the body skips
# any seed whose last.pt already exists. A shard cut off mid-run loses that one
# run, not the shard.
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
