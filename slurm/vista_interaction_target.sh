#!/bin/bash
#SBATCH -J ixtarget
#SBATCH -o logs/ixtarget_%j.out
#SBATCH -e logs/ixtarget_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -A ATM23014
#
# 2 x 2 x 5 seeds = 20 DeepSD runs, plus building the two stores first.
# `gh` rather than `gh-dev`: gh-dev caps at 2 h and allows three submitted jobs
# per user, and this needs more than that. The body skips any seed whose
# last.pt already exists, so a wall-time kill loses nothing and resubmitting
# continues where it stopped.
#
# Submit from the repo root:  sbatch slurm/vista_interaction_target.sh

set -uo pipefail

ROOT="${WORK}/downscaling"
cd "$ROOT"
mkdir -p logs

echo "=== node $(hostname)  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export PYTHON="${WORK}/miniforge3/envs/dsc/bin/python"
export DOWNSCALE_MAX_WORKERS=8

bash slurm/interaction_target_body.sh
