#!/bin/bash
#SBATCH -J tfcov
#SBATCH -o logs/tfcov_%j.out
#SBATCH -e logs/tfcov_%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014
set -uo pipefail
cd "${WORK}/downscaling"
mkdir -p logs
echo "=== node $(hostname)  $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
source "${WORK}/miniforge3/etc/profile.d/conda.sh"
conda activate dsc
# V2 (13 channels) is the default set; the body no longer carries a copy.
export DOWNSCALE_CONFIG_data=configs/data_sc_v2.yaml
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTHON=python DEVICE=cuda PRECISION=fp32
export ZARR="data_store_sc/super/dataset.zarr" EPOCHS=20
source slurm/transformers_body.sh
