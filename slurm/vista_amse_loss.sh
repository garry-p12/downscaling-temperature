#!/bin/bash
#SBATCH -J amse
#SBATCH -o logs/amse_%j.out
#SBATCH -e logs/amse_%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM23014
#
# AMSE loss experiment. See slurm/amse_loss_body.sh for the hypothesis, the
# pre-registered success criteria, and why RMSE is not one of them.
#
# gh-dev rather than gh: 2 h cap and one running job per user, but it starts in
# seconds while gh is routinely hundreds of jobs deep. Five DeepSD runs at
# ~3.5 min each fit inside the cap with room to spare, and the body script
# skips any seed whose last.pt already exists, so a wall-time kill loses
# nothing and resubmitting continues.
#
# Submit from the repo root:  sbatch slurm/vista_amse_loss.sh

set -euo pipefail

ROOT="${WORK}/downscaling"          # $WORK already ends in /vista
cd "$ROOT"
mkdir -p logs

echo "=== node $(hostname)  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source "${WORK}/miniforge3/etc/profile.d/conda.sh"
conda activate dsc
python -c "import torch; print('torch', torch.__version__,
           '| cuda', torch.cuda.is_available())" || true

export DOWNSCALE_CONFIG_data=configs/data_sc_super.yaml
export WANDB_MODE=offline          # compute nodes have no outbound network
export PYTHONUNBUFFERED=1

export PYTHON=python
export DEVICE=cuda
export ZARR="data_store_sc/super/dataset.zarr"
export EPOCHS=20
export PRECISION=fp32
export OUT="image_outputs/amse_loss"

source slurm/amse_loss_body.sh
