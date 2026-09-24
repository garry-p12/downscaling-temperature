#!/bin/bash
#SBATCH -J ofs_smoke
#SBATCH -o logs/ofs_smoke_%j.out
#SBATCH -e logs/ofs_smoke_%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -A ATM23014
#
# Error-budget split of the loss. See slurm/offset_split_body.sh for the
# hypothesis, the arms (including the full-field control that keeps the result
# interpretable), and why raw RMSE is NOT a success criterion here.
#
# gh-dev rather than gh: a 2 h cap and one running job per user, but gh is
# routinely hundreds of jobs deep. Ten DeepSD runs (two arms x five seeds) at
# a few minutes each should fit, and the body skips any seed whose last.pt
# exists, so a wall-time kill loses nothing and resubmitting continues where
# it stopped.
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
export EPOCHS=1
export PRECISION=fp32
export OUT="image_outputs/offset_split"

export ARMS="of1"
export SEEDS_OVERRIDE="1337"
export DO_EVAL=0
source slurm/offset_split_body.sh
