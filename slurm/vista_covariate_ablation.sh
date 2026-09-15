#!/bin/bash
#SBATCH -J covabl
#SBATCH -o logs/covabl_%j.out
#SBATCH -e logs/covabl_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A ATM23014
#
# Covariate ablation: does engineering the static inputs beat the baseline?
#
# Five variants x five seeds on DeepSD. DeepSD because it is 0.13 M parameters,
# trains in minutes, and is statistically tied with every larger architecture on
# this task — there is no reason to spend a 41 M model on a DATA question.
#
# FIVE SEEDS IS NOT OPTIONAL. The architecture comparison established that the
# between-model spread is 1.22x the between-seed noise, and that the apparent
# single-seed winner was seed luck. A one-run-per-variant ablation here would
# measure nothing. The decision rule is fixed in advance: a variant beats the
# baseline only if its 5-seed mean RMSE improves by more than the between-seed
# sd (~0.01 degC for DeepSD).
#
# Every variant reads the SAME superset store and selects channels from it, so
# no difference can be an artefact of the dataset build.
#
# Submit from the repo root:  sbatch slurm/vista_covariate_ablation.sh

set -euo pipefail

ROOT="${WORK}/downscaling"          # $WORK already ends in /vista
cd "$ROOT"
mkdir -p logs

echo "=== node $(hostname)  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source "${WORK}/miniforge3/etc/profile.d/conda.sh"
conda activate dsc
python -c "import torch; print('torch', torch.__version__, '| cuda',
           torch.cuda.is_available())" 2>/dev/null || true

# Config is selected by environment, never by editing (and never by sed-ing the
# tracked YAML in place, which is what the older Tier-B script did).
export DOWNSCALE_CONFIG_data=configs/data_sc_super.yaml
export WANDB_MODE=offline          # compute nodes have no outbound network
export PYTHONUNBUFFERED=1

# The experiment itself lives in one place so the cluster and laptop runs
# cannot drift apart. This wrapper only supplies the environment.
export PYTHON=python
export DEVICE=cuda
export ZARR="data_store_sc/super/dataset.zarr"
export EPOCHS=20
export PRECISION=fp32
export OUT="image_outputs/covariate_ablation"

source slurm/covariate_ablation_body.sh
