#!/bin/bash
#SBATCH -J multiseed
#SBATCH -o /work/11755/gurup12/vista/downscaling/logs/ms_%j.out
#SBATCH -e /work/11755/gurup12/vista/downscaling/logs/ms_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -A ATM23014
#
# Multi-seed repeats — the experiment that decides whether the nine-way tie in
# RESULTS.md §3 is real or an artifact of single runs.
#
# Six architectures spanning 0.13M -> 41.5M parameters, five seeds each. That
# range is chosen deliberately: it includes the nominal winner (deepsd), the
# largest model (swin), and the only apparent laggard (convnext). If the
# between-seed spread turns out to exceed the between-model spread, the tie is
# confirmed and architecture selection is settled for this problem.

set -euo pipefail
ROOT=/work/11755/gurup12/vista/downscaling
cd "$ROOT"; mkdir -p logs

echo "=== node $(hostname) $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader || true

source /work/11755/gurup12/vista/miniforge3/etc/profile.d/conda.sh
conda activate dsc
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export DOWNSCALE_CONFIG_data=configs/data_southcentral.yaml
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
sed -i 's/^device: .*/device: "cuda"/' configs/train.yaml
sed -i 's/^precision: .*/precision: "fp32"/' configs/train.yaml   # bf16 halves SSIM

ARCHS="deepsd esrt edsr restormer swin convnext"
SEEDS="1337 7 42 2024 31337"

for s in $SEEDS; do
  for a in $ARCHS; do
    name="${a}_s${s}"
    if [ -f "checkpoints/${name}/best.pt" ]; then
      echo "=== $name exists, skipping"; continue
    fi
    echo "=============== $name  $(date)"
    python -u -m training.train --arch "$a" --epochs 20 \
        --seed "$s" --run-name "$name" || echo "[ms] $name FAILED, continuing"
  done
done

# Every seed replicate scored on the Austin spatial holdout, same protocol.
ALL=""
for s in $SEEDS; do for a in $ARCHS; do ALL="$ALL ${a}_s${s}"; done; done
echo "=============== holdout eval  $(date)"
python -u -m evaluation.holdout_eval --archs $ALL --n-boot 200 --device cuda \
    --out image_outputs/multiseed || true

echo "=== done $(date)"
