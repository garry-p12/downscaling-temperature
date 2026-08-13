#!/bin/bash
#SBATCH -J tierb
#SBATCH -o /work/11755/gurup12/vista/downscaling/logs/tierb_%j.out
#SBATCH -e /work/11755/gurup12/vista/downscaling/logs/tierb_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -A ATM23014
#
# Tier B on Vista (NVIDIA GH200, aarch64): the four heavyweight architectures
# that cost ~10 h each on the laptop, plus the Austin spatial-holdout
# evaluation so every model lands in one comparable table.
#
# Everything — including the conda env build — runs INSIDE the job. TACC kills
# long-running processes on login nodes, which already killed one interactive
# install here.

set -euo pipefail
ROOT=/work/11755/gurup12/vista/downscaling
cd "$ROOT"
mkdir -p logs

echo "=== node: $(hostname)  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------- env ----- #
source /work/11755/gurup12/vista/miniforge3/etc/profile.d/conda.sh
# -c conda-forge explicitly: this conda ships with NO default channels
# configured, so a bare `conda create python=3.11` fails with
# "PackagesNotFoundError ... you may need to add a channel".
if ! conda env list | grep -q "^dsc "; then
  echo "=== creating env (first run only)"
  conda create -y -n dsc -c conda-forge --override-channels python=3.11
fi
conda activate dsc

python - <<'PY' || NEED_INSTALL=1
import importlib, sys
missing = [m for m in ("torch", "xarray", "zarr", "scipy", "pandas", "matplotlib")
           if importlib.util.find_spec(m) is None]
sys.exit(1 if missing else 0)
PY
if [ "${NEED_INSTALL:-0}" = "1" ]; then
  echo "=== installing deps (aarch64 + CUDA wheels)"
  # PyTorch publishes linux-aarch64 CUDA wheels; Vista is Grace Hopper, so the
  # default x86 index would silently give a CPU-only build.
  pip install --quiet torch --index-url https://download.pytorch.org/whl/cu126
  pip install --quiet numpy scipy pandas xarray zarr netcdf4 matplotlib pyproj einops
fi

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

# ------------------------------------------------------------- training --- #
export DOWNSCALE_CONFIG_data=configs/data_southcentral.yaml
export WANDB_MODE=offline          # compute nodes have no outbound network
export PYTHONUNBUFFERED=1

# cuda, not mps: resolve_device() falls back to mps/cpu only when cuda is absent
sed -i 's/^device: .*/device: "cuda"/' configs/train.yaml
# fp32, NOT bf16. autocast is gated on device.type == "cuda", so the laptop
# (MPS) silently trains in fp32 while the GPU would train in bf16 — Tier A and
# Tier B would then differ in numerical precision, not just architecture.
# bf16 also wrecks the SSIM loss term specifically: its 8-bit mantissa cannot
# hold the local variance/covariance products SSIM is built from. Observed
# directly here — swin reached SSIM 0.44 under bf16 vs 0.88 for Tier A in fp32.
sed -i 's/^precision: .*/precision: "fp32"/' configs/train.yaml

ARCHS="swin restormer maxvit convnext"
for a in $ARCHS; do
  echo "=============== $a  $(date)"
  python -u -m training.train --arch "$a" --epochs 20 || echo "[tierb] $a FAILED, continuing"
done

# ------------------------------------------------- Austin holdout eval ---- #
# The point of the whole run: score every architecture on terrain never trained
# on. Tier A checkpoints are rsynced in alongside, so this table covers all 10.
echo "=============== Austin holdout eval  $(date)"
python -u -m evaluation.holdout_eval \
    --archs deepsd edsr vit esrt segformer swinir_light swin restormer maxvit convnext \
    --n-boot 500 --device cuda --out image_outputs/austin_holdout || true

echo "=== done $(date)"
