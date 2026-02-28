#!/bin/bash
#SBATCH --job-name=nlpfp_setup
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.10}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1}"
USE_CUDA_MODULE="${USE_CUDA_MODULE:-0}"
VENV_DIR="${VENV_DIR:-$WORKDIR/.venv_torch}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

cd "$WORKDIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load "$PYTHON_MODULE"
  if [[ "$USE_CUDA_MODULE" == "1" ]]; then
    module load "$CUDA_MODULE"
  fi
fi

python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  numpy pandas pyarrow h5py \
  huggingface_hub requests tqdm \
  gymnasium robosuite
python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"

python - <<'PY'
import torch
print("python_ok")
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY

echo "Environment ready: $VENV_DIR"
