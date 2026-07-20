#!/bin/bash
#SBATCH --job-name=lb_vlm_env
#SBATCH --account=torch_pr_675_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
PYTHON_MODULE="${PYTHON_MODULE:-anaconda3/2025.06}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$WORKDIR/.conda_vlm}"
CONDA_SCRATCH_ROOT="${CONDA_SCRATCH_ROOT:-/scratch/$USER/.conda}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$CONDA_SCRATCH_ROOT/pkgs}"
CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$CONDA_SCRATCH_ROOT/envs}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/pip-cache}"
TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

cd "$WORKDIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  if [[ -n "$PYTHON_MODULE" ]]; then
    module load "$PYTHON_MODULE"
  fi
fi

export PYTHONNOUSERSITE=1
mkdir -p "$CONDA_SCRATCH_ROOT" "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" "$PIP_CACHE_DIR" "$TMPDIR"
export CONDA_PKGS_DIRS CONDA_ENVS_PATH PIP_CACHE_DIR TMPDIR

source "$(conda info --base)/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

conda deactivate 2>/dev/null || true
if [[ "${RECREATE_ENV:-1}" == "1" ]]; then
  rm -rf "$CONDA_ENV_DIR"
fi
if [[ ! -x "$CONDA_ENV_DIR/bin/python" ]]; then
  conda create -y -p "$CONDA_ENV_DIR" python=3.11 pip
fi
conda activate "$CONDA_ENV_DIR"

python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy pandas pyarrow pillow tqdm requests huggingface_hub safetensors accelerate qwen-vl-utils
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url "$TORCH_INDEX_URL"
python -m pip install transformers==4.49.0 peft==0.14.0 bitsandbytes==0.45.0

python - <<'PY'
import importlib
import sys
print('python', sys.version)
for name in ['torch', 'transformers', 'accelerate', 'PIL', 'qwen_vl_utils', 'safetensors', 'bitsandbytes', 'pandas', 'pyarrow']:
    mod = importlib.import_module(name)
    print(name, getattr(mod, '__version__', 'ok'))
import torch
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda_device', torch.cuda.get_device_name(0))
from transformers import AutoConfig, AutoProcessor
model_path = '/scratch/bj2410/models/Qwen2.5-VL-7B-Instruct'
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print('model_type', getattr(config, 'model_type', None))
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print('processor', type(processor).__name__)
PY

echo "VLM environment ready: $CONDA_ENV_DIR"
