#!/bin/bash
#SBATCH --job-name=nlpfp_setup
#SBATCH --account=torch_pr_675_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
PYTHON_MODULE="${PYTHON_MODULE:-anaconda3/2025.06}"
USE_CUDA_MODULE="${USE_CUDA_MODULE:-0}"
CUDA_MODULE="${CUDA_MODULE:-}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$WORKDIR/.conda_torch}"
CONDA_HOME_LINK="${CONDA_HOME_LINK:-$HOME/.conda}"
CONDA_SCRATCH_ROOT="${CONDA_SCRATCH_ROOT:-/scratch/$USER/.conda}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$CONDA_SCRATCH_ROOT/pkgs}"
CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$CONDA_SCRATCH_ROOT/envs}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/pip-cache}"
TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
INSTALL_EXTRA_SIM_DEPS="${INSTALL_EXTRA_SIM_DEPS:-1}"

cd "$WORKDIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  if [[ -n "$PYTHON_MODULE" ]]; then
    module load "$PYTHON_MODULE"
  fi
  if [[ "$USE_CUDA_MODULE" == "1" && -n "$CUDA_MODULE" ]]; then
    module load "$CUDA_MODULE"
  fi
fi

export PYTHONNOUSERSITE=1
mkdir -p "$CONDA_SCRATCH_ROOT" "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" "$PIP_CACHE_DIR" "$TMPDIR"

if [[ -e "$CONDA_HOME_LINK" && ! -L "$CONDA_HOME_LINK" ]]; then
  mv "$CONDA_HOME_LINK" "${CONDA_HOME_LINK}_backup_$(date +%s)"
fi
ln -sfn "$CONDA_SCRATCH_ROOT" "$CONDA_HOME_LINK"

source "$(conda info --base)/etc/profile.d/conda.sh"

export CONDA_PKGS_DIRS
export CONDA_ENVS_PATH
export PIP_CACHE_DIR
export TMPDIR

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

conda deactivate 2>/dev/null || true
rm -rf "$CONDA_ENV_DIR"
conda create -y -p "$CONDA_ENV_DIR" python=3.11
conda activate "$CONDA_ENV_DIR"

python -m pip install numpy pandas pyarrow h5py requests tqdm huggingface_hub
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url "$TORCH_INDEX_URL"

if [[ "$INSTALL_EXTRA_SIM_DEPS" == "1" ]]; then
  python -m pip install gymnasium
  python -m pip install termcolor psutil tensorboard tensorboardX imageio imageio-ffmpeg matplotlib
  python -m pip install robomimic --no-deps
fi

python - <<'PY'
import sys
import torch
print("python_version", sys.version)
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY

echo "Environment ready: $CONDA_ENV_DIR"
