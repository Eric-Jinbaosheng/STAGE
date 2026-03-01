#!/bin/bash
#SBATCH --job-name=nlpfp_pull_data
#SBATCH --account=torch_pr_675_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
PYTHON_MODULE="${PYTHON_MODULE:-anaconda3/2025.06}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$WORKDIR/.conda_torch}"
DOWNLOAD_LIBERO="${DOWNLOAD_LIBERO:-1}"
DEXYCB_CACHE_DIR="${DEXYCB_CACHE_DIR:-}"

cd "$WORKDIR"
if command -v module >/dev/null 2>&1; then
  module purge || true
  if [[ -n "$PYTHON_MODULE" ]]; then
    module load "$PYTHON_MODULE"
  fi
fi
export PYTHONNOUSERSITE=1
PYTHON_BIN="${CONDA_ENV_DIR}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at $PYTHON_BIN"
  exit 1
fi

mkdir -p third_party data/libero_raw data/raw/handoversim .libero

if [[ "$DOWNLOAD_LIBERO" == "1" ]]; then
  if [[ ! -d third_party/LIBERO/.git ]]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
  fi

  pushd third_party/LIBERO >/dev/null
  "$PYTHON_BIN" benchmark_scripts/download_libero_datasets.py --datasets libero_spatial
  "$PYTHON_BIN" benchmark_scripts/download_libero_datasets.py --datasets libero_object
  "$PYTHON_BIN" benchmark_scripts/download_libero_datasets.py --datasets libero_goal
  popd >/dev/null

  if [[ -d third_party/LIBERO/datasets ]]; then
    mkdir -p data/libero_raw
    cp -r third_party/LIBERO/datasets/. data/libero_raw/
  elif [[ -d third_party/LIBERO/libero/datasets ]]; then
    mkdir -p data/libero_raw
    cp -r third_party/LIBERO/libero/datasets/. data/libero_raw/
  fi

  echo "LIBERO files under data/libero_raw:"
  find data/libero_raw -type f | head -20 || true
else
  echo "Skipping LIBERO download because DOWNLOAD_LIBERO=$DOWNLOAD_LIBERO"
fi

if [[ -n "$DEXYCB_CACHE_DIR" && -d "$DEXYCB_CACHE_DIR" ]]; then
  "$PYTHON_BIN" scripts/export_raw_from_dexycb_cache.py \
    --cache-dir "$DEXYCB_CACHE_DIR" \
    --out-dir data/raw/handoversim \
    --num-episodes 10 \
    --dt 0.01 \
    --max-frames 800
else
  echo "Skipping handover export. Set DEXYCB_CACHE_DIR to an existing dex-ycb-cache directory to enable it."
fi

echo "Data pull stage finished."
