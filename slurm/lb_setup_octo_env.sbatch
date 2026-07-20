#!/bin/bash
#SBATCH --job-name=lb_setup_octo
#SBATCH --account=torch_pr_637_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
MICROMAMBA="${MICROMAMBA:-/scratch/bj2410/tools/micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/scratch/bj2410/micromamba-root}"
ENV_NAME="${ENV_NAME:-lb_octo}"
ENV_DIR="${ENV_DIR:-$WORKDIR/.conda_octo}"
OCTO_DIR="${OCTO_DIR:-$WORKDIR/third_party/octo}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

cd "$WORKDIR"
export MAMBA_ROOT_PREFIX
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/$USER/hf-cache}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/pip-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/$USER/xdg-cache}"
mkdir -p "$HF_HOME" "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" third_party

if [[ ! -x "$MICROMAMBA" ]]; then
  echo "micromamba not found at $MICROMAMBA" >&2
  exit 1
fi

if [[ ! -d "$ENV_DIR" ]]; then
  "$MICROMAMBA" create -y -p "$ENV_DIR" "python=$PYTHON_VERSION" pip
fi
PYTHON_BIN="$ENV_DIR/bin/python"
PIP_BIN="$ENV_DIR/bin/pip"

if [[ ! -d "$OCTO_DIR/.git" ]]; then
  rm -rf "$OCTO_DIR"
  git clone https://github.com/octo-models/octo.git "$OCTO_DIR"
fi

"$PIP_BIN" install --cache-dir "$PIP_CACHE_DIR" --upgrade pip setuptools wheel
# Octo official install. CPU JAX is enough for our small native action-sensitivity probe;
# it avoids CUDA/JAX version friction on shared clusters.
"$PIP_BIN" install --cache-dir "$PIP_CACHE_DIR" -e "$OCTO_DIR"
"$PIP_BIN" install --cache-dir "$PIP_CACHE_DIR" -r "$OCTO_DIR/requirements.txt"
"$PIP_BIN" install --cache-dir "$PIP_CACHE_DIR" "jax[cpu]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_releases.html
"$PIP_BIN" install --cache-dir "$PIP_CACHE_DIR" "scipy==1.11.4" "transformers==4.34.1" h5py pillow pandas pyarrow

"$PYTHON_BIN" - <<'PY'
import jax, octo
print('jax', jax.__version__)
print('octo import ok')
from octo.model.octo_model import OctoModel
print('OctoModel import ok')
PY
