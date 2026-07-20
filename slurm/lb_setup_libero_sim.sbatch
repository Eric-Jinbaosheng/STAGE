#!/bin/bash
#SBATCH --job-name=lb_setup_libero
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
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$WORKDIR/.conda_vlm}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_DIR/bin/python}"
PIP_BIN="${PIP_BIN:-$CONDA_ENV_DIR/bin/pip}"

cd "$WORKDIR"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
mkdir -p "$HF_HOME" "$TMPDIR"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PIP_BIN" install -r third_party/openvla/experiments/robot/libero/libero_requirements.txt
"$PIP_BIN" install "git+https://github.com/Lifelong-Robot-Learning/LIBERO.git"

"$PYTHON_BIN" - <<'PY'
for m in ['libero','robosuite','bddl','gym','mujoco']:
    mod=__import__(m)
    print(m, 'OK', getattr(mod, '__version__', 'no_version'))
from libero.libero import benchmark
print('benchmarks', sorted(benchmark.get_benchmark_dict().keys()))
PY
