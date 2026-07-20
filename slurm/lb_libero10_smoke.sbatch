#!/bin/bash
#SBATCH --job-name=lb_libero10_smoke
#SBATCH --account=torch_pr_675_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$WORKDIR/.conda_vlm}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_DIR/bin/python}"
MODEL_PATH="${MODEL_PATH:-/scratch/bj2410/models/openvla-7b}"
OUT_DIR="${OUT_DIR:-outputs/linguistic_blindness/exp6_libero10_smoke_dummy}"
NO_MODEL="${NO_MODEL:-1}"
MAX_TASKS="${MAX_TASKS:-1}"
NUM_TRIALS="${NUM_TRIALS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
UNNORM_KEY="${UNNORM_KEY:-bridge_orig}"
BF16="${BF16:-1}"
CENTER_CROP="${CENTER_CROP:-0}"

cd "$WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/third_party/LIBERO_pkg:$WORKDIR/third_party/openvla:$WORKDIR/src:$WORKDIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$WORKDIR/.libero_config"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/$USER/hf-cache}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$WORKDIR/.hf_modules_cache}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$TMPDIR" "$(dirname "$OUT_DIR")"

CMD=(
  "$PYTHON_BIN" scripts/lb_run_libero10_smoke.py
  --task-suite-name libero_10
  --max-tasks "$MAX_TASKS"
  --num-trials-per-task "$NUM_TRIALS"
  --model-path "$MODEL_PATH"
  --unnorm-key "$UNNORM_KEY"
  --out-dir "$OUT_DIR"
  --trust-remote-code
)
if [[ -n "$MAX_STEPS" ]]; then
  CMD+=(--max-steps "$MAX_STEPS")
fi
if [[ "$NO_MODEL" == "1" ]]; then
  CMD+=(--no-model)
fi
if [[ "$BF16" == "1" ]]; then
  CMD+=(--bf16)
fi
if [[ "$CENTER_CROP" == "1" ]]; then
  CMD+=(--center-crop)
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
