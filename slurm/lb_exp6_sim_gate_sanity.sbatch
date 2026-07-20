#!/bin/bash
#SBATCH --job-name=lb_exp6_sim_gate
#SBATCH --account=torch_pr_675_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$WORKDIR/.conda_vlm}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_DIR/bin/python}"
TASK_SUITE="${TASK_SUITE:-libero_90}"
MODEL_PATH="${MODEL_PATH:-/scratch/bj2410/models/openvla-7b-finetuned-libero-90}"
UNNORM_KEY="${UNNORM_KEY:-libero_90_no_noops}"
OUT_DIR="${OUT_DIR:-outputs/linguistic_blindness/exp6_libero90_schema_gate_sanity_10x1}"
MAX_TASKS="${MAX_TASKS:-10}"
NUM_TRIALS="${NUM_TRIALS:-1}"
NORMAL_MAX_STEPS="${NORMAL_MAX_STEPS:-}"
INVALID_MAX_STEPS="${INVALID_MAX_STEPS:-80}"
CENTER_CROP="${CENTER_CROP:-1}"
BF16="${BF16:-1}"
PERTURBATIONS="${PERTURBATIONS:-blank_instruction,impossible_instruction,negation}"

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
  "$PYTHON_BIN" scripts/lb_run_exp6_sim_gate_sanity.py
  --task-suite-name "$TASK_SUITE"
  --max-tasks "$MAX_TASKS"
  --num-trials-per-task "$NUM_TRIALS"
  --invalid-max-steps "$INVALID_MAX_STEPS"
  --model-path "$MODEL_PATH"
  --unnorm-key "$UNNORM_KEY"
  --out-dir "$OUT_DIR"
  --perturbations "$PERTURBATIONS"
  --trust-remote-code
)
if [[ -n "$NORMAL_MAX_STEPS" ]]; then
  CMD+=(--normal-max-steps "$NORMAL_MAX_STEPS")
fi
if [[ "$CENTER_CROP" == "1" ]]; then
  CMD+=(--center-crop)
fi
if [[ "$BF16" == "1" ]]; then
  CMD+=(--bf16)
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
