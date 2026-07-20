#!/bin/bash
#SBATCH --job-name=lb_exp6_proj
#SBATCH --account=torch_pr_637_tandon_priority
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

TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TASK_IDS="${TASK_IDS:-}"
MAX_TASKS="${MAX_TASKS:-10}"
NUM_TRIALS="${NUM_TRIALS:-1}"
MAX_STEPS="${MAX_STEPS:-120}"
MODEL_PATH="${MODEL_PATH:-/scratch/bj2410/models/openvla-7b-finetuned-libero-90}"
UNNORM_KEY="${UNNORM_KEY:-libero_90_no_noops}"
OUT_DIR="${OUT_DIR:-outputs/linguistic_blindness/exp6_schema_action_projection_libero_spatial_smoke}"
ALPHA="${ALPHA:-0.7}"
ALIGNMENT_THRESHOLD="${ALIGNMENT_THRESHOLD:-0.3}"
PROJECT_STEP_SCALE="${PROJECT_STEP_SCALE:-1.0}"
PROJECTION_MODE="${PROJECTION_MODE:-alignment}"
RELEASE_DISTANCE="${RELEASE_DISTANCE:-0.05}"
REACQUIRE_DISTANCE="${REACQUIRE_DISTANCE:-0.12}"
PHASE_DESCEND_STEPS="${PHASE_DESCEND_STEPS:-12}"
PHASE_CLOSE_STEPS="${PHASE_CLOSE_STEPS:-18}"
PHASE_LIFT_STEPS="${PHASE_LIFT_STEPS:-45}"
PHASE_DESCEND_DZ="${PHASE_DESCEND_DZ:--0.01}"
PHASE_LIFT_DZ="${PHASE_LIFT_DZ:-0.02}"
LIFT_SUCCESS_DZ="${LIFT_SUCCESS_DZ:-0.03}"
CENTER_CROP="${CENTER_CROP:-1}"
BF16="${BF16:-1}"
SAVE_STEP_TRACES="${SAVE_STEP_TRACES:-1}"
ALWAYS_PROJECT="${ALWAYS_PROJECT:-0}"

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
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$TMPDIR" "$(dirname "$OUT_DIR")"

echo "==== Runtime diagnostics ===="
date
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
echo "============================="

CMD=(
  "$PYTHON_BIN" scripts/lb_run_exp6_schema_action_projection.py
  --task-suite-name "$TASK_SUITE"
  --max-tasks "$MAX_TASKS"
  --num-trials-per-task "$NUM_TRIALS"
  --max-steps "$MAX_STEPS"
  --model-path "$MODEL_PATH"
  --unnorm-key "$UNNORM_KEY"
  --out-dir "$OUT_DIR"
  --alpha "$ALPHA"
  --alignment-threshold "$ALIGNMENT_THRESHOLD"
  --project-step-scale "$PROJECT_STEP_SCALE"
  --projection-mode "$PROJECTION_MODE"
  --release-distance "$RELEASE_DISTANCE"
  --reacquire-distance "$REACQUIRE_DISTANCE"
  --phase-descend-steps "$PHASE_DESCEND_STEPS"
  --phase-close-steps "$PHASE_CLOSE_STEPS"
  --phase-lift-steps "$PHASE_LIFT_STEPS"
  --phase-descend-dz "$PHASE_DESCEND_DZ"
  --phase-lift-dz "$PHASE_LIFT_DZ"
  --lift-success-dz "$LIFT_SUCCESS_DZ"
  --trust-remote-code
)
if [[ -n "$TASK_IDS" ]]; then
  CMD+=(--task-ids "$TASK_IDS")
fi
if [[ "$CENTER_CROP" == "1" ]]; then
  CMD+=(--center-crop)
fi
if [[ "$BF16" == "1" ]]; then
  CMD+=(--bf16)
fi
if [[ "$SAVE_STEP_TRACES" == "1" ]]; then
  CMD+=(--save-step-traces)
fi
if [[ "$ALWAYS_PROJECT" == "1" ]]; then
  CMD+=(--always-project)
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
