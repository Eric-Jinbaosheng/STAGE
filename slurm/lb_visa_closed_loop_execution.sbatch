#!/bin/bash
#SBATCH --job-name=lb-visa-closed
#SBATCH --account=torch_pr_675_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$WORKDIR/.conda_vlm}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_DIR/bin/python}"

MODEL_PATH="${MODEL_PATH:-/scratch/bj2410/models/openvla-7b-finetuned-libero-90}"
UNNORM_KEY="${UNNORM_KEY:-libero_90_no_noops}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TASK_IDS="${TASK_IDS:-1,3,4,5,6}"
EVALUATION_MODE="${EVALUATION_MODE:-counterfactual}"
METHODS="${METHODS:-native_openvla,generic_selection,visa_closed_loop,oracle_handoff}"
COUNTERFACTUAL_SPEC_FILE="${COUNTERFACTUAL_SPEC_FILE:-}"
MAX_TASKS="${MAX_TASKS:-10}"
NUM_TRIALS="${NUM_TRIALS:-1}"
MAX_STEPS="${MAX_STEPS:-220}"
OUT_DIR="${OUT_DIR:-outputs/linguistic_blindness/visa_closed_loop_execution_spatial_5tasks_1trial}"
CONTACT_DISTANCE="${CONTACT_DISTANCE:-0.06}"
LIFT_SUCCESS_DZ="${LIFT_SUCCESS_DZ:-0.03}"
PRIMITIVE_STEP="${PRIMITIVE_STEP:-0.02}"
RESIDUAL_ALPHAS="${RESIDUAL_ALPHAS:-0,0.25,0.5,0.75}"
SELECTIVE_RESIDUAL_ALPHAS="${SELECTIVE_RESIDUAL_ALPHAS:-0,0.1,0.2,0.3}"
RETRY_ON_FAIL="${RETRY_ON_FAIL:-1}"

cd "$WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/third_party/LIBERO_pkg:$WORKDIR/third_party/openvla:$WORKDIR/src:$WORKDIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$WORKDIR/.libero_config"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/$USER/hf-cache}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$WORKDIR/.hf_modules_cache}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$TMPDIR" "$(dirname "$OUT_DIR")"

"$PYTHON_BIN" scripts/lb_run_visa_closed_loop_execution.py \
  --task-suite-name "$TASK_SUITE" \
  --task-ids "$TASK_IDS" \
  --evaluation-mode "$EVALUATION_MODE" \
  --methods "$METHODS" \
  --counterfactual-spec-file "$COUNTERFACTUAL_SPEC_FILE" \
  --max-tasks "$MAX_TASKS" \
  --num-trials-per-task "$NUM_TRIALS" \
  --max-steps "$MAX_STEPS" \
  --model-path "$MODEL_PATH" \
  --unnorm-key "$UNNORM_KEY" \
  --out-dir "$OUT_DIR" \
  --contact-distance "$CONTACT_DISTANCE" \
  --lift-success-dz "$LIFT_SUCCESS_DZ" \
  --primitive-step "$PRIMITIVE_STEP" \
  --residual-alphas "$RESIDUAL_ALPHAS" \
  --selective-residual-alphas "$SELECTIVE_RESIDUAL_ALPHAS" \
  --retry-on-fail "$RETRY_ON_FAIL" \
  --device cuda:0 \
  --bf16 \
  --trust-remote-code \
  --center-crop
