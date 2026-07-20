#!/bin/bash
#SBATCH --job-name=lb_octo_act
#SBATCH --account=torch_pr_637_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-$WORKDIR/.conda_octo/bin/python}"
BENCHMARK="${BENCHMARK:-outputs/linguistic_blindness/benchmark_val_full_clean/counterfactual_examples.jsonl}"
QWEN_PREDICTIONS="${QWEN_PREDICTIONS:-outputs/linguistic_blindness/qwen25vl7b_schema_libero_target_swap_clean_600/predictions.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/linguistic_blindness/exp2_octo_small_target_swap_100}"
OCTO_CHECKPOINT="${OCTO_CHECKPOINT:-hf://rail-berkeley/octo-small-1.5}"
MAX_EXAMPLES="${MAX_EXAMPLES:-100}"
DATASET_FILTER="${DATASET_FILTER:-libero}"
PERTURBATION_TYPES="${PERTURBATION_TYPES:-target_swap}"
REQUIRE_COUNTERFACTUAL_VALID="${REQUIRE_COUNTERFACTUAL_VALID:-1}"
SHUFFLE="${SHUFFLE:-1}"
SAMPLE_SEED="${SAMPLE_SEED:-7}"
CONTROL_MODE="${CONTROL_MODE:-paraphrase}"
THRESHOLD_METHOD="${THRESHOLD_METHOD:-p95}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-25}"

cd "$WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/src:$WORKDIR/scripts:$WORKDIR/third_party/octo${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/$USER/hf-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/pip-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/$USER/xdg-cache}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$(dirname "$OUT_DIR")"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at $PYTHON_BIN" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" scripts/lb_run_octo_action_sensitivity.py
  --benchmark "$BENCHMARK"
  --qwen-predictions "$QWEN_PREDICTIONS"
  --out-dir "$OUT_DIR"
  --octo-checkpoint "$OCTO_CHECKPOINT"
  --max-examples "$MAX_EXAMPLES"
  --dataset-filter "$DATASET_FILTER"
  --perturbation-types "$PERTURBATION_TYPES"
  --control-mode "$CONTROL_MODE"
  --threshold-method "$THRESHOLD_METHOD"
  --log-every "$LOG_EVERY"
  --save-every "$SAVE_EVERY"
)
if [[ "$REQUIRE_COUNTERFACTUAL_VALID" == "1" ]]; then
  CMD+=(--require-counterfactual-valid)
fi
if [[ "$SHUFFLE" == "1" ]]; then
  CMD+=(--shuffle --sample-seed "$SAMPLE_SEED")
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
