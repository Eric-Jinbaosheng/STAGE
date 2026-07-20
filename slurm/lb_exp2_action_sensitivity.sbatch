#!/bin/bash
#SBATCH --job-name=lb_exp2_action
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
MODEL_PATH="${MODEL_PATH:-/scratch/bj2410/models/openvla-7b}"
BENCHMARK="${BENCHMARK:-outputs/linguistic_blindness/benchmark_val_full_clean/counterfactual_examples.jsonl}"
QWEN_PREDICTIONS="${QWEN_PREDICTIONS:-outputs/linguistic_blindness/qwen25vl7b_schema_libero_target_swap_clean_600/predictions.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600}"
MAX_EXAMPLES="${MAX_EXAMPLES:-20}"
PERTURBATION_TYPES="${PERTURBATION_TYPES:-target_swap}"
DATASET_FILTER="${DATASET_FILTER:-libero}"
REQUIRE_COUNTERFACTUAL_VALID="${REQUIRE_COUNTERFACTUAL_VALID:-1}"
SHUFFLE="${SHUFFLE:-1}"
SAMPLE_SEED="${SAMPLE_SEED:-7}"
UNNORM_KEY="${UNNORM_KEY:-bridge_orig}"
CONTROL_MODE="${CONTROL_MODE:-paraphrase}"
THRESHOLD_METHOD="${THRESHOLD_METHOD:-p95}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
ALLOW_PLACEHOLDER_IMAGE="${ALLOW_PLACEHOLDER_IMAGE:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-50}"

cd "$WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/src:$WORKDIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/$USER/hf-cache}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$WORKDIR/.hf_modules_cache}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$TMPDIR" "$(dirname "$OUT_DIR")"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$BENCHMARK" ]]; then
  echo "Benchmark not found at $BENCHMARK" >&2
  exit 1
fi
if [[ ! -f "$QWEN_PREDICTIONS" ]]; then
  echo "Qwen predictions not found at $QWEN_PREDICTIONS" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" scripts/lb_run_exp2_action_sensitivity.py
  --benchmark "$BENCHMARK"
  --qwen-predictions "$QWEN_PREDICTIONS"
  --model-path "$MODEL_PATH"
  --out-dir "$OUT_DIR"
  --max-examples "$MAX_EXAMPLES"
  --perturbation-types "$PERTURBATION_TYPES"
  --dataset-filter "$DATASET_FILTER"
  --unnorm-key "$UNNORM_KEY"
  --control-mode "$CONTROL_MODE"
  --threshold-method "$THRESHOLD_METHOD"
  --trust-remote-code
  --log-every "$LOG_EVERY"
  --save-every "$SAVE_EVERY"
)
if [[ "$REQUIRE_COUNTERFACTUAL_VALID" == "1" ]]; then
  CMD+=(--require-counterfactual-valid)
fi
if [[ "$SHUFFLE" == "1" ]]; then
  CMD+=(--shuffle --sample-seed "$SAMPLE_SEED")
fi
if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
  CMD+=(--attn-implementation "$ATTN_IMPLEMENTATION")
fi
if [[ "$BF16" == "1" ]]; then
  CMD+=(--bf16)
fi
if [[ "$FP16" == "1" ]]; then
  CMD+=(--fp16)
fi
if [[ "$ALLOW_PLACEHOLDER_IMAGE" == "1" ]]; then
  CMD+=(--allow-placeholder-image)
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
