#!/bin/bash
#SBATCH --job-name=lb_exp3_openvla
#SBATCH --account=torch_pr_675_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$WORKDIR/.conda_vlm}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_DIR/bin/python}"
MODEL_PATH="${MODEL_PATH:-/scratch/bj2410/models/openvla-7b}"
BENCHMARK="${BENCHMARK:-outputs/linguistic_blindness/exp3_invalid_instruction_clean_600/benchmark.jsonl}"
QWEN_PREDICTIONS="${QWEN_PREDICTIONS:-outputs/linguistic_blindness/qwen25vl7b_schema_exp3_invalid_clean_600/predictions.jsonl}"
EXP2_PREDICTIONS="${EXP2_PREDICTIONS:-outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/predictions.jsonl}"
EXP2_METRICS="${EXP2_METRICS:-outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/metrics.json}"
OUT_DIR="${OUT_DIR:-outputs/linguistic_blindness/exp3_invalid_instruction_clean_600/openvla_action_inhibition}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
UNNORM_KEY="${UNNORM_KEY:-bridge_orig}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
ALLOW_PLACEHOLDER_IMAGE="${ALLOW_PLACEHOLDER_IMAGE:-0}"
LOG_EVERY="${LOG_EVERY:-25}"
SAVE_EVERY="${SAVE_EVERY:-100}"

cd "$WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/src:$WORKDIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/$USER/hf-cache}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$WORKDIR/.hf_modules_cache}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$TMPDIR" "$(dirname "$OUT_DIR")"

for f in "$BENCHMARK" "$QWEN_PREDICTIONS" "$EXP2_PREDICTIONS" "$EXP2_METRICS"; do
  if [[ ! -f "$f" ]]; then
    echo "Required file not found: $f" >&2
    exit 1
  fi
done
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path not found at $MODEL_PATH" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" scripts/lb_run_exp3_invalid_action.py
  --benchmark "$BENCHMARK"
  --qwen-predictions "$QWEN_PREDICTIONS"
  --exp2-predictions "$EXP2_PREDICTIONS"
  --exp2-metrics "$EXP2_METRICS"
  --model-path "$MODEL_PATH"
  --out-dir "$OUT_DIR"
  --max-examples "$MAX_EXAMPLES"
  --unnorm-key "$UNNORM_KEY"
  --trust-remote-code
  --log-every "$LOG_EVERY"
  --save-every "$SAVE_EVERY"
)
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
