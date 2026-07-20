#!/bin/bash
#SBATCH --job-name=lb_exp4_qwen_norm
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
MODEL_PATH="${MODEL_PATH:-/scratch/bj2410/models/Qwen2.5-VL-7B-Instruct}"
BENCHMARK="${BENCHMARK:-outputs/linguistic_blindness/exp4_normal_instruction_clean_600/benchmark.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/linguistic_blindness/qwen25vl7b_schema_exp4_normal_clean_600}"
METHOD="${METHOD:-qwen25vl7b_exp4_normal_schema}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
LOG_EVERY="${LOG_EVERY:-25}"
SAVE_EVERY="${SAVE_EVERY:-100}"
PROMPT_PROFILE="${PROMPT_PROFILE:-default}"

cd "$WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/$USER/hf-cache}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TMPDIR" "$(dirname "$OUT_DIR")"

for f in "$BENCHMARK"; do
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
  "$PYTHON_BIN" scripts/lb_run_qwen_schema.py
  --benchmark "$BENCHMARK"
  --model-path "$MODEL_PATH"
  --out-dir "$OUT_DIR"
  --method "$METHOD"
  --max-examples "$MAX_EXAMPLES"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --max-length "$MAX_LENGTH"
  --device-map "$DEVICE_MAP"
  --trust-remote-code
  --prompt-profile "$PROMPT_PROFILE"
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

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
