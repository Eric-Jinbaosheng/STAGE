#!/bin/bash
#SBATCH --job-name=nlpfp_schema_eval
#SBATCH --account=torch_pr_675_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
PYTHON_MODULE="${PYTHON_MODULE:-anaconda3/2025.06}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$WORKDIR/.conda_torch}"
PYTHON_BIN="${CONDA_ENV_DIR}/bin/python"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-VL-2B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-artifacts/schema_vlm_lora/best}"
INDEX_PATH="${INDEX_PATH:-data/processed/libero_index.parquet}"
SCHEMA_TEXT_PATH="${SCHEMA_TEXT_PATH:-data/processed/schema_text.parquet}"
INSTR_WRONG_PATH="${INSTR_WRONG_PATH:-data/processed/instr_wrong.parquet}"
SPLIT_PATH="${SPLIT_PATH:-data/processed/episode_splits.json}"
SPLIT_NAME="${SPLIT_NAME:-val}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
MAX_LENGTH="${MAX_LENGTH:-384}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"
OUT_PATH="${OUT_PATH:-artifacts/schema_text_eval/result.json}"

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"

cd "$WORKDIR"
if command -v module >/dev/null 2>&1; then
  module purge || true
  if [[ -n "$PYTHON_MODULE" ]]; then
    module load "$PYTHON_MODULE"
  fi
fi
export PYTHONNOUSERSITE=1
export PYTHONPATH="${WORKDIR}/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at $PYTHON_BIN"
  exit 1
fi

CMD=(
  "$PYTHON_BIN" scripts/eval_schema_text.py
  --model-name "$MODEL_NAME"
  --adapter-path "$ADAPTER_PATH"
  --index "$INDEX_PATH"
  --schema-text "$SCHEMA_TEXT_PATH"
  --instr-wrong "$INSTR_WRONG_PATH"
  --split-path "$SPLIT_PATH"
  --split "$SPLIT_NAME"
  --max-samples "$MAX_SAMPLES"
  --image-size "$IMAGE_SIZE"
  --max-length "$MAX_LENGTH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --out "$OUT_PATH"
)

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  CMD+=(--trust-remote-code)
fi
if [[ "$BF16" == "1" ]]; then
  CMD+=(--bf16)
fi
if [[ "$FP16" == "1" ]]; then
  CMD+=(--fp16)
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"

echo "Schema text evaluation finished. Output in $OUT_PATH"
