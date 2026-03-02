#!/bin/bash
#SBATCH --job-name=nlpfp_schema_lora
#SBATCH --account=torch_pr_675_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
PYTHON_MODULE="${PYTHON_MODULE:-anaconda3/2025.06}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$WORKDIR/.conda_torch}"
PYTHON_BIN="${CONDA_ENV_DIR}/bin/python"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-VL-2B-Instruct}"
INDEX_PATH="${INDEX_PATH:-data/processed/libero_index.parquet}"
SCHEMA_TEXT_PATH="${SCHEMA_TEXT_PATH:-data/processed/schema_text.parquet}"
INSTR_WRONG_PATH="${INSTR_WRONG_PATH:-}"
SPLIT_PATH="${SPLIT_PATH:-data/processed/episode_splits.json}"
SPLIT_NAME="${SPLIT_NAME:-train}"
VAL_SPLIT_NAME="${VAL_SPLIT_NAME:-val}"
OUT_DIR="${OUT_DIR:-artifacts/schema_vlm_lora}"

IMAGE_SIZE="${IMAGE_SIZE:-224}"
MAX_LENGTH="${MAX_LENGTH:-384}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_STEPS="${MAX_STEPS:-2000}"
LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-200}"
SAVE_EVERY="${SAVE_EVERY:-500}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-16}"
SEED="${SEED:-7}"

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
DEVICE_MAP="${DEVICE_MAP:-}"
FLASH_ATTN_2="${FLASH_ATTN_2:-0}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-0}"

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
  "$PYTHON_BIN" scripts/train_schema_lora.py
  --model-name "$MODEL_NAME"
  --index "$INDEX_PATH"
  --schema-text "$SCHEMA_TEXT_PATH"
  --out-dir "$OUT_DIR"
  --image-size "$IMAGE_SIZE"
  --max-length "$MAX_LENGTH"
  --batch-size "$BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --warmup-ratio "$WARMUP_RATIO"
  --max-steps "$MAX_STEPS"
  --log-every "$LOG_EVERY"
  --eval-every "$EVAL_EVERY"
  --save-every "$SAVE_EVERY"
  --max-val-batches "$MAX_VAL_BATCHES"
  --seed "$SEED"
  --lora-r "$LORA_R"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --lora-target-modules "$LORA_TARGET_MODULES"
)

if [[ -n "$INSTR_WRONG_PATH" ]]; then
  CMD+=(--instr-wrong "$INSTR_WRONG_PATH")
fi
if [[ -n "$SPLIT_PATH" ]]; then
  CMD+=(--split-path "$SPLIT_PATH" --split "$SPLIT_NAME" --val-split "$VAL_SPLIT_NAME")
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  CMD+=(--trust-remote-code)
fi
if [[ "$BF16" == "1" ]]; then
  CMD+=(--bf16)
fi
if [[ "$FP16" == "1" ]]; then
  CMD+=(--fp16)
fi
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  CMD+=(--gradient-checkpointing)
fi
if [[ -n "$DEVICE_MAP" ]]; then
  CMD+=(--device-map "$DEVICE_MAP")
fi
if [[ "$FLASH_ATTN_2" == "1" ]]; then
  CMD+=(--flash-attn-2)
fi
if [[ "$LOAD_IN_4BIT" == "1" ]]; then
  CMD+=(--load-in-4bit)
fi
if [[ "$LOAD_IN_8BIT" == "1" ]]; then
  CMD+=(--load-in-8bit)
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"

echo "Schema LoRA training finished. Outputs in $OUT_DIR"
