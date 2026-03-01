#!/bin/bash
#SBATCH --job-name=nlpfp_train
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
OUT_DIR="${OUT_DIR:-artifacts/finetune_baseline}"
INDEX_PATH="${INDEX_PATH:-data/processed/libero_index.parquet}"
LABELS_PATH="${LABELS_PATH:-data/processed/schema_labels.parquet}"
USE_IMAGE="${USE_IMAGE:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
FRAME_WINDOW="${FRAME_WINDOW:-3}"
VIEWS="${VIEWS:-agentview_rgb,eye_in_hand_rgb}"
TEXT_VOCAB_SIZE="${TEXT_VOCAB_SIZE:-2048}"
TEXT_MAX_LEN="${TEXT_MAX_LEN:-16}"
TEXT_VOCAB_PATH="${TEXT_VOCAB_PATH:-}"
TEXT_EMBED_DIM="${TEXT_EMBED_DIM:-64}"
TEXT_HIDDEN_DIM="${TEXT_HIDDEN_DIM:-64}"
SPLIT_PATH="${SPLIT_PATH:-}"
SPLIT_NAME="${SPLIT_NAME:-}"
STEPS="${STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
TASK="${TASK:-multi}"

cd "$WORKDIR"
if command -v module >/dev/null 2>&1; then
  module purge || true
  if [[ -n "$PYTHON_MODULE" ]]; then
    module load "$PYTHON_MODULE"
  fi
fi
export PYTHONNOUSERSITE=1
PYTHON_BIN="${CONDA_ENV_DIR}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at $PYTHON_BIN"
  exit 1
fi

CMD=(
  "$PYTHON_BIN" scripts/train_finetune_baseline.py
  --index "$INDEX_PATH"
  --labels "$LABELS_PATH"
  --out-dir "$OUT_DIR"
  --steps "$STEPS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --image-size "$IMAGE_SIZE"
  --frame-window "$FRAME_WINDOW"
  --views "$VIEWS"
  --text-vocab-size "$TEXT_VOCAB_SIZE"
  --text-max-len "$TEXT_MAX_LEN"
  --text-embed-dim "$TEXT_EMBED_DIM"
  --text-hidden-dim "$TEXT_HIDDEN_DIM"
  --task "$TASK"
)

if [[ "$USE_IMAGE" == "1" ]]; then
  CMD+=(--use-image)
fi

if [[ -n "$TEXT_VOCAB_PATH" ]]; then
  CMD+=(--text-vocab-path "$TEXT_VOCAB_PATH")
fi
if [[ -n "$SPLIT_PATH" && -n "$SPLIT_NAME" ]]; then
  CMD+=(--split-path "$SPLIT_PATH" --split "$SPLIT_NAME")
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"

echo "Training finished. Outputs in $OUT_DIR"
