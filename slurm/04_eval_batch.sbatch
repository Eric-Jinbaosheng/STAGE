#!/bin/bash
#SBATCH --job-name=nlpfp_eval
#SBATCH --account=torch_pr_675_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
PYTHON_MODULE="${PYTHON_MODULE:-anaconda3/2025.06}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$WORKDIR/.conda_torch}"
CHECKPOINT="${CHECKPOINT:-artifacts/finetune_baseline/checkpoint.pt}"
TEXT_VOCAB="${TEXT_VOCAB:-artifacts/finetune_baseline/text_vocab.json}"
INDEX_PATH="${INDEX_PATH:-}"
LABELS_PATH="${LABELS_PATH:-}"
DATASET_FILTER="${DATASET_FILTER:-libero}"
MAX_SAMPLES="${MAX_SAMPLES:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OUT_PATH="${OUT_PATH:-artifacts/batch_eval/result.json}"
SPLIT_PATH="${SPLIT_PATH:-}"
SPLIT_NAME="${SPLIT_NAME:-}"

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
  "$PYTHON_BIN" scripts/run_batch_eval.py
  --checkpoint "$CHECKPOINT"
  --text-vocab "$TEXT_VOCAB"
  --dataset "$DATASET_FILTER"
  --max-samples "$MAX_SAMPLES"
  --batch-size "$BATCH_SIZE"
  --out "$OUT_PATH"
)

if [[ -n "$INDEX_PATH" ]]; then
  CMD+=(--index "$INDEX_PATH")
fi
if [[ -n "$LABELS_PATH" ]]; then
  CMD+=(--labels "$LABELS_PATH")
fi
if [[ -n "$SPLIT_PATH" && -n "$SPLIT_NAME" ]]; then
  CMD+=(--split-path "$SPLIT_PATH" --split "$SPLIT_NAME")
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"

"$PYTHON_BIN" - <<PY
import json
d = json.load(open("$OUT_PATH", "r", encoding="utf-8"))
print("target_object_accuracy", d["metrics"]["target_object_accuracy"])
print("phase_accuracy", d["metrics"]["phase_accuracy"])
print("evaluated_samples", d["config"]["evaluated_samples"])
PY

echo "Batch evaluation finished."
