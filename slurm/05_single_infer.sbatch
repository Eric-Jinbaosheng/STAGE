#!/bin/bash
#SBATCH --job-name=nlpfp_infer1
#SBATCH --account=torch_pr_675_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
VENV_DIR="${VENV_DIR:-$WORKDIR/.venv_torch}"
CHECKPOINT="${CHECKPOINT:-artifacts/finetune_baseline/checkpoint.pt}"
TEXT_VOCAB="${TEXT_VOCAB:-artifacts/finetune_baseline/text_vocab.json}"
EPISODE_ID="${EPISODE_ID:-open_the_middle_drawer_of_the_cabinet_demo:demo_0}"
FRAME_ID="${FRAME_ID:-0}"
OUT_PATH="${OUT_PATH:-artifacts/single_infer/result.json}"
INDEX_PATH="${INDEX_PATH:-}"
LABELS_PATH="${LABELS_PATH:-}"

cd "$WORKDIR"
source "$VENV_DIR/bin/activate"

CMD=(
  python scripts/run_single_inference.py
  --checkpoint "$CHECKPOINT"
  --text-vocab "$TEXT_VOCAB"
  --episode-id "$EPISODE_ID"
  --frame-id "$FRAME_ID"
  --out "$OUT_PATH"
)

if [[ -n "$INDEX_PATH" ]]; then
  CMD+=(--index "$INDEX_PATH")
fi
if [[ -n "$LABELS_PATH" ]]; then
  CMD+=(--labels "$LABELS_PATH")
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"

cat "$OUT_PATH"
