#!/bin/bash
#SBATCH --job-name=lb-visa-rerank-k50
#SBATCH --account=torch_pr_675_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$WORKDIR/.conda_vlm}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_DIR/bin/python}"

MODEL_PATH="${MODEL_PATH:-/scratch/bj2410/models/openvla-7b-finetuned-libero-90}"
UNNORM_KEY="${UNNORM_KEY:-libero_90_no_noops}"
OUT_DIR="${OUT_DIR:-outputs/linguistic_blindness/short_horizon_visa_rerank_k50}"
TASK_IDS="${TASK_IDS:-1,3,4,5,6}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-2}"
HORIZON="${HORIZON:-50}"

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

"$PYTHON_BIN" scripts/lb_run_short_horizon_visa_rerank.py \
  --task-suite-name libero_spatial \
  --task-ids "$TASK_IDS" \
  --num-trials-per-task "$NUM_TRIALS_PER_TASK" \
  --horizon "$HORIZON" \
  --num-steps-wait 10 \
  --model-path "$MODEL_PATH" \
  --unnorm-key "$UNNORM_KEY" \
  --out-dir "$OUT_DIR" \
  --device cuda:0 \
  --bf16 \
  --trust-remote-code
