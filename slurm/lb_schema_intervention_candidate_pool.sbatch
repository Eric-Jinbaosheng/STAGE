#!/bin/bash
#SBATCH --job-name=lb-schema-pool
#SBATCH --account=torch_pr_675_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$WORKDIR/.conda_vlm}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_DIR/bin/python}"

MODEL_PATH="${MODEL_PATH:-/scratch/bj2410/models/openvla-7b}"
UNNORM_KEY="${UNNORM_KEY:-bridge_orig}"
MAX_EXAMPLES="${MAX_EXAMPLES:-80}"
INPUT_JSONL="${INPUT_JSONL:-outputs/linguistic_blindness/satbenchpp_v1/analysis/satbenchpp_predictions_augmented.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/linguistic_blindness/schema_intervention_candidate_pool_satbenchpp_n80}"
BUDGETS="${BUDGETS:-1,2,4,8,16}"
CONDITIONS="${CONDITIONS:-generic,correct_schema,target_swapped_schema,shuffled_schema,field_dropped_schema}"
ALIGNMENT_MARGIN="${ALIGNMENT_MARGIN:-0.05}"
SAMPLE_SEED="${SAMPLE_SEED:-13}"
SPLIT_FILTER="${SPLIT_FILTER:-}"
FAMILY_FILTER="${FAMILY_FILTER:-}"

cd "$WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/src:$WORKDIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/$USER/hf-cache}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$WORKDIR/.hf_modules_cache}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$TMPDIR" "$(dirname "$OUT_DIR")"

"$PYTHON_BIN" scripts/lb_run_schema_intervention_candidate_pool.py \
  --input-jsonl "$INPUT_JSONL" \
  --index data/processed/unified_index.parquet \
  --model-path "$MODEL_PATH" \
  --unnorm-key "$UNNORM_KEY" \
  --out-dir "$OUT_DIR" \
  --max-examples "$MAX_EXAMPLES" \
  --shuffle \
  --sample-seed "$SAMPLE_SEED" \
  --split-filter "$SPLIT_FILTER" \
  --family-filter "$FAMILY_FILTER" \
  --budgets "$BUDGETS" \
  --conditions "$CONDITIONS" \
  --alignment-margin "$ALIGNMENT_MARGIN" \
  --device cuda:0 \
  --bf16 \
  --trust-remote-code \
  --log-every 10 \
  --save-every 20
