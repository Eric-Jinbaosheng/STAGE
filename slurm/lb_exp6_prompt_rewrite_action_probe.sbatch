#!/bin/bash
#SBATCH --job-name=lb-prw-probe
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

MODEL_PATH=${MODEL_PATH:-/scratch/bj2410/models/openvla-7b}
UNNORM_KEY=${UNNORM_KEY:-bridge_orig}
MAX_EXAMPLES=${MAX_EXAMPLES:-600}
REWRITE_PROFILE=${REWRITE_PROFILE:-concise}
OUT_DIR=${OUT_DIR:-outputs/linguistic_blindness/exp6_prompt_rewrite_action_probe_target_swap_600_${REWRITE_PROFILE}}
EXP2_PREDICTIONS=${EXP2_PREDICTIONS:-outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/predictions.jsonl}
QWEN_PREDICTIONS=${QWEN_PREDICTIONS:-outputs/linguistic_blindness/qwen25vl7b_schema_libero_target_swap_clean_600/predictions.jsonl}

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
if [[ ! -f "$EXP2_PREDICTIONS" ]]; then
  echo "Exp2 predictions not found at $EXP2_PREDICTIONS" >&2
  exit 1
fi
if [[ ! -f "$QWEN_PREDICTIONS" ]]; then
  echo "Qwen predictions not found at $QWEN_PREDICTIONS" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/lb_run_exp6_prompt_rewrite_action_probe.py \
  --exp2-predictions "$EXP2_PREDICTIONS" \
  --qwen-predictions "$QWEN_PREDICTIONS" \
  --index data/processed/unified_index.parquet \
  --model-path "$MODEL_PATH" \
  --unnorm-key "$UNNORM_KEY" \
  --out-dir "$OUT_DIR" \
  --max-examples "$MAX_EXAMPLES" \
  --rewrite-profile "$REWRITE_PROFILE" \
  --threshold 2.2851185083448717 \
  --device cuda:0 \
  --bf16 \
  --trust-remote-code \
  --log-every 25 \
  --save-every 50
