#!/bin/bash
#SBATCH --job-name=lb_ovla_patch
#SBATCH --account=torch_pr_637_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-$WORKDIR/.conda_vlm/bin/python}"
BENCHMARK="${BENCHMARK:-outputs/linguistic_blindness/benchmark_val_full_clean/counterfactual_examples.jsonl}"
EXP2_PREDICTIONS="${EXP2_PREDICTIONS:-outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/predictions.jsonl}"
MODEL_PATH="${MODEL_PATH:-/scratch/bj2410/models/openvla-7b}"
OUT_DIR="${OUT_DIR:-outputs/linguistic_blindness/openvla_activation_patching_100_layer16}"
MAX_EXAMPLES="${MAX_EXAMPLES:-100}"
LAYER_INDEX="${LAYER_INDEX:-16}"
ALPHA="${ALPHA:-1.0}"
THRESHOLD="${THRESHOLD:-2.2851185083448717}"
cd "$WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/src:$WORKDIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/$USER/hf-cache}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$WORKDIR/.hf_modules_cache}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$TMPDIR" "$(dirname "$OUT_DIR")"
"$PYTHON_BIN" scripts/lb_run_openvla_activation_patching.py \
  --benchmark "$BENCHMARK" \
  --exp2-predictions "$EXP2_PREDICTIONS" \
  --model-path "$MODEL_PATH" \
  --out-dir "$OUT_DIR" \
  --max-examples "$MAX_EXAMPLES" \
  --layer-index "$LAYER_INDEX" \
  --alpha "$ALPHA" \
  --threshold "$THRESHOLD" \
  --trust-remote-code \
  --bf16 \
  --log-every 10
