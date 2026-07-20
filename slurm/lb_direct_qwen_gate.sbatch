#!/bin/bash
#SBATCH --job-name=lb_direct_gate
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
MODEL_PATH="${MODEL_PATH:-/scratch/bj2410/models/Qwen2.5-VL-7B-Instruct}"
BENCHMARK="${BENCHMARK:-outputs/linguistic_blindness/direct_qwen_gate_combined_2700/benchmark.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/linguistic_blindness/direct_qwen_gate_combined_2700/qwen25vl7b_direct_gate}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
cd "$WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/$USER/hf-cache}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TMPDIR" "$(dirname "$OUT_DIR")"
CMD=("$PYTHON_BIN" scripts/lb_run_direct_qwen_gate.py --benchmark "$BENCHMARK" --model-path "$MODEL_PATH" --out-dir "$OUT_DIR" --max-examples "$MAX_EXAMPLES" --bf16 --trust-remote-code --log-every 50 --save-every 100)
printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
