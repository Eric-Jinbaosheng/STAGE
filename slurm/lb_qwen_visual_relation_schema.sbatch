#!/bin/bash
#SBATCH --job-name=lb_qwen_visrel
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
BENCHMARK="${BENCHMARK:-outputs/linguistic_blindness/visual_relation_counterfactual_600/benchmark.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/linguistic_blindness/qwen25vl7b_visual_relation_schema_600}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
MODEL_PATH="${MODEL_PATH:-/scratch/bj2410/models/Qwen2.5-VL-7B-Instruct}"
cd "$WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/src:$WORKDIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/$USER/hf-cache}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TMPDIR" "$(dirname "$OUT_DIR")"
CMD=("$PYTHON_BIN" scripts/lb_run_qwen_visual_relation_schema.py --benchmark "$BENCHMARK" --model-path "$MODEL_PATH" --out-dir "$OUT_DIR" --max-examples "$MAX_EXAMPLES" --bf16 --trust-remote-code --log-every 25 --save-every 100)
printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
