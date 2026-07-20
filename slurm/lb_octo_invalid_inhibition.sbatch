#!/bin/bash
#SBATCH --job-name=lb_octo_inv
#SBATCH --account=torch_pr_637_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-$WORKDIR/.conda_octo/bin/python}"
BENCHMARK="${BENCHMARK:-outputs/linguistic_blindness/exp3_invalid_instruction_clean_600/benchmark.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/linguistic_blindness/octo_invalid_instruction_inhibition_600}"
MAX_PER_PERTURBATION="${MAX_PER_PERTURBATION:-200}"
cd "$WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/src:$WORKDIR/scripts:$WORKDIR/third_party/octo${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/$USER/hf-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/$USER/xdg-cache}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$(dirname "$OUT_DIR")"
CMD=("$PYTHON_BIN" scripts/lb_run_octo_invalid_inhibition.py --benchmark "$BENCHMARK" --out-dir "$OUT_DIR" --max-per-perturbation "$MAX_PER_PERTURBATION" --log-every 25)
printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
