#!/bin/bash
#SBATCH --job-name=lb_short_h
#SBATCH --account=torch_pr_637_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-$WORKDIR/.conda_vlm/bin/python}"
OUT_DIR="${OUT_DIR:-outputs/linguistic_blindness/short_horizon_target_approach_50}"
TASK_IDS="${TASK_IDS:-1,3,4,5,6}"
NUM_TRIALS="${NUM_TRIALS:-5}"
HORIZON="${HORIZON:-10}"
cd "$WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/third_party/LIBERO_pkg:$WORKDIR/third_party/openvla:$WORKDIR/src:$WORKDIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$WORKDIR/.libero_config"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/$USER/hf-cache}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$WORKDIR/.hf_modules_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$TMPDIR" "$(dirname "$OUT_DIR")"
CMD=("$PYTHON_BIN" scripts/lb_run_short_horizon_target_approach.py --task-ids "$TASK_IDS" --num-trials-per-task "$NUM_TRIALS" --horizon "$HORIZON" --out-dir "$OUT_DIR" --bf16 --trust-remote-code --center-crop)
printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
