#!/bin/bash
#SBATCH --job-name=nlpfp_pull_data
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
VENV_DIR="${VENV_DIR:-$WORKDIR/.venv_torch}"
DOWNLOAD_LIBERO="${DOWNLOAD_LIBERO:-1}"
DEXYCB_CACHE_DIR="${DEXYCB_CACHE_DIR:-}"

cd "$WORKDIR"
source "$VENV_DIR/bin/activate"

mkdir -p third_party data/libero_raw data/raw/handoversim .libero

if [[ "$DOWNLOAD_LIBERO" == "1" ]]; then
  if [[ ! -d third_party/LIBERO/.git ]]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
  fi

  cat > .libero/config.yaml <<EOF
benchmark_root: ${WORKDIR}/data/libero_raw
bddl_files: ${WORKDIR}/third_party/LIBERO/libero/libero/bddl_files
init_states: ${WORKDIR}/third_party/LIBERO/libero/libero/init_files
datasets: ${WORKDIR}/data/libero_raw
assets: ${WORKDIR}/third_party/LIBERO/libero/libero/assets
EOF

  python third_party/LIBERO/libero/lifelong/datasets.py --download-libero_spatial
  python third_party/LIBERO/libero/lifelong/datasets.py --download-libero_object
  python third_party/LIBERO/libero/lifelong/datasets.py --download-libero_goal
else
  echo "Skipping LIBERO download because DOWNLOAD_LIBERO=$DOWNLOAD_LIBERO"
fi

if [[ -n "$DEXYCB_CACHE_DIR" && -d "$DEXYCB_CACHE_DIR" ]]; then
  python scripts/export_raw_from_dexycb_cache.py \
    --cache-dir "$DEXYCB_CACHE_DIR" \
    --out-dir data/raw/handoversim \
    --num-episodes 10 \
    --dt 0.01 \
    --max-frames 800
else
  echo "Skipping handover export. Set DEXYCB_CACHE_DIR to an existing dex-ycb-cache directory to enable it."
fi

echo "Data pull stage finished."
