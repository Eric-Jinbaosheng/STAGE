#!/bin/bash
#SBATCH --job-name=nlpfp_build_data
#SBATCH --account=torch_pr_675_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
VENV_DIR="${VENV_DIR:-$WORKDIR/.venv_torch}"
LIBERO_DATASETS="${LIBERO_DATASETS:-libero_spatial,libero_object,libero_goal}"

cd "$WORKDIR"
source "$VENV_DIR/bin/activate"

mkdir -p data/processed

python src/data/build_index_libero.py \
  --raw-dir data/libero_raw \
  --out data/processed/libero_index.parquet \
  --meta-out data/processed/meta.json \
  --datasets "$LIBERO_DATASETS"

python src/data/build_index_handover.py \
  --raw-dir data/raw/handoversim \
  --out data/processed/handover_index.parquet \
  --meta-out data/processed/meta.json

python src/data/build_schema_labels.py \
  --libero-index data/processed/libero_index.parquet \
  --handover-index data/processed/handover_index.parquet \
  --out data/processed/schema_labels.parquet \
  --meta-out data/processed/meta.json \
  --vocab-out data/processed/object_vocab.json

python - <<'PY'
import json
import pandas as pd
print("libero_index", pd.read_parquet("data/processed/libero_index.parquet").shape)
print("handover_index", pd.read_parquet("data/processed/handover_index.parquet").shape)
print("schema_labels", pd.read_parquet("data/processed/schema_labels.parquet").shape)
meta = json.load(open("data/processed/meta.json", "r", encoding="utf-8"))
print("ruleset", meta.get("schema_label_ruleset_version"), meta.get("target_object_ruleset_version"))
PY

echo "Processed data build finished."
