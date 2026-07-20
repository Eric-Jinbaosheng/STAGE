#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${LB_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv_torch/bin/python" ]]; then
    PYTHON_BIN=".venv_torch/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

SCHEMA_PREDICTIONS="${1:-}"
ACTION_PREDICTIONS="${2:-}"
OUT_DIR="${3:-outputs/linguistic_blindness/semantic_action_gap}"

if [[ -z "$SCHEMA_PREDICTIONS" || -z "$ACTION_PREDICTIONS" ]]; then
  echo "Usage: scripts/lb_semantic_action_gap.sh SCHEMA_PREDICTIONS_JSONL ACTION_PREDICTIONS_JSONL [OUT_DIR]" >&2
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
"$PYTHON_BIN" src/linguistic_blindness/evaluation/semantic_action_gap.py \
  --schema-predictions "$SCHEMA_PREDICTIONS" \
  --action-predictions "$ACTION_PREDICTIONS" \
  --out-dir "$OUT_DIR"
