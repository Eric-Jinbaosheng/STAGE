#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${LB_PYTHON:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x "${ROOT}/.venv_torch/bin/python" ]]; then
    PY="${ROOT}/.venv_torch/bin/python"
  else
    PY="python"
  fi
fi
PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}" "${PY}" -m linguistic_blindness.evaluation.evaluate "$@"
