#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 outputs/linguistic_blindness/eval/<run_dir>" >&2
  exit 2
fi
run_dir="$1"
if [[ ! -f "${run_dir}/paper_tables.md" ]]; then
  echo "missing ${run_dir}/paper_tables.md; run scripts/lb_run_eval.sh first" >&2
  exit 1
fi
printf 'Markdown table: %s\n' "${run_dir}/paper_tables.md"
printf 'LaTeX table: %s\n' "${run_dir}/latex_tables.tex"
