# Reproducibility

This repository is organized so the offline evaluation can be rerun without committing local model caches or generated outputs.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

For full VLM/VLA experiments, use the environment setup in `slurm/lb_setup_vlm_env.sbatch`, `slurm/lb_setup_libero_sim.sbatch`, or `slurm/lb_setup_octo_env.sbatch` as appropriate.

## Core Offline Pipeline

```bash
scripts/lb_build_benchmark.sh --max-observations 1000 --split val --out-dir outputs/linguistic_blindness/benchmark_val_1k
scripts/lb_run_eval.sh --benchmark outputs/linguistic_blindness/benchmark_val_1k/counterfactual_examples.jsonl --out-dir outputs/linguistic_blindness/eval_val_1k
scripts/lb_make_tables.sh outputs/linguistic_blindness/eval_val_1k
```

## Reporting Rule

Runs with `"smoke_test": true` in their metadata validate code paths only. Do not report smoke-test numbers as final results.

## Expected Generated Files

Evaluation runs write:

- `predictions.jsonl`
- `main_results.json`
- `main_results.csv`
- `failure_taxonomy.json`
- `failure_taxonomy.csv`
- `ablation_results.json`
- `ablation_results.csv`
- `paper_tables.md`
- `latex_tables.tex`
- `run_metadata.json`
