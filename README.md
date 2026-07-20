# STAGE: Semantic Transfer and Grounding Evaluation

Official repository for the EMNLP submission on linguistic blindness in embodied VLM/VLA agents.

This codebase contains an offline counterfactual evaluation pipeline for testing whether embodied agents preserve instruction semantics at the action-decision interface. The paper framing is language understanding and verification, not low-level robot-control performance.

## Overview

Task success is not the same as instruction understanding. An embodied agent can appear successful under a standard instruction while remaining insensitive to counterfactual language changes when the visual observation and robot state are fixed.

We study this failure mode as linguistic blindness: action-relevant interaction semantics fail to change correctly when only the language instruction changes. The repository provides:

- Counterfactual benchmark construction for target swaps, blank instructions, impossible targets, safety conflicts, negation, and phase conflicts.
- A Verifiable Interaction Schema interface for action-relevant language semantics.
- Deterministic schema checking and action gating.
- Metrics and table generation for semantic-action sensitivity, blind execution, safety violations, and failure taxonomy.
- Slurm launchers for optional VLM/VLA experiments.

## Repository Layout

```text
configs/                    Reproducibility configuration snapshots
docs/                       Paper notes, metrics, table definitions, and experiment summaries
scripts/lb_*.py             Experiment, analysis, and table-generation entry points
scripts/lb_*.sh             Convenience wrappers for core offline commands
slurm/lb_*.sbatch           Optional cluster launchers
src/linguistic_blindness/   Core benchmark, model wrapper, verification, and evaluation package
tests/                      Lightweight smoke tests
```

Large local data, model caches, third-party checkouts, virtual environments, and generated outputs are intentionally excluded from Git.

## Installation

Create a clean Python environment and install the lightweight offline package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

The core parser/checker/gate path only needs the Python standard library. `pandas` and `pyarrow` are used when building benchmarks from parquet data. Heavy VLA or simulation runs require their own environments and model dependencies; see the relevant `slurm/lb_*.sbatch` launchers.

## Data

The main benchmark builder expects processed files under `data/processed/`:

```text
data/processed/unified_index.parquet
data/processed/schema_labels.parquet
data/processed/instr_wrong.parquet
data/processed/episode_splits.json
```

If those files are missing, the builder defaults to a small smoke-test benchmark so the repository can still be checked end to end. Smoke-test numbers are only for pipeline validation and must not be reported as final experimental results.

## Reproduction

Build a counterfactual benchmark:

```bash
scripts/lb_build_benchmark.sh \
  --max-observations 1000 \
  --split val \
  --out-dir outputs/linguistic_blindness/benchmark_val_1k
```

Run parser/checker/gate evaluation:

```bash
scripts/lb_run_eval.sh \
  --benchmark outputs/linguistic_blindness/benchmark_val_1k/counterfactual_examples.jsonl \
  --out-dir outputs/linguistic_blindness/eval_val_1k
```

Run semantic-action gap analysis after schema and action predictions exist:

```bash
scripts/lb_semantic_action_gap.sh \
  schema_predictions.jsonl \
  action_predictions.jsonl \
  outputs/linguistic_blindness/semantic_action_gap/run
```

Print the table output locations for an evaluation run:

```bash
scripts/lb_make_tables.sh outputs/linguistic_blindness/eval_val_1k
```

Run the optional Qwen schema probe on Slurm:

```bash
sbatch slurm/lb_qwen_schema_eval.sbatch
```

## Outputs

Generated artifacts are written under `outputs/linguistic_blindness/` and are ignored by Git. Common files include:

- `counterfactual_examples.jsonl`
- `benchmark_statistics.json`
- `predictions.jsonl`
- `main_results.json` and `main_results.csv`
- `failure_taxonomy.json` and `failure_taxonomy.csv`
- `paper_tables.md`
- `latex_tables.tex`

## Development Check

```bash
pytest
```

## Citation

If you use this repository, please cite the associated paper. A placeholder citation file is provided in `CITATION.cff` and should be updated with the final EMNLP metadata after acceptance.

## License

This repository is released under the MIT License. See `LICENSE`.
