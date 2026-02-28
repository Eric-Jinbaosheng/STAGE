# Week 2 Minimal Schema Baseline

This baseline turns summaries into constrained schema-like JSON and applies checker/repair.

## 1. Generate baseline schema JSONL

```powershell
$env:PYTHONPATH="src"
python scripts/generate_schema_baseline.py `
  --input artifacts/summaries/handover_sim/handover_sim_raw_sample.jsonl `
  --output artifacts/schema/handover_sim_raw_sample.schema.jsonl
```

## 2. Evaluate schema validity

```powershell
$env:PYTHONPATH="src"
python scripts/eval_schema_validity.py `
  --input artifacts/schema/handover_sim_raw_sample.schema.jsonl
```

## Baseline logic

- Phase is inferred with simple relation-threshold heuristics.
- Uncertainty fields are derived from relation scores.
- Checker enforces:
  - valid phase
  - no `retract` before robot secure
  - uncertainty-triggered affordance filtering to safe actions

This is intentionally minimal and should be replaced by constrained MLLM decoding and learned calibrator in later weeks.
