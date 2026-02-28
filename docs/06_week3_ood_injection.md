# Week 3 OOD Injection

This stage builds controllable unexpected cases directly on unified episode JSONL.

## Supported OOD events

- `not_releasing`
- `withdrawal`
- `occlusion`
- `ambiguous_contact`

## Inject one event

```powershell
$env:PYTHONPATH="src"
python scripts/inject_unexpected_ood.py `
  --input data/unified/genh2r/genh2r_raw_sample.jsonl `
  --output data/ood/genh2r/genh2r_raw_sample.occlusion.jsonl `
  --event occlusion `
  --start-ratio 0.5 `
  --duration-ratio 0.4 `
  --seed 7
```

## End-to-end after injection

```powershell
$env:PYTHONPATH="src"
python scripts/build_summaries.py `
  --input data/ood/genh2r/genh2r_raw_sample.occlusion.jsonl `
  --output artifacts/summaries/genh2r/genh2r_raw_sample.occlusion.summary.jsonl

python scripts/generate_schema_baseline.py `
  --input artifacts/summaries/genh2r/genh2r_raw_sample.occlusion.summary.jsonl `
  --output artifacts/schema/genh2r/genh2r_raw_sample.occlusion.schema.jsonl
```

## Build full 4-event OOD suite

```powershell
$env:PYTHONPATH="src"
python scripts/build_ood_suite.py `
  --input-dir data/unified/genh2r `
  --output-dir data/ood_suite/genh2r `
  --start-ratio 0.55 `
  --duration-ratio 0.25 `
  --seed 7
```

Then run summaries + schema + validity in batch:

```powershell
$env:PYTHONPATH="src"
python scripts/build_summaries_batch.py `
  --input-dir data/ood_suite/genh2r `
  --output-dir artifacts/summaries/ood_suite/genh2r `
  --window 10

python scripts/generate_schema_baseline_batch.py `
  --input-dir artifacts/summaries/ood_suite/genh2r `
  --output-dir artifacts/schema/ood_suite/genh2r

python scripts/eval_schema_validity_batch.py `
  --input-dir artifacts/schema/ood_suite/genh2r
```
