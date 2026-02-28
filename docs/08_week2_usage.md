# Week2 Usage

## 1. Generate schema from summaries

```powershell
$env:PYTHONPATH="src"
python src/schema_gen/run_schema_gen.py `
  --input summaries `
  --output artifacts/week2/schema_all.jsonl `
  --schema-spec schemas/interaction_schema.json `
  --max-retries 2
```

## 2. Sample 50-frame baseline check (Day2 target)

```powershell
$env:PYTHONPATH="src"
python src/schema_gen/run_schema_gen.py `
  --input summaries `
  --output artifacts/week2/schema_sample_50.jsonl `
  --schema-spec schemas/interaction_schema.json `
  --max-retries 2 `
  --sample-frames 50 `
  --seed 7
```

## 3. Full Week2 report

```powershell
python scripts/run_week2_baseline.py `
  --summaries-dir summaries `
  --schema-spec schemas/interaction_schema.json `
  --out-dir artifacts/week2 `
  --report reports/week2_baseline.md `
  --sample-frames 50
```
