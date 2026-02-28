# Unified Data Pipeline (LIBERO + Handover)

## 1) Build Handover index

```powershell
python src/data/build_index_handover.py `
  --raw-dir data/raw/handoversim `
  --out data/processed/handover_index.parquet `
  --meta-out data/processed/meta.json
```

## 2) Build LIBERO index

```powershell
python src/data/build_index_libero.py `
  --raw-dir data/libero_raw `
  --out data/processed/libero_index.parquet `
  --meta-out data/processed/meta.json
```

## 3) Build schema labels

```powershell
python src/data/build_schema_labels.py `
  --libero-index data/processed/libero_index.parquet `
  --handover-index data/processed/handover_index.parquet `
  --out data/processed/schema_labels.parquet `
  --meta-out data/processed/meta.json `
  --vocab-out data/processed/object_vocab.json
```

## 4) Sanity report

```powershell
python src/data/viz_sanity.py `
  --libero-index data/processed/libero_index.parquet `
  --handover-index data/processed/handover_index.parquet `
  --schema-labels data/processed/schema_labels.parquet `
  --report reports/week1_sanity.md `
  --out-dir reports/week1_assets
```

## 5) Dataloader smoke run

```powershell
python src/data/dataloader.py `
  --index data/processed/handover_index.parquet `
  --labels data/processed/schema_labels.parquet `
  --steps 200 `
  --batch-size 64
```
