# Training Machine Migration

This project is frozen at the current weak-label baseline. Do not change label rules on the training machine before the first transfer run.

## Frozen Baseline

- Schema label ruleset: `week1_frozen_libero_phase_v2`
- Target object ruleset: `week1_frozen_target_object_v2`
- Active LIBERO subsets: `libero_spatial`, `libero_object`, `libero_goal`
- Skipped for now: `libero_10`, `libero_90`, `libero_100`
- No fine-tuning is performed on this machine

## Copy To Training Machine

Copy these files first:

- `data/processed/libero_index.parquet`
- `data/processed/handover_index.parquet`
- `data/processed/schema_labels.parquet`
- `data/processed/object_vocab.json`
- `data/processed/meta.json`
- `schema/schema_min.json`
- `src/data/dataloader.py`
- `src/data/build_schema_labels.py`
- `reports/week1_sanity.md`
- `reports/phase_template_summary.md`

If you want local reproducibility of raw/index rebuilds, also copy:

- `src/data/build_index_libero.py`
- `src/data/build_index_handover.py`
- `src/data/viz_sanity.py`
- `data/libero_raw/` (only if the training machine needs direct raw access)
- `data/raw/handoversim/` (only if the training machine needs direct raw access)

## Minimal Directory Layout

```text
project/
  data/
    processed/
      libero_index.parquet
      handover_index.parquet
      schema_labels.parquet
      object_vocab.json
      meta.json
  schema/
    schema_min.json
  src/
    data/
      dataloader.py
      build_schema_labels.py
```

## Verify After Copy

Run these checks on the training machine:

```powershell
python -c "import pandas as pd; print(pd.read_parquet('data/processed/libero_index.parquet').shape)"
python -c "import pandas as pd; print(pd.read_parquet('data/processed/handover_index.parquet').shape)"
python -c "import pandas as pd; s=pd.read_parquet('data/processed/schema_labels.parquet'); print(s['dataset'].value_counts().to_dict())"
python -c "import json; print(json.load(open('data/processed/meta.json','r',encoding='utf-8'))['schema_label_ruleset_version'])"
```

Expected baseline:

- `libero_index.parquet`: about `200k` rows
- `handover_index.parquet`: `6000` rows
- `schema_labels.parquet`: about `206k` rows
- `schema_label_ruleset_version`: `week1_frozen_libero_phase_v2`

## Rebuild Labels Only

If you only need to regenerate labels from the copied indexes:

```powershell
python src/data/build_schema_labels.py `
  --libero-index data/processed/libero_index.parquet `
  --handover-index data/processed/handover_index.parquet `
  --out data/processed/schema_labels.parquet `
  --meta-out data/processed/meta.json `
  --vocab-out data/processed/object_vocab.json
```

This should keep the same frozen rule versions unless you change the script.

## First Training Run

For the first transfer run:

- use only the frozen processed files
- do not rebuild labels with modified heuristics
- do not add new LIBERO subsets
- verify `meta.json` before training starts

If you need to change label logic later, bump the ruleset version first and save outputs to a new artifact path.
