# GitHub Repo, Data Pull, And Fine-Tuning Setup

This document defines how to publish the code to GitHub without uploading datasets, then recreate the data locally on another machine and prepare for fine-tuning.

## Scope

- Push code, configs, schema specs, and docs to GitHub
- Do not push raw or processed data
- Re-pull data on each working or training machine
- Rebuild indexes and weak labels locally
- Prepare the machine for fine-tuning

Current frozen baseline:

- `schema_label_ruleset_version`: `week1_frozen_libero_phase_v2`
- `target_object_ruleset_version`: `week1_frozen_target_object_v2`
- LIBERO subsets in use: `libero_spatial`, `libero_object`, `libero_goal`
- Skipped for now: `libero_10`, `libero_90`, `libero_100`

## What Goes To GitHub

Push these:

- `src/`
- `scripts/`
- `schemas/`
- `configs/`
- `docs/`
- `README.md`
- `.gitignore`

Do not push these:

- `data/libero_raw/`
- `data/raw/`
- `data/proc/`
- `data/processed/`
- `data/unified/`
- `data/ood/`
- `data/ood_suite/`
- `summaries/`
- `viz/`
- `reports/`
- `artifacts/`
- `third_party/`

The repo already ignores these paths in `.gitignore`.

## Create The GitHub Repository

If this folder is not a git repo yet:

```powershell
git init
git add .
git commit -m "Initial project skeleton"
```

Create a remote repo on GitHub, then attach it:

```powershell
git remote add origin <your-github-repo-url>
git branch -M main
git push -u origin main
```

If you use GitHub CLI:

```powershell
gh repo create <repo-name> --private --source . --remote origin --push
```

Use a private repo unless you are sure the code and writeup are ready to share.

## Clone On A New Machine

```powershell
git clone <your-github-repo-url>
cd <repo-name>
```

Install the basic Python dependencies you need for data rebuild:

```powershell
pip install pandas pyarrow numpy h5py
```

If you plan to use official LIBERO utilities on that machine, also install:

```powershell
pip install robosuite gymnasium
```

Only install training dependencies after the data pipeline is verified.

## Re-Pull LIBERO Data

This project assumes you keep LIBERO raw files under `data/libero_raw/`.

### Option A: Copy Existing Raw Data

If you already have the raw LIBERO files on another disk or machine, copy only:

- `data/libero_raw/libero_spatial/`
- `data/libero_raw/libero_object/`
- `data/libero_raw/libero_goal/`

This is the fastest path.

### Option B: Re-Download With Official LIBERO Code

Clone the official LIBERO repo locally outside your project data paths:

```powershell
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
```

Create the local config directory:

```powershell
New-Item -ItemType Directory -Force .libero | Out-Null
```

Write `.libero/config.yaml` so LIBERO downloads into this project.
Replace `d:/NLP_Final_Project` below with the actual absolute path on the new machine:

```yaml
benchmark_root: d:/NLP_Final_Project/data/libero_raw
bddl_files: d:/NLP_Final_Project/third_party/LIBERO/libero/libero/bddl_files
init_states: d:/NLP_Final_Project/third_party/LIBERO/libero/libero/init_files
datasets: d:/NLP_Final_Project/data/libero_raw
assets: d:/NLP_Final_Project/third_party/LIBERO/libero/libero/assets
```

Then download only the subsets you currently use:

```powershell
python third_party/LIBERO/libero/lifelong/datasets.py --download-libero_spatial
python third_party/LIBERO/libero/lifelong/datasets.py --download-libero_object
python third_party/LIBERO/libero/lifelong/datasets.py --download-libero_goal
```

Do not download `libero_10`, `libero_90`, or `libero_100` for this baseline.

## Re-Pull Handover Data

This project assumes handover raw files live under `data/raw/handoversim/`.

### Option A: Copy Existing Exported Raw Episodes

If you already exported the JSON episodes, just copy:

- `data/raw/handoversim/*.json`

This is enough to rebuild `handover_index.parquet`.

### Option B: Rebuild From DexYCB Cache

If you have the cache files, place them under:

- `third_party/handover-sim/handover/data/dex-ycb-cache/`

Then run:

```powershell
python scripts/export_raw_from_dexycb_cache.py `
  --cache-dir third_party/handover-sim/handover/data/dex-ycb-cache `
  --out-dir data/raw/handoversim `
  --num-episodes 10 `
  --dt 0.01 `
  --max-frames 800
```

This rebuilds the raw JSON episodes used by the current handover baseline.

## Rebuild Local Indexes And Labels

After raw data is in place, rebuild indexes:

```powershell
python src/data/build_index_libero.py `
  --raw-dir data/libero_raw `
  --out data/processed/libero_index.parquet `
  --meta-out data/processed/meta.json `
  --datasets libero_spatial,libero_object,libero_goal
```

```powershell
python src/data/build_index_handover.py `
  --raw-dir data/raw/handoversim `
  --out data/processed/handover_index.parquet `
  --meta-out data/processed/meta.json
```

Then rebuild weak labels:

```powershell
python src/data/build_schema_labels.py `
  --libero-index data/processed/libero_index.parquet `
  --handover-index data/processed/handover_index.parquet `
  --out data/processed/schema_labels.parquet `
  --meta-out data/processed/meta.json `
  --vocab-out data/processed/object_vocab.json
```

## Verify Before Fine-Tuning

Run these checks before any training:

```powershell
python -c "import pandas as pd; print(pd.read_parquet('data/processed/libero_index.parquet').shape)"
python -c "import pandas as pd; print(pd.read_parquet('data/processed/handover_index.parquet').shape)"
python -c "import pandas as pd; s=pd.read_parquet('data/processed/schema_labels.parquet'); print(s['dataset'].value_counts().to_dict())"
python -c "import json; d=json.load(open('data/processed/meta.json','r',encoding='utf-8')); print(d['schema_label_ruleset_version'], d['target_object_ruleset_version'])"
```

Expected baseline:

- `libero_index.parquet`: about `200k` rows
- `handover_index.parquet`: `6000` rows
- `schema_labels.parquet`: about `206k` rows
- ruleset versions:
  - `week1_frozen_libero_phase_v2`
  - `week1_frozen_target_object_v2`

If these do not match, do not start fine-tuning yet.

## Fine-Tuning Start Point

This repo currently has:

- dataset/index builders
- weak-label generation
- `src/data/dataloader.py`

This repo does not yet contain the final training script you will use for fine-tuning on the other machine.

So the correct order is:

1. clone code
2. pull raw data locally
3. rebuild `data/processed/*`
4. verify `meta.json` and row counts
5. add or copy the training script on the training machine
6. start fine-tuning

## Minimal Smoke Test Before Training

Before full fine-tuning, confirm the dataloader can read the rebuilt files:

```powershell
python -c "from src.data.dataloader import UnifiedSchemaDataset; ds=UnifiedSchemaDataset('data/processed/libero_index.parquet','data/processed/schema_labels.parquet'); print(len(ds))"
```

If this fails, fix the data paths before training.

## Recommended First Fine-Tuning Policy

For the first run on the training machine:

- freeze the label rules
- do not add new datasets
- do not regenerate labels with edited heuristics
- train on the current frozen processed files only
- record the exact commit hash used for training

After the first stable run, you can version the next label changes separately.
