# Fine-Tune Baseline Skeleton

This is the first training-script skeleton for the training machine. It is intentionally simple.

## What It Does

- reads frozen processed files
- loads `UnifiedSchemaDataset`
- trains a small multi-head classifier
- predicts:
  - `target_object`
  - `phase`
- writes:
  - `checkpoint.pt`
  - `metrics.json`

Current script:

- `scripts/train_finetune_baseline.py`

## What It Is Not

- not the final VLM fine-tuning pipeline
- not image-based yet
- not using a pretrained vision encoder yet
- not doing distributed training

Right now it is a training scaffold that proves the data/label path and optimization loop are wired correctly.

## Why This Is Still Useful

Use this first on the training machine to verify:

- parquet paths are correct
- labels load correctly
- torch runs
- loss decreases
- checkpoints can be saved

Once this works, replace the numeric feature encoder with the real observation encoder.

## Run

For a first smoke run:

```powershell
python scripts/train_finetune_baseline.py `
  --index data/processed/libero_index.parquet `
  --labels data/processed/schema_labels.parquet `
  --out-dir artifacts/finetune_baseline `
  --steps 200 `
  --batch-size 64 `
  --task multi
```

If you only want one head:

```powershell
python scripts/train_finetune_baseline.py --task target_object
python scripts/train_finetune_baseline.py --task phase
```

## Expected Output

The script prints metric snapshots at step 1, every 50 steps, and the final step.

Output files:

- `artifacts/finetune_baseline/checkpoint.pt`
- `artifacts/finetune_baseline/metrics.json`

## Next Replacement Point

When you move to the real fine-tuning stage, replace:

- the 5D numeric `x` feature input
- the small MLP encoder

with:

- image or image-window loading from `obs_ptr`
- optional instruction encoder
- the actual pretrained backbone you want to fine-tune

Keep the output heads and frozen weak-label pipeline stable for the first migration run.
