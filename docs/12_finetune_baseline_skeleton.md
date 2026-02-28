# Fine-Tune Baseline Skeleton

This is the first training-script skeleton for the training machine. It is intentionally simple.

## What It Does

- reads frozen processed files
- loads `UnifiedSchemaDataset`
- trains a small multi-head classifier
- can load image windows from `obs_ptr` for LIBERO rows
- can fuse multiple camera views
- can tokenize the instruction, export a stable vocab file, and learn a text encoder
- predicts:
  - `target_object`
  - `phase`
- writes:
  - `checkpoint.pt`
  - `metrics.json`
  - `text_vocab.json`

Current script:

- `scripts/train_finetune_baseline.py`

## What It Is Not

- not the final VLM fine-tuning pipeline
- not using a pretrained vision encoder yet
- not doing distributed training

Right now it is a training scaffold that proves the data/label path, image-window loading path, text path, and optimization loop are wired correctly.

## Why This Is Still Useful

Use this first on the training machine to verify:

- parquet paths are correct
- labels load correctly
- torch runs
- loss decreases
- checkpoints can be saved

Once this works, replace the current lightweight encoders with the real observation encoder.

Current behavior:

- LIBERO rows: read RGB frames from HDF5 via `obs_ptr`
- optional multi-view: for example `agentview_rgb,eye_in_hand_rgb`
- optional multi-frame window around the current frame
- instruction text is tokenized into `input_ids` + `attention_mask`
- the dataset can export a stable `text_vocab.json`
- a learnable `Embedding + GRU` builds the text feature
- handover rows: no image exists yet, so the loader returns a zero image tensor and still uses numeric features

## Run

For a first smoke run:

```powershell
python scripts/train_finetune_baseline.py `
  --index data/processed/libero_index.parquet `
  --labels data/processed/schema_labels.parquet `
  --out-dir artifacts/finetune_baseline `
  --steps 200 `
  --batch-size 64 `
  --use-image `
  --image-size 64 `
  --frame-window 3 `
  --views agentview_rgb,eye_in_hand_rgb `
  --text-vocab-size 2048 `
  --text-max-len 16 `
  --text-embed-dim 64 `
  --text-hidden-dim 64 `
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
- `artifacts/finetune_baseline/text_vocab.json`

## Next Replacement Point

When you move to the real fine-tuning stage, replace:

- the current HDF5 frame-window loader
- the current lightweight tokenizer + GRU text encoder
- the small CNN + numeric/text fusion encoder

with:

- image or image-window loading from `obs_ptr`
- optional instruction encoder
- the actual pretrained backbone you want to fine-tune

Keep the output heads and frozen weak-label pipeline stable for the first migration run.
