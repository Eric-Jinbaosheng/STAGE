# Single-Sample Inference

This script runs one forward pass from a trained baseline checkpoint.

Script:

- `scripts/run_single_inference.py`

## Inputs

Required:

- `--checkpoint`: trained `checkpoint.pt`
- `--text-vocab`: matching `text_vocab.json`

Selection:

- preferred: `--episode-id` + `--frame-id`
- fallback: `--row-idx`

Optional:

- `--index`: override index parquet path
- `--labels`: override labels parquet path
- `--out`: write JSON result to a file

If `--index` and `--labels` are omitted, the script reuses the paths stored in the checkpoint args.

## Example

```powershell
python scripts/run_single_inference.py `
  --checkpoint artifacts/finetune_baseline_gru_smoke/checkpoint.pt `
  --text-vocab artifacts/finetune_baseline_gru_smoke/text_vocab.json `
  --episode-id open_the_middle_drawer_of_the_cabinet_demo:demo_0 `
  --frame-id 0
```

## Output

The script prints JSON with:

- sample metadata
- predicted `target_object`
- predicted `phase`
- top-3 class scores for both heads
- ground-truth weak labels from the merged table

This is the fastest way to verify:

- the checkpoint loads
- the vocab matches
- inference uses the same preprocessing path as training
