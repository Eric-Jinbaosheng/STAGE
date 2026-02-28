# Batch Evaluation

This script runs a trained baseline checkpoint on a batch of samples and reports:

- `target_object` accuracy
- `phase` accuracy
- confusion matrix for both heads
- a short prediction preview

Script:

- `scripts/run_batch_eval.py`

## Inputs

Required:

- `--checkpoint`
- `--text-vocab`

Optional:

- `--index`
- `--labels`
- `--dataset` to filter by dataset name, for example `libero`
- `--max-samples`
- `--batch-size`
- `--out`

If `--index` and `--labels` are omitted, the script reuses the paths saved in the checkpoint.

## Example

```powershell
python scripts/run_batch_eval.py `
  --checkpoint artifacts/finetune_baseline_gru_smoke/checkpoint.pt `
  --text-vocab artifacts/finetune_baseline_gru_smoke/text_vocab.json `
  --dataset libero `
  --max-samples 64 `
  --batch-size 8 `
  --out artifacts/batch_eval_smoke/result.json
```

## Output

The output JSON includes:

- `metrics.target_object_accuracy`
- `metrics.phase_accuracy`
- `label_spaces`
- `confusion.target_object`
- `confusion.phase`
- `preview` with a few example predictions

The confusion matrices are square integer matrices using the same class order listed in `label_spaces`.
