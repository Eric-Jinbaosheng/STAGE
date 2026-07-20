# Second Schema Probe: Qwen2.5-VL-3B

Goal: add a second schema probe on the same 600 LIBERO target-swap examples used in Exp2. This tests whether the semantic-action gap is specific to Qwen2.5-VL-7B or persists with another VLM probe size.

## Run 3B Schema Probe

Use the exact same benchmark/filter/sample seed as the frozen 7B Exp2 setup:

```bash
MODEL_PATH=/scratch/bj2410/models/Qwen2.5-VL-3B-Instruct \
METHOD=qwen25vl3b_direct_schema \
OUT_DIR=outputs/linguistic_blindness/qwen25vl3b_schema_libero_target_swap_clean_600 \
BENCHMARK=outputs/linguistic_blindness/benchmark_val_full_clean/counterfactual_examples.jsonl \
MAX_EXAMPLES=600 \
DATASET_FILTER=libero \
PERTURBATION_TYPES=target_swap \
REQUIRE_COUNTERFACTUAL_VALID=1 \
SHUFFLE=1 \
SAMPLE_SEED=7 \
LOG_EVERY=50 \
SAVE_EVERY=100 \
BF16=1 \
sbatch slurm/lb_qwen_schema_eval.sbatch
```

Expected output:

- `outputs/linguistic_blindness/qwen25vl3b_schema_libero_target_swap_clean_600/predictions.jsonl`
- `outputs/linguistic_blindness/qwen25vl3b_schema_libero_target_swap_clean_600/main_results.json`
- `outputs/linguistic_blindness/qwen25vl3b_schema_libero_target_swap_clean_600/run_metadata.json`

Note: on 2026-05-11, submission failed twice with `Unable to contact slurm controller`; this was a cluster/controller issue, not a script/model-path issue.

## Analyze Against Exp2 OpenVLA Actions

After the 3B job finishes:

```bash
scripts/lb_compare_second_schema_probe.py \
  --second-schema outputs/linguistic_blindness/qwen25vl3b_schema_libero_target_swap_clean_600/predictions.jsonl \
  --exp2-predictions outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/predictions.jsonl \
  --out-dir outputs/linguistic_blindness/second_probe_qwen25vl3b_target_swap_600 \
  --probe-name qwen25vl3b_direct_schema
```

Expected analysis outputs:

- `aligned_predictions.jsonl`
- `second_probe_metrics.csv`
- `second_probe_metrics.json`
- `target_pair_breakdown.csv`
- `latex_table_second_schema_probe.tex`

## Paper Interpretation

If Qwen2.5-VL-3B schema sensitivity is also high while OpenVLA action sensitivity remains low, write:

> The semantic-action gap is not an artifact of a single Qwen2.5-VL-7B schema probe. A second VLM schema probe on the same counterfactual examples also identifies the counterfactual target substantially more often than the OpenVLA action head changes its native 7-DoF action.

If 3B is weaker but still above action sensitivity, write:

> Even a smaller schema probe retains more counterfactual target sensitivity than the OpenVLA action head, although schema quality degrades relative to Qwen2.5-VL-7B.

If 3B fails, write it as a limitation and use this to motivate schema quality as an important part of the method.
