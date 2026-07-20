# Second Schema Probe Results: Qwen2.5-VL-3B

This experiment adds a second schema probe on the same 600 LIBERO target-swap examples used in Exp2. The OpenVLA action outputs are reused from the frozen Exp2 run, so the only change is the schema probe model.

## Inputs

- Benchmark: `outputs/linguistic_blindness/benchmark_val_full_clean/counterfactual_examples.jsonl`
- Filters: `dataset=libero`, `perturbation_type=target_swap`, `counterfactual_valid=true`
- Sampling: `shuffle=true`, `sample_seed=7`, `max_examples=600`
- Schema probe: `/scratch/bj2410/models/Qwen2.5-VL-3B-Instruct`
- Comparison action file: `outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/predictions.jsonl`

## Commands

Schema generation:

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

Comparison:

```bash
scripts/lb_compare_second_schema_probe.py \
  --second-schema outputs/linguistic_blindness/qwen25vl3b_schema_libero_target_swap_clean_600/predictions.jsonl \
  --exp2-predictions outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/predictions.jsonl \
  --out-dir outputs/linguistic_blindness/second_probe_qwen25vl3b_target_swap_600 \
  --probe-name qwen25vl3b_direct_schema
```

## Results

| Metric | Mean [95% CI] |
|---|---:|
| Qwen2.5-VL-3B target correctness | 1.000 [1.000, 1.000] |
| Qwen2.5-VL-3B schema sensitivity | 1.000 [1.000, 1.000] |
| OpenVLA action sensitivity | 0.068 [0.048, 0.090] |
| Semantic-action gap case rate | 0.932 [0.912, 0.950] |
| Aggregate schema-action gap | 0.932 |

Target-pair breakdown:

| Target Pair | N | 3B Schema Sens. | OpenVLA Action Sens. | Gap Case Rate |
|---|---:|---:|---:|---:|
| black bowl -> cookie box | 213 | 1.000 | 0.066 | 0.934 |
| black bowl -> ramekin | 255 | 1.000 | 0.047 | 0.953 |
| middle drawer -> top drawer | 132 | 1.000 | 0.114 | 0.886 |

## Interpretation

The second schema probe reaches the same target-sensitivity conclusion as Qwen2.5-VL-7B: schema-level target semantics track the counterfactual instruction, while OpenVLA's native 7-DoF action remains largely insensitive. This supports the claim that the semantic-action gap is not an artifact of a single Qwen2.5-VL-7B schema probe.

Careful wording: this is a second probe size within the Qwen2.5-VL family, not a completely different VLM architecture. It strengthens robustness but does not fully replace a cross-architecture VLM or second VLA/domain experiment.
