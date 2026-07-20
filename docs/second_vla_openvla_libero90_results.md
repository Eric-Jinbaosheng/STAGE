# Second VLA Checkpoint Results: OpenVLA LIBERO-90 Finetuned

This repeats Exp2 target-swap action sensitivity with a second OpenVLA checkpoint: `/scratch/bj2410/models/openvla-7b-finetuned-libero-90`. The schema probe remains Qwen2.5-VL-7B, and the benchmark/sample is the same 600-example LIBERO target-swap split used in Exp2.

## Setup

- Benchmark: `outputs/linguistic_blindness/benchmark_val_full_clean/counterfactual_examples.jsonl`
- Dataset: LIBERO
- Perturbation: target_swap
- Examples: 600
- Schema predictions: `outputs/linguistic_blindness/qwen25vl7b_schema_libero_target_swap_clean_600/predictions.jsonl`
- VLA checkpoint: `/scratch/bj2410/models/openvla-7b-finetuned-libero-90`
- Unnorm key: `libero_90_no_noops`
- Output: `outputs/linguistic_blindness/exp2_second_vla_openvla_libero90_target_swap_600/`

The runner injected local `dataset_statistics.json` into `model.norm_stats`, which is required for this checkpoint's `libero_90_no_noops` action unnormalization.

## Results

| VLA Checkpoint | N | Schema Sens. | Action Sens. | Semantic-Action Gap | Low Sens. Rate |
|---|---:|---:|---:|---:|---:|
| OpenVLA-7B base | 600 | 1.000 | 0.068 | 0.932 | 0.932 |
| OpenVLA-7B LIBERO-90 finetuned | 600 | 1.000 | 0.057 | 0.943 | 0.943 |

Additional details for LIBERO-90 finetuned checkpoint:

- action sensitivity threshold: 2.798
- mean normalized action delta: 1.243
- mean paraphrase-control normalized delta: 1.061
- mean action cosine: 0.254

## Interpretation

The semantic-action gap persists under a second OpenVLA action checkpoint. In fact, the LIBERO-90 finetuned checkpoint is slightly less action-sensitive than the base checkpoint under the same target-swap protocol.

Recommended wording:

> We repeat the target-swap diagnostic with a LIBERO-90-finetuned OpenVLA checkpoint. Despite task-specific finetuning, action sensitivity remains low (0.057), while schema sensitivity remains 1.000, producing a semantic-action gap of 0.943. This suggests that the language-to-action bottleneck is not specific to the base OpenVLA checkpoint.

Careful wording:

- This is a second OpenVLA checkpoint/action policy, not a different VLA architecture.
- It is not zero-shot because the checkpoint is LIBERO-90 finetuned.
- The result supports robustness of the action-head/interface bottleneck claim, not a full cross-architecture VLA claim.
