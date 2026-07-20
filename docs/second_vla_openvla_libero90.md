# Second VLA Checkpoint: OpenVLA Finetuned on LIBERO-90

Goal: add a second action policy/checkpoint for the Exp2 target-swap semantic-action gap. This is not a new VLA architecture, but it is a second OpenVLA action head/checkpoint with LIBERO-90 finetuning and a different action normalization key.

## Submit Job

```bash
MODEL_PATH=/scratch/bj2410/models/openvla-7b-finetuned-libero-90 \
OUT_DIR=outputs/linguistic_blindness/exp2_second_vla_openvla_libero90_target_swap_600 \
BENCHMARK=outputs/linguistic_blindness/benchmark_val_full_clean/counterfactual_examples.jsonl \
QWEN_PREDICTIONS=outputs/linguistic_blindness/qwen25vl7b_schema_libero_target_swap_clean_600/predictions.jsonl \
MAX_EXAMPLES=600 \
DATASET_FILTER=libero \
PERTURBATION_TYPES=target_swap \
REQUIRE_COUNTERFACTUAL_VALID=1 \
SHUFFLE=1 \
SAMPLE_SEED=7 \
UNNORM_KEY=libero_90_no_noops \
CONTROL_MODE=paraphrase \
THRESHOLD_METHOD=p95 \
LOG_EVERY=25 \
SAVE_EVERY=50 \
BF16=1 \
sbatch slurm/lb_exp2_action_sensitivity.sbatch
```

Expected output:

- `outputs/linguistic_blindness/exp2_second_vla_openvla_libero90_target_swap_600/predictions.jsonl`
- `outputs/linguistic_blindness/exp2_second_vla_openvla_libero90_target_swap_600/metrics.json`
- `outputs/linguistic_blindness/exp2_second_vla_openvla_libero90_target_swap_600/table2a_semantic_action_gap.csv`
- `outputs/linguistic_blindness/exp2_second_vla_openvla_libero90_target_swap_600/table2b_action_sensitivity_details.csv`

## Current Submission Status

On 2026-05-12, `sbatch` failed with:

```text
sbatch: error: Batch job submission failed: Unable to contact slurm controller (connect failure)
```

This is a Slurm controller issue; the command and input paths are valid.

## Interpretation

Use this as a second VLA checkpoint/action-policy robustness check:

> We repeat Exp2 with a LIBERO-90-finetuned OpenVLA checkpoint. This tests whether target-swap action insensitivity persists after task-specific finetuning.

Careful wording:

- This is a second checkpoint/action policy, not a new VLA architecture.
- Because it is finetuned on LIBERO-90, it should not be described as zero-shot.
- If it still shows low counterfactual action sensitivity, that strengthens the action-head/interface bottleneck claim.
- If it improves substantially, that is still informative: finetuning can partially reduce the gap, but then compare against schema sensitivity and normal/paraphrase controls.
