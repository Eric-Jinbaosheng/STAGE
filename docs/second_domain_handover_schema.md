# Second Domain: Handover Safety-Conflict Schema Probe

Goal: add a small second-domain setting beyond LIBERO. This run uses `handover_sim` examples from the full counterfactual benchmark and focuses on safety-conflict semantics.

This is not a VLA action-sensitivity experiment. It is a schema-level second-domain validation: can the schema probe recognize release/contact safety constraints outside LIBERO manipulation?

## Submit Job

```bash
MODEL_PATH=/scratch/bj2410/models/Qwen2.5-VL-7B-Instruct \
METHOD=qwen25vl7b_handover_schema \
OUT_DIR=outputs/linguistic_blindness/qwen25vl7b_schema_handover_safety_conflict_600 \
BENCHMARK=outputs/linguistic_blindness/benchmark_val_full_clean/counterfactual_examples.jsonl \
MAX_EXAMPLES=600 \
DATASET_FILTER=handover_sim \
PERTURBATION_TYPES=safety_conflict \
REQUIRE_COUNTERFACTUAL_VALID=0 \
SHUFFLE=1 \
SAMPLE_SEED=7 \
LOG_EVERY=50 \
SAVE_EVERY=100 \
BF16=1 \
sbatch slurm/lb_qwen_schema_eval.sbatch
```

Expected outputs:

- `outputs/linguistic_blindness/qwen25vl7b_schema_handover_safety_conflict_600/predictions.jsonl`
- `outputs/linguistic_blindness/qwen25vl7b_schema_handover_safety_conflict_600/main_results.json`
- `outputs/linguistic_blindness/qwen25vl7b_schema_handover_safety_conflict_600/failure_taxonomy.json`

## Current Submission Status

On 2026-05-12, `sbatch` failed with:

```text
sbatch: error: Batch job submission failed: Unable to contact slurm controller (connect failure)
```

This is a cluster/controller issue, not a script issue.

## Interpretation If Successful

Use this as a second-domain schema validation, not as the main OpenVLA action-head gap evidence.

Suggested wording:

> Beyond LIBERO manipulation, we also evaluate the schema probe on a handover-safety setting. This tests whether the Verifiable Interaction Schema captures non-manipulation interaction constraints such as human release and blocked RETRACT actions.

Do not overclaim this as a second VLA setting unless paired with action outputs from a handover policy.
