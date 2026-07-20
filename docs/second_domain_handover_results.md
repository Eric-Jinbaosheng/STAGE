# Second Domain Results: Handover Safety Conflict

This run evaluates Qwen2.5-VL-7B schema generation on a second domain, `handover_sim`, focusing on safety-conflict instructions. It is a schema-level domain robustness check, not a second VLA action-head experiment.

## Setup

- Benchmark: `outputs/linguistic_blindness/benchmark_val_full_clean/counterfactual_examples.jsonl`
- Dataset filter: `handover_sim`
- Perturbation type: `safety_conflict`
- Examples: 600
- Model: `/scratch/bj2410/models/Qwen2.5-VL-7B-Instruct`
- Output: `outputs/linguistic_blindness/qwen25vl7b_schema_handover_safety_conflict_600/`

## Main Results

| Metric | Value |
|---|---:|
| n | 600 |
| schema validity | 1.000 |
| safe deferral decision | 1.000 |
| RETRACT blocked | 1.000 |
| safety violation | 0.000 |
| action constraint consistency | 1.000 |
| exact phase/state match | 0.367 |

The schema probe always selected a safe deferral action (`ASK`) and always blocked `RETRACT`, which is the safety-critical behavior for this setup.

## Field-Level Notes

Generated next_action distribution:

- ASK: 600/600

Generated phase distribution:

- contact: 220
- reach: 176
- align: 146
- transfer: 58

Generated `human_released` distribution:

- false: 220
- true: 58
- null: 322

Generated `human_contact` distribution:

- true: 220
- false: 380

Generated `robot_contact` distribution:

- true: 278
- false: 322

## Interpretation

This result is useful as a second-domain safety-schema validation:

> In a handover-safety domain, the schema probe reliably identifies that execution should be deferred and that RETRACT should be blocked under safety conflict.

But it should not be overclaimed as perfect handover state understanding:

> Although the safety decision is correct, exact phase/contact-state reconstruction is imperfect. This supports the paper's distinction between execution-level safety semantics and full state estimation.

## Paper Wording

Recommended wording:

> Beyond LIBERO manipulation, we evaluate schema prediction on a handover-safety setting. The probe achieves 100% safe deferral and blocks RETRACT in all 600 safety-conflict cases, showing that the schema interface can express safety constraints outside the main LIBERO domain. However, exact phase/contact-state matching is lower, indicating that schema-level safety decisions can be reliable even when fine-grained state reconstruction is imperfect.

Avoid wording:

- Do not say this is a second VLA action-head gap experiment.
- Do not say Qwen perfectly understands handover state.
- Do not claim full handover control or rollout success from this run.
