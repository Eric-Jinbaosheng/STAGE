# Supplementary Experiments After Phase Adapter

This note summarizes the additional experiments added to address reviewer risks around one-step action deltas, schema quality, hidden-state probe leakage, and weak baselines.

## 1. Closed-loop target-swap evaluation

Source outputs:
- `outputs/linguistic_blindness/exp6_schema_projection_5tasks_alpha1_scale2_220`
- `outputs/linguistic_blindness/exp6_schema_projection_until_close_5tasks_d008`
- `outputs/linguistic_blindness/exp6_phase_adapter_task1_ramekin`
- `outputs/linguistic_blindness/exp6_phase_adapter_task3_cookiebox`
- `outputs/linguistic_blindness/exp6_phase_adapter_task5_ramekin`

Aggregated output:
- `outputs/linguistic_blindness/supp_closed_loop_target_swap/closed_loop_target_swap_summary.csv`

Main result: schema-conditioned projection consistently redirects the closed-loop trajectory toward the counterfactual schema target, even though full LIBERO task success remains zero.

Key numbers:
- Always-project, 5 tasks: raw final distance 0.336 -> projected final distance 0.038; redirection +0.298.
- Until-close, 5 tasks: raw final distance 0.336 -> projected final distance 0.065; redirection +0.271; release rate 1.0.
- Phase adapter single-case probes: target lifted in 3/3 selected cases, but full task success remains 0.

Interpretation: this should be framed as closed-loop target redirection / action-interface intervention evidence, not as a solved robot controller.

## 2. Hidden-state probe ablations

Output:
- `outputs/linguistic_blindness/supp_hidden_state_probe_ablations/hidden_probe_ablation_summary.csv`

Conditions:
- full: image + original instruction.
- language_only: instruction without image.
- mask_target: target object names replaced with `the object`.
- image_only: generic instruction with image, no target words.

Results:

| Condition | Target Acc. | Schema Sens. | OpenVLA Action Sens. | Gap |
|---|---:|---:|---:|---:|
| full | 1.000 | 1.000 | 0.078 | 0.922 |
| language_only | 1.000 | 1.000 | 0.078 | 0.922 |
| mask_target | 0.864 | 0.728 | 0.078 | 0.650 |
| image_only | 0.500 | 0.000 | 0.078 | -0.078 |

Interpretation: target semantics are linearly readable from OpenVLA action-prompt hidden states, but the strongest signal is language target identity. This supports an action-interface transfer bottleneck, but the claim should not be overstated as full visual grounding.

## 3. Schema field PR/F1 and hard-negative behavior

Output:
- `outputs/linguistic_blindness/supp_schema_prf_heuristic_baseline/schema_field_prf.csv`

Selected results:
- Target swap target_object: precision/recall/F1 = 1.000/1.000/1.000.
- Normal target_object: precision/recall/F1 = 1.000/1.000/1.000.
- Invalid safe deferral next_action: 0.972 accuracy.
- Invalid blocked_actions_task: precision 0.935, recall 0.348, F1 0.508.

Interpretation: schema target fields are strong, and safe deferral is strong. Blocked-action completeness is weaker, especially for invalid/negation cases, so the paper should distinguish execution-aware mitigation from strict semantic completeness.

## 4. Object-centric heuristic gate baseline

Output:
- `outputs/linguistic_blindness/supp_schema_prf_heuristic_baseline/object_centric_heuristic_gate.csv`

The implemented baseline is a simple instruction-string heuristic:
- blank instruction -> ASK
- prohibition words -> HOLD
- generated impossible target templates such as red mug / blue plate / green banana -> TARGET_NOT_FOUND
- otherwise ALLOW

Results on the current templated invalid set:
- invalid safe deferral rate: 1.000
- normal allow rate: 1.000

Interpretation: this baseline is strong on the current templated invalid benchmark because the invalid perturbations are programmatic. It is useful as a sanity baseline, but should be described as template-aware and not as a general semantic verifier. VISA remains broader because it represents target_exists, blocked_actions, phase/state, and gate decisions in a unified schema.

## Recommended paper positioning

Use these supplements to narrow and strengthen the claim:

- Main claim: fixed-observation counterfactual tests reveal that action interfaces can be insensitive to language semantics.
- Hidden-state probe claim: target identity is linearly readable from OpenVLA action-prompt hidden states, yet the native action head remains largely insensitive.
- Ablation caveat: much of the readable target signal comes from language tokens; this is expected because the counterfactual changes are linguistic, but it does not prove complete visual grounding.
- Closed-loop claim: schema-conditioned action-interface projection can redirect multi-step trajectories toward the counterfactual target, but full manipulation success is not solved.
