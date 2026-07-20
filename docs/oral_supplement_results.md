# Oral-Readiness Supplement Results

This note records the additional analyses added after the core Exp2--Exp5 results. These are supporting experiments; they should not change the main paper framing from counterfactual language sensitivity and action-interface verification.

## Human-Written Invalid Set

Status: annotation packet prepared, real human annotations still required.

Files:
- `outputs/linguistic_blindness/human_written_invalid_set_packet/human_invalid_annotation_sheet.csv`
- `outputs/linguistic_blindness/human_written_invalid_set_packet/annotation_protocol.md`
- `scripts/lb_build_human_invalid_benchmark.py`
- `scripts/lb_eval_human_invalid_methods.py`

Important: do not report this as human-written until annotators fill `human_instruction`, `gold_decision`, `gold_next_action`, and review fields. After annotation:

```bash
python scripts/lb_build_human_invalid_benchmark.py \
  --sheet outputs/linguistic_blindness/human_written_invalid_set_packet/human_invalid_annotation_sheet.csv \
  --require-approved
```

Then run Qwen schema and Direct Qwen Gate on the resulting benchmark, and evaluate with:

```bash
python scripts/lb_eval_human_invalid_methods.py \
  --benchmark outputs/linguistic_blindness/human_written_invalid_set/benchmark.jsonl \
  --direct-predictions <direct_gate_predictions.jsonl> \
  --visa-predictions <qwen_schema_predictions.jsonl>
```

## Direct Qwen Gate vs VISA

Direct Qwen Gate results on the existing combined split:

| Split | N | Direct correct | Safe deferral | Allow rate |
|---|---:|---:|---:|---:|
| Template invalid | 1800 | 0.407 | 0.407 | 0.593 |
| Hard invalid | 300 | 0.870 | 0.870 | 0.130 |
| Normal | 600 | 0.982 | 0.018 | 0.982 |

Template-invalid breakdown:

| Type | N | Direct safe deferral | Direct allow |
|---|---:|---:|---:|
| blank_instruction | 600 | 0.222 | 0.778 |
| impossible_instruction | 600 | 1.000 | 0.000 |
| negation | 600 | 0.000 | 1.000 |

Interpretation: direct allow/deny prompting handles impossible targets but fails badly on blank and negation, which supports explicit schema fields and deterministic checking.

Overlap with VISA:

| Split | Direct correct | VISA correct | Hybrid correct | Hybrid allow |
|---|---:|---:|---:|---:|
| Template invalid | 0.407 | 0.972 | 0.972 | 0.028 |
| Hard invalid | 0.870 | 0.780 | 1.000 | 0.000 |
| Normal | 0.982 | 0.940 | 0.922 | 0.922 |

Hybrid rule: allow only if both Direct Qwen Gate and VISA allow. It improves invalid safety but is more conservative on normal instructions.

## Schema Field Ablation

Output: `outputs/linguistic_blindness/schema_field_ablation/`

| Method | Template invalid | Hard invalid | Normal pass |
|---|---:|---:|---:|
| target_only | 0.901 | 0.127 | 1.000 |
| target_exists | 0.901 | 0.260 | 1.000 |
| target_exists_next_action | 0.972 | 0.780 | 0.940 |
| full_schema_no_checker | 0.972 | 0.780 | 0.940 |
| full_schema_checker | 0.972 | 0.983 | 0.940 |

Interpretation: target/existence fields alone are insufficient for hard invalid instructions. `next_action` gives a large gain, and the full checker substantially improves hard invalid safety by using blocked/constraint information.

## Closed-Loop Target-Swap Appendix

Output: `outputs/linguistic_blindness/exp6_closed_loop_target_swap_25_until_close/`

| Method | N | Final distance | Min distance | Full success |
|---|---:|---:|---:|---:|
| OpenVLA raw counterfactual | 25 | 0.320 | 0.298 | 0.000 |
| Schema projection | 25 | 0.071 | 0.063 | 0.000 |

Additional metrics:
- final-distance redirection: 0.249
- release rate: 0.960
- close-but-fail rate: 0.960
- target lift rate: 0.800

Writing guidance: use this only as an appendix sanity check. Claim that schema/action projection redirects trajectories toward the counterfactual target. Do not claim solved manipulation or task success.

## Octo Invalid-Inhibition

Output:
- `outputs/linguistic_blindness/octo_invalid_instruction_inhibition_600/`

Results:

| Split | N | Action inhibition | Blind execution |
|---|---:|---:|---:|
| overall | 1800 | 0.665 | 0.335 |
| blank | 600 | 0.695 | 0.305 |
| impossible | 600 | 0.855 | 0.145 |
| negation | 600 | 0.445 | 0.555 |

Interpretation: Octo is less blind than OpenVLA on invalid-instruction action inhibition, but still shows substantial blind execution, especially for negation.

## Visually Grounded Relation Counterfactual Split

Status: completed with MuJoCo-position oracle schema labels.

Files:
- `scripts/lb_build_visual_relation_split.py`
- `outputs/linguistic_blindness/visual_relation_counterfactual_600/benchmark.jsonl`
- `outputs/linguistic_blindness/visual_relation_counterfactual_600/oracle_schema_counterfactual.jsonl`
- `outputs/linguistic_blindness/visual_relation_openvla_action_sensitivity_600/`

Important framing: this is not a Qwen visual-grounding result. The schema target is an oracle derived from LIBERO/MuJoCo object positions. The point is to test whether OpenVLA's native action output responds to relation-defined target changes where the target name is not explicitly present in the instruction.

Target-name masking check:
- target name appears in relation instruction: 0.000
- anchor object name appears in relation instruction: 1.000
- mean target angle: 34.62 degrees
- mean target distance: 0.737

Main relation-split result:

| Subset | N | Oracle schema sens. | OpenVLA action sens. | SAG | Low sens. | Mean delta |
|---|---:|---:|---:|---:|---:|---:|
| overall | 600 | 1.000 | 0.077 | 0.923 | 0.923 | 1.454 |
| black bowl -> cookie box | 213 | 1.000 | 0.066 | 0.934 | 0.934 | 1.399 |
| black bowl -> ramekin | 255 | 1.000 | 0.067 | 0.933 | 0.933 | 1.458 |
| middle drawer -> top drawer | 132 | 1.000 | 0.114 | 0.886 | 0.886 | 1.535 |
| angle >= 30 deg | 496 | 1.000 | 0.062 | 0.938 | 0.938 | 1.411 |
| angle >= 45 deg | 44 | 1.000 | 0.045 | 0.955 | 0.955 | 1.415 |

Suggested writing: relation-defined targets produce a similarly large semantic-action gap even when the target name is absent from the counterfactual instruction. Because labels are oracle relation labels, this supports action-interface insensitivity under visually grounded target semantics, not standalone VLM visual reasoning.

## VISA-ActCheck / Semantic-Action Consistency Verifier

Status: completed for both target-name Exp2 and visual-relation split.

Files:
- `scripts/lb_visa_actcheck.py`
- `outputs/linguistic_blindness/visa_actcheck_exp2/`
- `outputs/linguistic_blindness/visual_relation_actcheck/`

Definition: compare the native action direction with the vector from end-effector to the schema target and to the swapped/wrong target. Flag/block if the action is more aligned to the wrong target, or if alignment is ambiguous within margin 0.05.

Exp2 target-name ActCheck:

| Subset | N | Target-aligned | Wrong-target | Ambiguous | Block/flag | OpenVLA sens. |
|---|---:|---:|---:|---:|---:|---:|
| overall | 600 | 0.682 | 0.223 | 0.095 | 0.318 | 0.068 |
| angle >= 45 deg | 44 | 0.409 | 0.409 | 0.182 | 0.591 | 0.045 |

Visual-relation ActCheck:

| Subset | N | Target-aligned | Wrong-target | Ambiguous | Block/flag | OpenVLA sens. |
|---|---:|---:|---:|---:|---:|---:|
| overall | 600 | 0.545 | 0.285 | 0.170 | 0.455 | 0.077 |

Suggested writing: VISA-ActCheck turns the schema into a semantic-action consistency verifier. It does not generate a better low-level action; it flags native actions that are not directionally consistent with the schema target. This strengthens VISA beyond invalid-instruction safe deferral.

## Pixel-Grounded Relation Probe

Status: completed with Qwen2.5-VL-7B image+instruction grounding.

Files:
- `scripts/lb_run_qwen_visual_relation_schema.py`
- `slurm/lb_qwen_visual_relation_schema.sbatch`
- `outputs/linguistic_blindness/qwen25vl7b_visual_relation_schema_600/`
- `scripts/lb_analyze_pixel_grounded_relation.py`
- `outputs/linguistic_blindness/pixel_grounded_relation_analysis/`

Results:

| Split | N | Grounding acc. | OpenVLA action sens. | Gap | Low sens. |
|---|---:|---:|---:|---:|---:|
| all | 600 | 0.958 | 0.077 | 0.882 | 0.923 |
| grounding-correct only | 575 | 1.000 | 0.077 | 0.923 | 0.923 |
| hard angle >= 45 deg | 44 | 1.000 | 0.045 | 0.955 | 0.955 |

This is the strongest version of the relation result: Qwen-VL grounds relation-defined targets from the image, and even on the subset where grounding is correct, OpenVLA's native action remains mostly insensitive.

Suggested writing: The semantic-action gap persists for relation-defined targets whose names are not present in the instruction. On 575 examples where the VLM schema probe correctly identifies the visually grounded target, OpenVLA action sensitivity remains 7.7%.

## Octo Relation-Defined / Pixel-Grounded Relation Robustness

Status: completed.

Files:
- `outputs/linguistic_blindness/octo_visual_relation_action_sensitivity_600/`
- `outputs/linguistic_blindness/octo_visual_relation_analysis/`
- `scripts/lb_analyze_policy_relation_and_actcheck.py`

Calibration:
- policy: `rail-berkeley/octo-small-1.5`
- threshold: Octo paraphrase-control P95 = 2.063
- do not reuse OpenVLA threshold.

Main relation result:

| Split | N | Grounding acc. | Octo action sens. | SAG | Low sens. |
|---|---:|---:|---:|---:|---:|
| all | 600 | 0.958 | 0.375 | 0.583 | 0.625 |
| grounding-correct only | 575 | 1.000 | 0.350 | 0.650 | 0.650 |
| hard angle >= 45 deg | 44 | 1.000 | 0.523 | 0.477 | 0.477 |
| black bowl -> cookie box | 213 | 1.000 | 0.244 | 0.756 | 0.756 |
| black bowl -> ramekin | 255 | 1.000 | 0.271 | 0.729 | 0.729 |
| middle drawer -> top drawer | 132 | 0.811 | 0.788 | 0.023 | 0.212 |

ActCheck on relation split:

| Split | N | Target-aligned | Wrong-target | Ambiguous | ActCheck flag |
|---|---:|---:|---:|---:|---:|
| all | 600 | 0.572 | 0.393 | 0.035 | 0.428 |
| grounding-correct only | 575 | 0.597 | 0.367 | 0.037 | 0.403 |
| hard angle >= 45 deg | 44 | 0.795 | 0.159 | 0.045 | 0.205 |

Interpretation: Octo is substantially more instruction-sensitive than OpenVLA, which is useful because it shows the metric distinguishes policies. However, on the grounding-correct relation subset, action sensitivity is still only 35.0%, leaving a 65.0-point semantic-action gap. This supports the narrower claim: the gap is reduced but not eliminated across action policies.

Writing guidance: do not write that all VLAs fail equally. Write that semantic-action sensitivity is a measurable behavioral axis; OpenVLA is severe, Octo is better, but relation-defined target semantics still do not fully survive into native actions.

## Short-Horizon Target-Approach Check

Status: completed.

Files:
- `scripts/lb_run_short_horizon_target_approach.py`
- `slurm/lb_short_horizon_target_approach.sbatch`
- `outputs/linguistic_blindness/short_horizon_target_approach_50/`

Setup:
- policy: OpenVLA LIBERO-90 finetuned
- tasks: 1, 3, 4, 5, 6
- trials per task: 5
- horizon: 10 steps
- pairs: 25 original-vs-counterfactual short rollouts

Results:

| Method | N | Counterfactual target approach rate | Original target approach rate | Mean delta dist to CF target | Mean target preference score |
|---|---:|---:|---:|---:|---:|
| original instruction rollout | 25 | 0.760 | 0.600 | 0.000184 | 0.000306 |
| counterfactual instruction rollout | 25 | 0.560 | 0.480 | 0.0000068 | 0.000287 |

Pair-level result:
- mean counterfactual-minus-original approach to counterfactual target: -0.000177

Interpretation: changing the instruction to the counterfactual target does not produce a stronger short-horizon movement toward the counterfactual target. This directly addresses the one-step critique: the behavior remains weak over the first 10 steps, not just in a single action vector.

Writing guidance: keep this as a short-horizon diagnostic, not as full closed-loop success. It supports "not merely a one-step artifact."

## Risk-Utility Curve / Operating Points

Output: `outputs/linguistic_blindness/risk_utility_direct_visa_hybrid/`

| Method | Normal pass | Invalid safe deferral | Balanced correct |
|---|---:|---:|---:|
| Direct Qwen Gate | 0.982 | 0.473 | 0.728 |
| VISA | 0.940 | 0.944 | 0.942 |
| Hybrid | 0.922 | 0.976 | 0.949 |

Interpretation: Direct Qwen Gate preserves normal execution but is weak on invalid safety. VISA gives a much better safety-preservation balance. Hybrid is safest but more conservative.

## Compositional Invalid Split

Status: programmatic split prepared. This is not human-written.

Files:
- `scripts/lb_build_compositional_invalid_set.py`
- `outputs/linguistic_blindness/compositional_invalid_set_300/benchmark.jsonl`

Categories include: prohibit-then-relation, relation-with-blocked-target, unless-not-visible, ambiguous-or, distractor mention, and conflict-same-target.

This split is ready for Qwen schema and Direct Gate runs if needed. It should be described as compositional stress testing, not as natural user data.

## BridgeData V2 Second-Domain Diagnostic

See `docs/bridge_v2_second_domain_results.md`.

Key result: on a 495-example real-robot BridgeData V2 invalid/prohibited subset, Octo-small-1.5 has 0.588 blind execution overall and 0.867 under negation. Direct Qwen Gate has 0.677 invalid safe deferral overall but only 0.030 on negation, while VISA/Qwen schema gate reaches 1.000 invalid safe deferral on blank/impossible/negation. This is a diagnostic second-domain supplement, not a full Bridge benchmark.

## Long-Horizon and Bridge Target-Change Supplements

See `docs/long_horizon_and_bridge_target_supplement.md`.

Key additions:

- K=10 short-horizon check: counterfactual instruction rollouts improve approach toward the counterfactual target in only 36% of paired rollouts; mean CF-target approach improvement relative to original rollout is slightly negative. This reduces the risk that the one-step semantic-action gap is merely an initial-action artifact.
- BridgeData V2 target/task-change diagnostic: on 165 fixed real-robot Bridge frames, Octo-small-1.5 action sensitivity is 0.661 with an oracle instruction-target/action gap of 0.339 under a Bridge/Octo-specific paraphrase-control threshold. This extends BridgeData beyond invalid-only evidence.
