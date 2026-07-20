# Three Additional Oral-Support Experiments

## 1. Pixel-Relation OpenVLA Hidden-State Probe

Goal: test whether OpenVLA hidden states encode relation-grounded target identity when the target name is not present in the instruction.

Data: 600 LIBERO pixel-grounded relation examples, e.g. `pick up the object to the right of the black bowl`.

Outputs:
- `outputs/linguistic_blindness/pixel_relation_hidden_state_probe/pixel_relation_hidden_probe_summary.csv`
- `outputs/linguistic_blindness/pixel_relation_hidden_state_probe/README.md`

| Condition | Target Acc | Schema Sens. | Action Sens. | Gap |
|---|---:|---:|---:|---:|
| full image + relation instruction | 1.000 | 1.000 | 0.094 | 0.906 |
| language-only / no image | 0.797 | 0.594 | 0.094 | 0.500 |
| shuffled-image | 0.819 | 0.656 | 0.094 | 0.561 |
| anchor-masked | 1.000 | 1.000 | 0.094 | 0.906 |
| relation-masked | 1.000 | 1.000 | 0.094 | 0.906 |
| random-label | 0.281 | 0.067 | 0.094 | -0.028 |

Interpretation: the full image+relation hidden representation perfectly recovers the grounded relation target on the held-out split, while language-only and shuffled-image controls drop substantially. Native OpenVLA action sensitivity remains low on the same held-out split. This supports a grounded semantic-to-action bottleneck, not just lexical target-word extraction.

Caveat: anchor-masked and relation-masked remain high, likely because these scenes often contain only two salient candidate objects and the remaining visual/textual cues are sufficient. The stronger controls are language-only, shuffled-image, and random-label.

## 2. ActCheck Predicts K=20 Short-Horizon Behavior

Goal: test whether one-step ActCheck labels predict subsequent short-horizon trajectory behavior.

Data: 100 paired K=20 LIBERO rollouts, with first-step ActCheck label recorded before rollout execution.

Outputs:
- `outputs/linguistic_blindness/actcheck_predicts_short_horizon_k20_100/actcheck_rollout_group_summary.csv`
- `outputs/linguistic_blindness/actcheck_predicts_short_horizon_k20_100/actcheck_rollout_distance_curves.csv`
- `outputs/linguistic_blindness/actcheck_predicts_short_horizon_k20_100/actcheck_rollout_examples.csv`

| First-step ActCheck | N | CF approach | Final target preference | Integrated target preference | Mean cos to CF | Mean cos to original |
|---|---:|---:|---:|---:|---:|---:|
| aligned | 32 | 0.844 | 0.063 | 0.063 | 0.712 | 0.566 |
| wrong-target | 12 | 0.750 | 0.044 | 0.044 | 0.357 | 0.445 |
| ambiguous | 56 | 0.696 | -0.013 | -0.013 | 0.496 | 0.497 |
| overall | 100 | 0.750 | 0.018 | 0.018 | 0.548 | 0.513 |

Interpretation: aligned first-step actions have the strongest positive target preference over K=20. Ambiguous first-step actions have negative final and integrated target preference, meaning the robot ends up closer to the original/wrong side on average. This supports using ActCheck as a predictor of short-horizon target behavior rather than only a one-step diagnostic.

## 3. BridgeData V2 Strict Target Audit Packet

Goal: prepare a strict real-robot BridgeData V2 subset that can be hand-audited for object-grounded target consistency.

Outputs:
- `outputs/linguistic_blindness/bridge_v2_strict_target_audit_subset/bridge_strict_target_audit_sheet.csv`
- `outputs/linguistic_blindness/bridge_v2_strict_target_audit_subset/contact_sheets/`
- `outputs/linguistic_blindness/bridge_v2_strict_target_audit_subset/bridge_strict_candidate_summary.csv`

Prepared candidates: 59 concrete-object examples.

Provisional concrete-text candidate result:

| Subset | N | Action Sensitivity | Caveat |
|---|---:|---:|---|
| concrete text candidates requiring visual audit | 59 | 0.678 | Not hand-labeled yet; do not call this strict object-grounded until the audit labels are filled. |

Important: I did not relabel this as a hand-labeled subset. The packet is ready for human visual audit. Once `strict_target_visible_label` and `competing_object_visible_label` are filled, rerun the summary on audited rows only.

## Final Sanity Supplements: CI and Significance

No new model runs were launched for this section. These are post-hoc bootstrap/permutation analyses over existing outputs.

Outputs:
- `outputs/linguistic_blindness/final_sanity_ci/pixel_relation_hidden_probe_ci_and_image_dependent_subset.csv`
- `outputs/linguistic_blindness/final_sanity_ci/pixel_relation_image_dependent_examples.csv`
- `outputs/linguistic_blindness/final_sanity_ci/actcheck_predictive_ci_by_group.csv`
- `outputs/linguistic_blindness/final_sanity_ci/actcheck_predictive_significance_tests.csv`

### Pixel-relation hidden probe CI

| Subset | N | Schema Sens. 95% CI | Action Sens. 95% CI | Gap 95% CI |
|---|---:|---:|---:|---:|
| full, all test pairs | 180 | 1.000 [1.000, 1.000] | 0.094 [0.056, 0.139] | 0.906 [0.861, 0.944] |
| language-only, all test pairs | 180 | 0.594 [0.522, 0.667] | 0.094 [0.056, 0.139] | 0.500 [0.417, 0.583] |
| shuffled-image, all test pairs | 180 | 0.656 [0.583, 0.722] | 0.094 [0.056, 0.139] | 0.561 [0.483, 0.639] |
| random-label, all test pairs | 180 | 0.067 [0.033, 0.106] | 0.094 [0.056, 0.139] | -0.028 [-0.083, 0.028] |
| image-dependent subset | 100 | 1.000 [1.000, 1.000] | 0.060 [0.020, 0.110] | 0.940 [0.890, 0.980] |

Image-dependent subset definition: full image+relation probe is correct, while language-only or shuffled-image control fails on the same test example.

### ActCheck predictive significance

Aligned vs ambiguous first-step ActCheck groups:

| Metric | Difference | 95% CI | Permutation p |
|---|---:|---:|---:|
| final target preference | 0.0760 | [0.0624, 0.0884] | 0.0001 |
| integrated target preference | 0.0761 | [0.0624, 0.0887] | 0.0001 |
| counterfactual approach rate | 0.1473 | [-0.0357, 0.3170] | 0.2022 |

Interpretation: ActCheck aligned actions predict significantly better final/integrated target preference over K=20. Approach-rate difference is directionally positive but not significant, so the stronger claim should use target-preference metrics, not binary approach rate.
