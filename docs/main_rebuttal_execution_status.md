# Main-Track Rebuttal Execution Status

This document tracks the new rebuttal package for the "main-track" story:
the semantic-action gap is broader than the original diagnostic, behavior-relevant,
and actionable at the interface boundary.

## Completed Local Artifacts

### SAT-Bench++

Generated:

- `outputs/linguistic_blindness/satbenchpp_v1/benchmark.jsonl`
- `outputs/linguistic_blindness/satbenchpp_v1/oracle_schema_counterfactual.jsonl`
- `outputs/linguistic_blindness/satbenchpp_v1/oracle_schema_original.jsonl`
- `outputs/linguistic_blindness/satbenchpp_v1/benchmark_statistics.json`

Builder:

```bash
python scripts/lb_build_satbenchpp.py \
  --out-dir outputs/linguistic_blindness/satbenchpp_v1 \
  --per-compositional-family 200 \
  --temporal-n 400
```

Current benchmark size:

| Split | N |
|---|---:|
| compositional | 600 |
| temporal/procedural | 400 |
| total | 1000 |

Family breakdown:

| Family | N |
|---|---:|
| attribute_target | 200 |
| relation_only | 200 |
| attribute_relation_compositional | 200 |
| first_target_order_swap | 134 |
| open_or_pick_before_other | 133 |
| move_before_manipulate | 133 |

The split is fixed-observation: observation, robot state, and target positions
are copied from the positioned LIBERO target-swap examples; only instruction
semantics and the gold schema target/first subgoal change.

### Existing Rebuttal Supplement Tables

Generated:

- `outputs/linguistic_blindness/rebuttal_main_supplements/threshold_free_auc_summary.csv`
- `outputs/linguistic_blindness/rebuttal_main_supplements/schema_action_quadrant_summary.csv`
- `outputs/linguistic_blindness/rebuttal_main_supplements/normal_utility_breakdown.csv`
- `outputs/linguistic_blindness/rebuttal_main_supplements/policy_domain_matrix.csv`
- `outputs/linguistic_blindness/rebuttal_main_supplements/schema_conditioned_action_repair_summary.csv`
- `outputs/linguistic_blindness/rebuttal_main_supplements/negation_stress_benchmark.jsonl`
- `outputs/linguistic_blindness/rebuttal_main_supplements/human_invalid_annotation_template.csv`

Command:

```bash
python scripts/lb_make_oral_final_supplements.py \
  --out-dir outputs/linguistic_blindness/rebuttal_main_supplements
```

Key existing numbers:

| Setting | N | Schema/oracle sens. | Native action sens. | SAG |
|---|---:|---:|---:|---:|
| OpenVLA target-name | 600 | 1.000 | 0.068 | 0.932 |
| OpenVLA pixel-relation | 600 | 0.958 | 0.077 | 0.882 |
| OpenVLA-LIBERO90 target-name | 600 | 1.000 | 0.057 | 0.943 |
| Octo target-name | 600 | 1.000 | 0.427 | 0.573 |
| Octo pixel-relation | 600 | 0.958 | 0.375 | 0.583 |

Threshold-free AUC:

| Setting | Target-vs-control AUC |
|---|---:|
| OpenVLA target-name | 0.573 |
| OpenVLA pixel-relation | 0.583 |
| BridgeData V2 Octo target/task change | 0.929 |

Schema/action quadrant on pixel-relation:

| Case | Count | Rate |
|---|---:|---:|
| schema correct + action consistent | 311 | 0.518 |
| schema correct + action wrong | 264 | 0.440 |
| schema wrong + action consistent | 16 | 0.027 |
| schema wrong + action wrong | 9 | 0.015 |

VISA utility:

| Setting | Value |
|---|---:|
| normal pass rate | 0.940 |
| invalid blind execution after VISA v3 | 0.028 |
| spatial-relation normal pass rate | 1.000 |

VISA-Rerank action-boundary result:

| Method | N | Allow coverage | Target-aligned | Wrong/ambiguous | Intervention |
|---|---:|---:|---:|---:|---:|
| Native counterfactual | 600 | n/a | 0.682 | 0.318 | 0.000 |
| Best schema-conditioned prompt candidate | 600 | n/a | 0.900 | 0.100 | 1.000 |
| VISA-Rerank | 600 | 0.938 | 1.000 | 0.000 | 0.718 |

This should be described carefully as a schema-conditioned action-boundary
reranking/handoff interface, not a learned controller or full manipulation
success.

## Submitted GPU Jobs

Final Slurm status:

| Job ID | Purpose | Output |
|---:|---|---|
| 13000216 | SAT-Bench++ primary submission | canceled after backup started |
| 13000220 | K=50 primary submission | canceled after backup started |
| 13000230 | VISA-R prompt-rewrite re-query on target-name | completed: `outputs/linguistic_blindness/visa_r_prompt_rewrite_target_swap_600_concise/` |
| 13000374 | SAT-Bench++ backup submission on `torch_pr_1091_general` | completed: `outputs/linguistic_blindness/satbenchpp_v1/openvla_action_sensitivity_1000_alt1091/` |
| 13000375 | K=50 backup submission on `torch_pr_1091_general` | completed: `outputs/linguistic_blindness/short_horizon_target_approach_k50_100_alt1091/` |
| 13000376 | VISA-R backup submission on `torch_pr_1091_general` | canceled after primary completed |
| 13063152 | VISA-Rerank target-swap action-boundary run | completed: `outputs/linguistic_blindness/visa_rerank_target_swap_600/` |
| 13063459 | VISA-Rerank 100-example quick check | completed: `outputs/linguistic_blindness/visa_rerank_target_swap_100_alt1091/` |
| 13067012 | VISA-Rerank on SAT-Bench++ compositional/temporal | completed: `outputs/linguistic_blindness/visa_rerank_satbenchpp_1000_primary/` |
| 13067011 | VISA-Rerank K=50 short-horizon trajectory probe | completed: `outputs/linguistic_blindness/short_horizon_visa_rerank_k50_primary/` |
| 13069772 | VISA-Rerank K=50 N=100 allowed-subset probe | completed: `outputs/linguistic_blindness/short_horizon_visa_rerank_k50_n100_primary/` |
| 13166250 | VISA-Rerank K=100 N=50 with no-schema baseline | completed: `outputs/linguistic_blindness/short_horizon_visa_rerank_k100_n50_schema_baselines_primary/` |
| 13166282 | Best-of-N no-schema target-swap static probe | completed: `outputs/linguistic_blindness/visa_rerank_no_schema_target_swap_600/` |
| 13166283 | Best-of-N no-schema SAT-Bench++ static probe | completed: `outputs/linguistic_blindness/visa_rerank_no_schema_satbenchpp_1000/` |
| 13210870 | Schema-intervention candidate pool on target-swap N=80 | completed: `outputs/linguistic_blindness/schema_intervention_candidate_pool_target_swap_n80/` |
| 13210871 | Schema-intervention candidate pool on SAT-Bench++ N=80 | completed: `outputs/linguistic_blindness/schema_intervention_candidate_pool_satbenchpp_n80/` |
| 13212573 | Larger schema-intervention SAT-Bench++ N=300, budgets 2/4 | completed: `outputs/linguistic_blindness/schema_intervention_candidate_pool_satbenchpp_n300_b2b4/` |
| 13212575 | Larger schema-intervention temporal/procedural N=120, budgets 2/4 | completed: `outputs/linguistic_blindness/schema_intervention_candidate_pool_temporal_n120_b2b4/` |
| 13216362 | VISA-ClosedLoop true LIBERO env execution pilot | completed but quarantined: `outputs/linguistic_blindness/visa_closed_loop_execution_spatial_5tasks_1trial_b220_r0/` |
| 13223155 | Native original-task reproduction check for old 4/10 LIBERO90 setting | completed: `outputs/linguistic_blindness/exp6_sim_gate_native_repro_4tasks_20260709_233345/` |
| 13231021 | Same-wrapper Native original validation on previous LIBERO90 success tasks 0/3 | completed: `outputs/linguistic_blindness/visa_closed_loop_original_validation_libero90_t0_t3_native_20260710_003606/` |
| 13231654 | Same-wrapper Native original validation on spatial movable counterfactual candidates 1/3/5/6 | completed: `outputs/linguistic_blindness/visa_closed_loop_original_validation_spatial_movable_native_20260710_004947/` |
| 13234617 | Same-wrapper Native original validation on LIBERO90 base-competent candidate tasks 3/4 | completed: `outputs/linguistic_blindness/visa_closed_loop_original_validation_libero90_t3_t4_native_20260710_015626/` |
| local | Counterfactual evaluator oracle state intervention on LIBERO90 tasks 3/4, chocolate-pudding target | completed: `outputs/linguistic_blindness/counterfactual_evaluator_sanity_libero90_t3_t4_choc/` |
| 13235527 | Model-free VISA-Handoff upper-bound pilot on LIBERO90 tasks 3/4, chocolate-pudding target | completed but not positive: `outputs/linguistic_blindness/visa_closed_loop_base_competent_cf_libero90_t3_t4_choc_handoff_only_220_l40s_20260710_021641/` |
| 13237729 | VISA-Residual pilot on LIBERO90 tasks 3/4, chocolate-pudding target | completed, initial N=2 positive pilot: `outputs/linguistic_blindness/visa_residual_base_competent_cf_libero90_t3_t4_choc_b220_l40s_20260710_030426/` |
| 13238250 | Generic-residual baseline on LIBERO90 tasks 3/4, chocolate-pudding target | completed: `outputs/linguistic_blindness/generic_residual_base_competent_cf_libero90_t3_t4_choc_b220_l40s_20260710_031408/` |
| 13238651 | Expanded paired residual steering on LIBERO90 tasks 3/4, 5 trials each | completed, not positive on contact/lift: `outputs/linguistic_blindness/visa_residual_base_competent_cf_libero90_t3_t4_choc_trials5_b220_20260710_032346/` |
| 13252509 | Selective phase-aware residual correction on LIBERO90 tasks 3/4, 5 trials each | completed on h200_tandon | `outputs/linguistic_blindness/visa_selective_residual_base_competent_cf_libero90_t3_t4_choc_trials5_b220_h200_tandon_20260710_0515/` |

Selective phase-aware correction result: completed job 13252509 on
`h200_tandon` with `torch_pr_769_tandon_advanced`. Earlier failed selective submission
13248480 only ran the native task-3 subset because comma-separated Slurm
environment variables were parsed incorrectly; do not use it as the selective
result. The selective/phase-aware implementation is in
`scripts/lb_run_visa_closed_loop_execution.py` with methods `gate_only`,
`generic_selective_residual`, `visa_selective_residual`, and
`oracle_selective_residual`.

| Method | N | Correct contact | Correct lift | Wrong contact | Wrong lift | Wrong-object approach | Final target pref | Correction rate | Mean alpha |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native OpenVLA | 10 | 0.900 | 0.700 | 0.000 | 0.000 | 0.700 | 0.117 | 0.000 | 0.000 |
| Gate-only | 10 | 0.700 | 0.600 | 0.000 | 0.000 | 0.700 | 0.096 | 0.000 | 0.000 |
| Generic selective residual | 10 | 0.900 | 0.900 | 0.000 | 0.000 | 0.400 | 0.178 | 0.350 | 0.105 |
| VISA selective residual | 10 | 0.900 | 0.800 | 0.000 | 0.000 | 0.400 | 0.190 | 0.287 | 0.086 |
| Oracle selective residual | 10 | 1.000 | 0.900 | 0.000 | 0.000 | 0.500 | 0.196 | 0.271 | 0.081 |

Lift transitions relative to Native:

| Method | Native fail -> success | Native success -> success | Native success -> fail | Native fail -> fail |
|---|---:|---:|---:|---:|
| Gate-only | 0 | 6 | 1 | 3 |
| Generic selective residual | 2 | 7 | 0 | 1 |
| VISA selective residual | 2 | 6 | 1 | 1 |
| Oracle selective residual | 3 | 6 | 1 | 0 |

Interpretation: the strict selective/phase-aware correction fixes the major
failure of continuous residual steering. It no longer destroys manipulation:
VISA-selective improves correct lift over Native from 7/10 to 8/10, and
Oracle-selective reaches 9/10 instead of the earlier Oracle-Residual 3/10.
VISA-selective repairs 2/3 Native lift failures but breaks 1/7 Native lift
successes. This supports the design rule that correction should be gated to
high-confidence wrong-target approach and should use small xy residuals.
However, schema-specific advantage is not established because Generic-selective
reaches 9/10 lift and breaks 0/7 Native successes. Treat this as a promising
auxiliary closed-loop design probe, not as full controller success.

## Final Output Package

Main final table:

- `outputs/linguistic_blindness/main_rebuttal_final/main_rebuttal_tables.md`

SAT-Bench++ analysis:

- `outputs/linguistic_blindness/satbenchpp_v1/analysis/satbenchpp_summary.csv`
- `outputs/linguistic_blindness/satbenchpp_v1/analysis/satbenchpp_report.md`
- `outputs/linguistic_blindness/satbenchpp_v1/analysis/satbenchpp_predictions_augmented.jsonl`

K=50 rollout analysis:

- `outputs/linguistic_blindness/actcheck_predicts_short_horizon_k50_100/actcheck_rollout_group_summary.csv`
- `outputs/linguistic_blindness/actcheck_predicts_short_horizon_k50_100/actcheck_rollout_examples.csv`
- `outputs/linguistic_blindness/actcheck_predicts_short_horizon_k50_100/report.md`

VISA-R prompt-rewrite:

- `outputs/linguistic_blindness/visa_r_prompt_rewrite_target_swap_600_concise/metrics.json`
- `outputs/linguistic_blindness/visa_r_prompt_rewrite_target_swap_600_concise/target_pair_summary.csv`

VISA-Rerank action-boundary interface:

- `outputs/linguistic_blindness/visa_rerank_target_swap_600/metrics.json`
- `outputs/linguistic_blindness/visa_rerank_target_swap_600/summary.csv`
- `outputs/linguistic_blindness/visa_rerank_target_swap_600/predictions.jsonl`
- `outputs/linguistic_blindness/visa_rerank_satbenchpp_1000_primary/summary_by_split.csv`
- `outputs/linguistic_blindness/visa_rerank_satbenchpp_1000_primary/summary_by_family.csv`

VISA-Rerank K=50 behavior probe:

- `outputs/linguistic_blindness/short_horizon_visa_rerank_k50_primary/summary.csv`
- `outputs/linguistic_blindness/short_horizon_visa_rerank_k50_primary/summary.json`
- `outputs/linguistic_blindness/short_horizon_visa_rerank_k50_primary/short_horizon_visa_rerank_rollouts.jsonl`
- `outputs/linguistic_blindness/short_horizon_visa_rerank_k50_n100_primary/summary.csv`
- `outputs/linguistic_blindness/short_horizon_visa_rerank_k50_n100_primary/summary_allowed_matched.csv`
- `outputs/linguistic_blindness/short_horizon_visa_rerank_k50_n100_primary/short_horizon_visa_rerank_rollouts.jsonl`

VISA-Rerank K=100 behavior probe with no-schema baseline:

- `outputs/linguistic_blindness/short_horizon_visa_rerank_k100_n50_schema_baselines_primary/summary.csv`
- `outputs/linguistic_blindness/short_horizon_visa_rerank_k100_n50_schema_baselines_primary/summary_allowed_matched.csv`
- `outputs/linguistic_blindness/short_horizon_visa_rerank_k100_n50_schema_baselines_primary/summary.json`
- `outputs/linguistic_blindness/short_horizon_visa_rerank_k100_n50_schema_baselines_primary/short_horizon_visa_rerank_rollouts.jsonl`

Best-of-N no-schema static baselines:

- `outputs/linguistic_blindness/visa_rerank_no_schema_target_swap_600/summary.csv`
- `outputs/linguistic_blindness/visa_rerank_no_schema_target_swap_600/predictions.jsonl`
- `outputs/linguistic_blindness/visa_rerank_no_schema_satbenchpp_1000/summary.csv`
- `outputs/linguistic_blindness/visa_rerank_no_schema_satbenchpp_1000/summary_by_split.csv`
- `outputs/linguistic_blindness/visa_rerank_no_schema_satbenchpp_1000/predictions.jsonl`

Candidate-pool budget and schema-intervention analyses:

- `outputs/linguistic_blindness/candidate_pool_budget_target_swap/candidate_budget_summary.csv`
- `outputs/linguistic_blindness/candidate_pool_budget_satbenchpp/candidate_budget_summary.csv`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_target_swap_n80/summary.csv`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_target_swap_n80/predictions.jsonl`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_satbenchpp_n80/summary.csv`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_satbenchpp_n80/predictions.jsonl`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_satbenchpp_n300_b2b4/summary.csv`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_satbenchpp_n300_b2b4/predictions.jsonl`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_satbenchpp_n300_b2b4/stats/paired_candidate_recall_stats.csv`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_satbenchpp_n300_b2b4/stats/schema_redirection_stats.csv`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_satbenchpp_n300_b2b4/target_flip/candidate_target_identity_summary.csv`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_satbenchpp_n300_b2b4/target_flip/paired_best_target_flip_summary.csv`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_satbenchpp_n300_b2b4/intervened_target_support/intervened_target_support_summary.csv`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_temporal_n120_b2b4/summary.csv`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_temporal_n120_b2b4/predictions.jsonl`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_temporal_n120_b2b4/stats/paired_candidate_recall_stats.csv`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_temporal_n120_b2b4/stats/schema_redirection_stats.csv`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_temporal_n120_b2b4/target_flip/candidate_target_identity_summary.csv`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_temporal_n120_b2b4/target_flip/paired_best_target_flip_summary.csv`
- `outputs/linguistic_blindness/schema_intervention_candidate_pool_temporal_n120_b2b4/intervened_target_support/intervened_target_support_summary.csv`

External validation packet:

- `outputs/linguistic_blindness/external_validation_packet_300/external_validation_sheet.csv`
- `outputs/linguistic_blindness/external_validation_packet_300/annotation_protocol.md`

## Final Results Snapshot

SAT-Bench++:

| Split | N | Semantic recovery | Native action sensitivity | SAG | Native ActCheck aligned | AUC |
|---|---:|---:|---:|---:|---:|---:|
| overall | 1000 | 1.000 | 0.061 | 0.939 | 0.544 | 0.538 |
| compositional | 600 | 1.000 | 0.070 | 0.930 | 0.513 | 0.571 |
| temporal/procedural | 400 | 1.000 | 0.048 | 0.953 | 0.590 | 0.491 |

K=50 rollout:

| First-step ActCheck | N | CF approach | Original approach | Final target pref. | Integrated pref. |
|---|---:|---:|---:|---:|---:|
| aligned | 32 | 0.938 | 0.719 | 0.062 | 0.063 |
| wrong-target | 12 | 0.667 | 0.750 | 0.042 | 0.044 |
| ambiguous | 56 | 0.732 | 0.696 | -0.012 | -0.013 |
| overall | 100 | 0.790 | 0.710 | 0.018 | 0.018 |

VISA-R prompt rewrite:

| N | Raw action sensitivity | Schema rewrite sensitivity | Gain | Rewrite improves over raw |
|---:|---:|---:|---:|---:|
| 600 | 0.068 | 0.067 | -0.002 | 0.503 |

VISA-Rerank action-boundary interface:

| Method | N | Allow coverage | Defer rate | Intervention rate | ActCheck aligned | Wrong/ambiguous | Action sensitivity | Mean consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native counterfactual | 600 | n/a | 0.000 | 0.000 | 0.682 | 0.318 | 0.068 | 0.147 |
| Best schema-conditioned prompt candidate | 600 | n/a | 0.000 | 1.000 | 0.900 | 0.100 | 0.095 | 0.302 |
| VISA-Rerank | 600 | 0.938 | 0.062 | 0.718 | 1.000 | 0.000 | 0.094 | 0.370 |

VISA-Rerank on SAT-Bench++:

| Split | Method | N | Allow coverage | ActCheck aligned | Wrong/ambiguous |
|---|---|---:|---:|---:|---:|
| compositional | Native counterfactual | 600 | n/a | 0.513 | 0.487 |
| compositional | VISA-Rerank | 600 | 0.890 | 1.000 | 0.000 |
| temporal/procedural | Native counterfactual | 400 | n/a | 0.590 | 0.410 |
| temporal/procedural | VISA-Rerank | 400 | 0.907 | 1.000 | 0.000 |

VISA-Rerank K=50 short-horizon trajectory probe:

| Method | N | Final target pref | Integrated pref | CF approach | Wrong-object approach | First-close correct | Defer rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native counterfactual | 100 | -0.0001 | 0.0177 | 0.790 | 0.640 | 0.000 | 0.000 |
| Single target prompt | 100 | 0.0008 | 0.0182 | 0.800 | 0.700 | 0.000 | 0.000 |
| VISA-Rerank | 100 | 0.0032 | 0.0194 | 0.750 | 0.520 | 0.000 | 0.691 |
| Primitive handoff | 100 | 0.0006 | 0.0184 | 1.000 | 1.000 | 0.000 | 0.000 |

VISA-Rerank K=50 allowed-subset matched probe:

| Subset | Method | N | Coverage | Final target pref | Integrated pref | CF approach | Wrong-object approach | Wrong-object AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| any allowed | Native counterfactual | 44 | 0.440 | -0.0008 | 0.0494 | 0.886 | 0.614 | 0.0058 |
| any allowed | VISA-Rerank | 44 | 0.440 | 0.0065 | 0.0525 | 0.841 | 0.568 | 0.0006 |
| fully allowed | Native counterfactual | 24 | 0.240 | 0.0015 | 0.0575 | 0.958 | 0.500 | 0.0002 |
| fully allowed | VISA-Rerank | 24 | 0.240 | 0.0112 | 0.0608 | 0.958 | 0.583 | 0.0008 |

Best-of-N no-schema static baseline:

| Setting | Method | N | Allow coverage | ActCheck aligned | Wrong/ambiguous | Defer rate |
|---|---|---:|---:|---:|---:|---:|
| target-swap | Native counterfactual | 600 | n/a | 0.682 | 0.318 | 0.000 |
| target-swap | Best generic prompt candidate | 600 | n/a | 0.935 | 0.065 | 0.000 |
| target-swap | No-schema rerank | 600 | 0.955 | 1.000 | 0.000 | 0.045 |
| compositional | Native counterfactual | 600 | n/a | 0.513 | 0.487 | 0.000 |
| compositional | Best generic prompt candidate | 600 | n/a | 0.920 | 0.080 | 0.000 |
| compositional | No-schema rerank | 600 | 0.935 | 1.000 | 0.000 | 0.065 |
| temporal/procedural | Native counterfactual | 400 | n/a | 0.590 | 0.410 | 0.000 |
| temporal/procedural | Best generic prompt candidate | 400 | n/a | 0.935 | 0.065 | 0.000 |
| temporal/procedural | No-schema rerank | 400 | 0.948 | 1.000 | 0.000 | 0.052 |

VISA-Rerank K=100 short-horizon trajectory probe with no-schema baseline:

| Method | N | Final target pref | Integrated pref | CF approach | Wrong-object approach | First-close correct | Defer rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native counterfactual | 50 | 0.0002 | 0.0178 | 0.780 | 0.720 | 0.020 | 0.000 |
| Single target prompt | 50 | 0.0035 | 0.0191 | 0.860 | 0.800 | 0.000 | 0.000 |
| Best-of-N no schema | 50 | 0.0069 | 0.0208 | 0.740 | 0.540 | 0.000 | 0.704 |
| VISA-Rerank | 50 | 0.0083 | 0.0216 | 0.760 | 0.600 | 0.000 | 0.720 |
| Primitive handoff | 50 | 0.0007 | 0.0172 | 1.000 | 1.000 | 0.000 | 0.000 |

K=100 matched subset where VISA-Rerank exposes at least one action:

| Method | N | Coverage | Final target pref | Integrated pref | CF approach | Wrong-object approach | Wrong-object AUC | Defer rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native counterfactual | 23 | 0.460 | -0.0047 | 0.0467 | 0.913 | 0.739 | 0.0222 | 0.000 |
| Single target prompt | 23 | 0.460 | 0.0046 | 0.0491 | 0.913 | 0.826 | 0.0256 | 0.000 |
| Best-of-N no schema | 23 | 0.460 | 0.0143 | 0.0542 | 0.783 | 0.522 | -0.0004 | 0.357 |
| VISA-Rerank | 23 | 0.460 | 0.0174 | 0.0559 | 0.870 | 0.696 | 0.0028 | 0.392 |
| Primitive handoff | 23 | 0.460 | 0.0009 | 0.0481 | 1.000 | 1.000 | 0.0095 | 0.000 |

Quarantined VISA-ClosedLoop true LIBERO env-step pilot:

| Method | N | Env success | Correct contact | Correct lift | Final target pref | Integrated pref | Target approach AUC | Wrong-object approach | Fallback rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native OpenVLA | 5 | 0.000 | 0.000 | 0.000 | 0.0059 | 0.0190 | 0.0129 | 0.800 | 0.000 |
| Generic proposal-selection | 5 | 0.000 | 0.000 | 0.000 | 0.0184 | 0.0281 | 0.0094 | 0.600 | 0.552 |
| VISA-ClosedLoop | 5 | 0.000 | 0.000 | 0.000 | 0.0267 | 0.0275 | 0.0231 | 0.800 | 0.715 |
| Oracle-target primitive handoff | 5 | 0.000 | 0.000 | 0.000 | 0.0013 | 0.0146 | 0.0226 | 1.000 | 0.000 |

Interpretation: do not use this as rebuttal evidence yet. The pilot is not
aligned with the old 4/10 OpenVLA normal-task sanity setting: it uses
LIBERO-spatial counterfactual tasks, while the old sanity check used LIBERO90
original task language and official success. Because Native OpenVLA and the
oracle-target primitive also have 0 contact/lift, this is a pipeline/evaluator
alignment warning, not clean evidence for or against VISA.

Native original-task reproduction for old LIBERO90 sanity setting:

| Setting | Task suite | Tasks | Trial | Normal success | Notes |
|---|---|---:|---:|---:|---|
| Old full sanity | LIBERO90 | 10 | 0 | 0.400 | `outputs/linguistic_blindness/exp6_libero90_schema_gate_sanity_10x1_v2_400invalid/` |
| New reproduction | LIBERO90 | 4 | 0 | 0.500 | `outputs/linguistic_blindness/exp6_sim_gate_native_repro_4tasks_20260709_233345/` |

Interpretation: the original-task OpenVLA execution path is still live under
the old setting: task 0 and task 3 succeed, task 1 and task 2 fail, matching the
old pattern. This supports quarantining the N=5 counterfactual closed-loop
pilot as an alignment issue rather than treating it as evidence that OpenVLA or
VISA categorically cannot execute in LIBERO.

Same-pipeline validation ladder:

| Check | Task suite | Task IDs | Method | Success/contact-lift result | Output |
|---|---|---|---|---|---|
| New VISA-ClosedLoop wrapper, original instruction | LIBERO90 | 0,3 | Native OpenVLA | 2/2 env success | `outputs/linguistic_blindness/visa_closed_loop_original_validation_libero90_t0_t3_native_20260710_003606/` |
| New VISA-ClosedLoop wrapper, original instruction | LIBERO90 | 3,4 | Native OpenVLA | 2/2 env success | `outputs/linguistic_blindness/visa_closed_loop_original_validation_libero90_t3_t4_native_20260710_015626/` |
| New VISA-ClosedLoop wrapper, original instruction | LIBERO-spatial | 1,3,5,6 | Native OpenVLA | 0/4 env success | `outputs/linguistic_blindness/visa_closed_loop_original_validation_spatial_movable_native_20260710_004947/` |
| Counterfactual evaluator oracle state intervention | LIBERO-spatial | 1,3,4,5,6 | direct qpos intervention | 4/5 contact/lift oracle pass | `outputs/linguistic_blindness/counterfactual_evaluator_sanity_spatial_5tasks/` |
| Counterfactual evaluator oracle state intervention | LIBERO90 | 3,4 | direct qpos intervention on chocolate pudding | 2/2 contact/lift oracle pass | `outputs/linguistic_blindness/counterfactual_evaluator_sanity_libero90_t3_t4_choc/` |
| VISA-Handoff primitive upper-bound | LIBERO90 | 3,4 | schema target primitive, no OpenVLA model | 0/2 correct contact, 0/2 correct lift | `outputs/linguistic_blindness/visa_closed_loop_base_competent_cf_libero90_t3_t4_choc_handoff_only_220_l40s_20260710_021641/` |
| VISA-Residual steering | LIBERO90 | 3,4 | native action plus non-destructive xy residual | 2/2 correct contact, 2/2 correct lift | `outputs/linguistic_blindness/visa_residual_base_competent_cf_libero90_t3_t4_choc_b220_l40s_20260710_030426/` |
| Generic residual steering | LIBERO90 | 3,4 | same residual selector baseline | 2/2 correct contact, 1/2 correct lift | `outputs/linguistic_blindness/generic_residual_base_competent_cf_libero90_t3_t4_choc_b220_l40s_20260710_031408/` |
| Expanded residual steering | LIBERO90 | 3,4 | paired Native/Generic/VISA/Oracle residual, 5 trials each | Native 8/10 lift, Generic 6/10, VISA 5/10, Oracle 3/10 | `outputs/linguistic_blindness/visa_residual_base_competent_cf_libero90_t3_t4_choc_trials5_b220_20260710_032346/` |

Interpretation: the wrapper itself is not broken, because it reproduces Native
success on the known successful LIBERO90 tasks. The spatial counterfactual
candidate tasks are not suitable for an end-to-end base-policy-competent subset,
because Native cannot complete their original instructions under the same
wrapper. The evaluator proxy is valid for movable free-joint objects
(ramekin/cookie box) but not for the top-drawer case, which needs a different
predicate.

The later LIBERO90 task 3/4 base-competent pilot validates the experimental
ladder but does not produce a positive end-to-end VISA result. Native
original-task execution is 2/2 under the same wrapper, and the
chocolate-pudding counterfactual contact/lift predicates pass oracle state
intervention (2/2). However, model-free VISA-Handoff reaches only weak target
distance approach and obtains 0/2 correct contact and 0/2 correct lift. A
canceled longer run also showed Native counterfactual task 3 can already
contact/lift the chocolate pudding, so this two-episode pilot is not evidence
that VISA improves closed-loop execution. Do not cite it as rebuttal evidence
for task-level success.

Non-destructive residual steering also should not be used as actionability
evidence. Instead of replacing the policy, VISA-Residual preserves the native
z/rotation/gripper commands and only steers xy translation toward the schema
target. The initial two-episode pilot was positive, but over 10 paired
counterfactual rollouts Native already reaches correct contact/lift on 8/10
episodes, while Generic-Residual reaches 6/10 and VISA-Residual reaches 5/10.
Oracle-Residual is also worse than Native (3/10 lift), so the failure is a
controller/timing issue rather than a schema-target issue. VISA-Residual improves
softer trajectory proxies, raising final target preference from 0.094 to 0.194
and lowering wrong-object approach rate from 0.80 to 0.20, but this does not
translate into manipulation success. Treat the entire closed-loop repair line as
internal negative evidence for simple handoff/residual controllers, not as a
rebuttal result.

Pre-rerank candidate budget curves from existing candidate pools:

| Setting | Generator | Recall@1 | Recall@2 | Recall@4 | Recall@6 | Best margin@6 |
|---|---|---:|---:|---:|---:|---:|
| target-swap | Schema-conditioned | 0.682 | 0.813 | 0.918 | 0.938 | 0.344 |
| target-swap | Generic no-schema | 0.682 | 0.813 | 0.922 | 0.955 | 0.330 |
| SAT-Bench++ | Schema-conditioned | 0.544 | 0.712 | 0.856 | 0.897 | 0.291 |
| SAT-Bench++ | Generic no-schema | 0.544 | 0.745 | 0.874 | 0.940 | 0.300 |

Archived N=80 schema-intervention sanity check:

| Setting | Conditioning | Recall@2 | Recall@4 | Recall@8 | Recall@16 | Schema-target presence |
|---|---|---:|---:|---:|---:|---:|
| target-swap N=80 | Generic | 0.775 | 0.875 | 0.975 | 0.988 | 0.000 at N=16 |
| target-swap N=80 | Correct schema | 0.838 | 0.938 | 0.975 | 0.975 | 0.000 at N=16 |
| target-swap N=80 | Field-dropped schema | 0.812 | 0.912 | 0.975 | 0.988 | n/a |
| target-swap N=80 | Target-swapped schema | 0.862 | 0.950 | 0.975 | 0.988 | 0.838 at N=16 |
| SAT-Bench++ N=80 | Generic | 0.738 | 0.838 | 0.925 | 0.988 | 0.000 at N=16 |
| SAT-Bench++ N=80 | Correct schema | 0.762 | 0.875 | 0.950 | 0.975 | 0.000 at N=16 |
| SAT-Bench++ N=80 | Field-dropped schema | 0.700 | 0.788 | 0.912 | 0.962 | n/a |
| SAT-Bench++ N=80 | Target-swapped schema | 0.788 | 0.938 | 0.938 | 0.975 | 0.850 at N=16 |
| temporal/procedural N=23 | Generic | 0.739 | 0.783 | 0.870 | 0.957 | 0.000 at N=8 |
| temporal/procedural N=23 | Correct schema | 0.826 | 0.913 | 0.957 | 0.957 | 0.000 at N=8 |
| temporal/procedural N=23 | Target-swapped schema | 0.826 | 0.913 | 0.913 | 0.957 | 0.913 at N=8 |

Larger schema-intervention candidate-pool experiment:

| Setting | Conditioning | Recall@2 | Recall@4 | Best margin@4 | Wrong/amb cand. rate@4 | Schema-target presence@4 |
|---|---|---:|---:|---:|---:|---:|
| SAT-Bench++ N=300 | Generic | 0.733 | 0.860 | 0.236 | 0.481 | 0.000 |
| SAT-Bench++ N=300 | Correct schema | 0.770 | 0.877 | 0.280 | 0.452 | 0.000 |
| SAT-Bench++ N=300 | Field-dropped schema | 0.683 | 0.797 | 0.205 | 0.550 | n/a |
| SAT-Bench++ N=300 | Target-swapped schema | 0.783 | 0.930 | 0.280 | 0.430 | 0.670 |
| temporal/procedural N=120 | Generic | 0.733 | 0.883 | 0.247 | 0.469 | 0.000 |
| temporal/procedural N=120 | Correct schema | 0.758 | 0.867 | 0.271 | 0.450 | 0.000 |
| temporal/procedural N=120 | Field-dropped schema | 0.742 | 0.842 | 0.208 | 0.548 | n/a |
| temporal/procedural N=120 | Target-swapped schema | 0.800 | 0.892 | 0.264 | 0.419 | 0.592 |

Paired recall stats on SAT-Bench++ N=300:

| Comparison | Budget | Diff | 95% bootstrap CI | McNemar p |
|---|---:|---:|---|---:|
| Correct schema - generic | 2 | +0.037 | [-0.013, 0.087] | 0.200 |
| Correct schema - generic | 4 | +0.017 | [-0.030, 0.063] | 0.568 |
| Correct schema - field-dropped | 2 | +0.087 | [0.040, 0.137] | 0.0005 |
| Correct schema - field-dropped | 4 | +0.080 | [0.030, 0.130] | 0.0027 |

Target identity / mixed-pool check:

| Setting | Conditioning | Budget | Cand. correct-target | Cand. schema-target | Best correct-target | Best schema-target | Mixed correct+schema pool |
|---|---|---:|---:|---:|---:|---:|---:|
| SAT-Bench++ N=300 | Target-swapped schema | 2 | 0.562 | 0.277 | 0.783 | 0.100 | 0.267 |
| SAT-Bench++ N=300 | Target-swapped schema | 4 | 0.570 | 0.288 | 0.930 | 0.020 | 0.600 |
| temporal/procedural N=120 | Target-swapped schema | 2 | 0.588 | 0.292 | 0.800 | 0.142 | 0.267 |
| temporal/procedural N=120 | Target-swapped schema | 4 | 0.581 | 0.277 | 0.892 | 0.042 | 0.492 |

Unified intervened-target support check:

| Setting | Conditioning | Budget | Cand. intervened-target | Pool intervened-target presence | Mixed correct+intervened pool | Best intervened-target |
|---|---|---:|---:|---:|---:|---:|
| SAT-Bench++ N=300 | Generic | 4 | 0.360 | 0.730 | 0.593 | 0.070 |
| SAT-Bench++ N=300 | Correct schema | 4 | 0.307 | 0.680 | 0.563 | 0.037 |
| SAT-Bench++ N=300 | Field-dropped schema | 4 | 0.380 | 0.710 | 0.517 | 0.063 |
| SAT-Bench++ N=300 | Target-swapped schema | 4 | 0.288 | 0.670 | 0.600 | 0.020 |
| temporal/procedural N=120 | Generic | 4 | 0.358 | 0.708 | 0.600 | 0.067 |
| temporal/procedural N=120 | Correct schema | 4 | 0.312 | 0.675 | 0.558 | 0.042 |
| temporal/procedural N=120 | Field-dropped schema | 4 | 0.383 | 0.750 | 0.592 | 0.083 |
| temporal/procedural N=120 | Target-swapped schema | 4 | 0.277 | 0.592 | 0.492 | 0.042 |

Interpretation: SAT-Bench++ strongly supports benchmark breadth; K=50 supports
behavioral relevance of ActCheck labels. Prompt-level VISA-R re-query did not
improve action sensitivity, so it should not be presented as successful prompt
repair. However, schema-conditioned action-boundary reranking does improve
target-consistent action exposure: it allows 93.8% of cases, defers 6.2%, and
removes wrong/ambiguous exposed actions among allowed cases. On SAT-Bench++, the
same action-exposure effect holds for compositional and temporal/procedural
splits. The N=100 K=50 and N=50 K=100 short-horizon trajectory probes give short-horizon
anti-circularity evidence: all-case VISA-Rerank improves final/integrated
target preference and reduces wrong-object approach relative to Native, but
defers frequently and does not improve first-close. Best-of-N no-schema is a
strong control, especially on wrong-object suppression, so the final rebuttal
should not claim schema is the sole source of the gains. The strongest framing
is that candidate generation plus action-boundary selection improves exposed
action quality, while VISA's schema adds structured grounding, defer semantics,
interpretability, and handoff. The candidate-pool experiments sharpen this:
generic prompting can catch up at larger budgets in the existing pools, but
correct schema improves low-budget recall on SAT-Bench++ but the paired CI
against generic crosses zero; the robust statistical evidence is that dropping
the target field hurts recall. Target-swapped schema creates mixed pools, but a
unified intervened-target comparison shows it does not increase intervened-target
support over generic. Thus the safe claim is that schema target fields improve
candidate quality relative to ablations, while schema-conditioned generation
alone produces mixed proposal pools. The target-flip analysis resolves the
apparent contradiction that target-swapped schema has both high correct-target
Recall@4 and high schema-target presence: Recall@N is an existence metric, and
60.0% of SAT-Bench++ budget-4 target-swapped pools contain both correct-target
and schema-target candidates. Best-candidate flips to the swapped target are
rare under the correct-target selector, reinforcing the role of verified
selection.

External human validation is not complete yet. The packet is prepared, but
three non-author filled annotation sheets are still required before reporting
agreement/kappa numbers.

## Commands To Run When Jobs Finish

Analyze SAT-Bench++ action results:

```bash
python scripts/lb_analyze_satbenchpp.py \
  --benchmark outputs/linguistic_blindness/satbenchpp_v1/benchmark.jsonl \
  --actions outputs/linguistic_blindness/satbenchpp_v1/openvla_action_sensitivity_1000/predictions.jsonl \
  --out-dir outputs/linguistic_blindness/satbenchpp_v1/analysis
```

Or run the auto-finalizer, which checks primary and backup output locations:

```bash
python scripts/lb_finalize_main_rebuttal.py \
  --out-dir outputs/linguistic_blindness/main_rebuttal_final
```

Analyze K=50 ActCheck-predicts-rollout:

```bash
python scripts/lb_analyze_actcheck_predicts_rollout.py \
  --rollouts outputs/linguistic_blindness/short_horizon_target_approach_k50_100/short_horizon_rollouts.jsonl \
  --out-dir outputs/linguistic_blindness/actcheck_predicts_short_horizon_k50_100
```

If VISA-R prompt-rewrite finishes, summarize it from:

```bash
outputs/linguistic_blindness/visa_r_prompt_rewrite_target_swap_600_concise/metrics.json
outputs/linguistic_blindness/visa_r_prompt_rewrite_target_swap_600_concise/target_pair_summary.csv
```

## Rebuttal Claim Status

Ready to state now:

- Original SAT-Bench target-name and pixel-relation gaps are strong.
- SAT-Bench++ compositional and temporal/procedural splits retain a large gap.
- K=50 rollout shows first-step ActCheck labels stratify later target preference.
- Threshold-free AUC shows the result is not only a threshold artifact.
- Pixel-relation quadrants show many failures are schema-correct but action-wrong.
- VISA is not all-stop: normal pass is high, and invalid blind execution drops sharply.
- VISA-Rerank improves action exposure by selecting among schema-conditioned
  candidate actions and deferring when no aligned candidate is available.
- VISA-Rerank action-exposure gains hold on SAT-Bench++ compositional and
  temporal/procedural splits.
- N=100 K=50 short-horizon trajectory probe is useful but not task-level success:
  final/integrated preference improve and wrong-object approach drops in
  all-case; allowed-subset matched analysis improves preference and wrong-object
  AUC, but first-close remains 0.
- A true LIBERO env-step VISA-ClosedLoop pilot was run, but it is quarantined:
  Native OpenVLA and an oracle-target primitive also fail to contact/lift, and
  the setup does not match the old 4/10 LIBERO90 original-task sanity check.
  Do not cite it in rebuttal until the original-task baseline is reproduced in
  the same pipeline and counterfactual predicates are validated.
- The old Native OpenVLA original-task sanity was reproduced on the first four
  LIBERO90 tasks with 2/4 success, confirming that the earlier 4/10 baseline was
  not imaginary and that the N=5 counterfactual all-zero pilot should not be
  interpreted as a method failure.
- The new VISA-ClosedLoop wrapper itself was validated on the known successful
  LIBERO90 tasks 0/3 with 2/2 Native success. However, the spatial movable
  counterfactual candidates have 0/4 Native original-task success, so they are
  not a valid base-policy-competent subset for end-to-end VISA claims.
- A stricter LIBERO90 task 3/4 base-policy-competent pilot was added. It
  validates the wrapper and counterfactual evaluator, but model-free
  VISA-Handoff does not improve execution: 0/2 correct contact and 0/2 correct
  lift. This should remain internal/quarantined, not a rebuttal main result.
- VISA-Residual should not be reported as a positive result. The expanded
  paired run is negative on contact/lift: Native counterfactual gets 8/10
  correct lift, while Generic-Residual gets 6/10, VISA-Residual gets 5/10, and
  Oracle-Residual gets 3/10. The improved target-preference proxy does not
  translate into manipulation success.
- Schema engineering is task-family-level rather than instance-level.

Cannot honestly claim yet:

- Completed external three-annotator validation. The annotation packet and scorer
  exist, and a 300-row invalid/compositional/temporal packet is prepared, but
  filled non-author annotations are not present in the repo.
- Single-prompt VISA-R rewriting improves the native policy. It did not improve
  action sensitivity in this run.
- VISA-R full controller success. The spatial true closed-loop pilot is
  quarantined because the selected tasks are not base-policy competent; the
  handoff-only LIBERO90 task 3/4 pilot validates the setup but does not show a
  positive gain. The expanded VISA-Residual run improves softer trajectory
  proxies but not contact/lift, so it also cannot support full controller
  success or task-level success.
- Linear probes prove causal use by the action decoder. Wording should be
  semantic availability only.

## Rebuttal Text Skeleton

Opening:

> We thank the reviewers for recognizing the importance of semantic-action
> transfer and the usefulness of the fixed-observation protocol. The main
> concerns are about scope, benchmark breadth, one-step behavioral relevance,
> causal interpretation of probes, schema engineering, and artifact clarity. We
> address these with targeted new results and revisions.

Main-track story:

> Our revised claim is sharper: SAT-Bench diagnoses whether recovered instruction
> semantics reach the action boundary; VISA exposes this boundary through
> ALLOW/DEFER/FLAG; and the new experiments test whether the gap generalizes
> beyond object/relation swaps, predicts short-horizon behavior, and can be made
> actionable through schema-conditioned reranking/handoff without replacing the
> low-level controller.

Probe/causality wording:

> We agree that linear probes do not establish causal use by the action decoder.
> We will revise the wording to consistently present probes as semantic
> availability evidence, not causal proof.

Scope wording:

> VISA is not a full robot controller and does not certify grasp feasibility,
> collision safety, or long-horizon success. Its intervention target is action
> exposure: whether a native action should be allowed, deferred, flagged, or
> reconditioned before execution.

## Schema Engineering Table

| Task family | Schema fields | New manual work | Instance-level manual label? |
|---|---|---:|---|
| target-name | target, exists, allowed action | low | no |
| relation | target, anchor, relation | low-mid | no |
| invalid/prohibition | allowed/blocked actions, decision | low | no |
| compositional | target, attribute/object descriptor, relation, anchor | low-mid | no |
| temporal/procedural | first_subgoal, next_symbolic_action, order constraint | mid | no |

Key sentence:

> The schema is task-family-level, not instance-level. We do not hand-write rules
> for each episode; we define a compact field set for each semantic family, and
> the same extraction/checking pipeline is applied to all examples.

## Threshold Calibration

Submission wording:

> For each policy/setting, we compute a 7D normalized action delta
> `|| (a_1 - a_2) / sigma ||_2`, where `sigma` is the per-dimension action
> standard deviation estimated from the processed action index
> (`data/processed/unified_index.parquet`). The binary sensitivity threshold is
> calibrated separately for each policy/setting as the 95th percentile of
> paraphrase-control deltas. A paraphrase control uses the same observation and
> original task semantics, but applies a wording-only rewrite such as
> "pick up" -> "grab" or "put" -> "place"; target-changing deltas are never used
> to set the threshold. Thus the threshold is paired to the same observation
> distribution but disjoint in perturbation type. We also report threshold-free
> target-vs-control AUC, so conclusions do not rely only on one cutoff.

Calibration values:

| Policy/setting | Control N | Threshold source | Threshold | Control mean | Target mean | Target-vs-control AUC | Source |
|---|---:|---|---:|---:|---:|---:|---|
| OpenVLA target-name | 600 | paraphrase-control P95 | 2.285 | 1.108 | 1.268 | 0.573 | `outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/` |
| OpenVLA pixel-relation | 600 | paraphrase-control P95 | 2.695 | 1.237 | 1.454 | 0.583 | `outputs/linguistic_blindness/visual_relation_openvla_action_sensitivity_600/` |
| OpenVLA-LIBERO90 target-name | 600 | paraphrase-control P95 | 2.798 | 1.061 | 1.243 | n/a | `outputs/linguistic_blindness/exp2_second_vla_openvla_libero90_target_swap_600/` |
| BridgeData V2 Octo target/task | 165 | paraphrase-control P95 | 0.645 | 0.220 | 1.912 | 0.929 | `outputs/linguistic_blindness/rebuttal_main_supplements/threshold_free_auc_summary.csv` |

Important correction: do not claim that threshold calibration uses separate
episodes for the main OpenVLA runs. The control distribution is computed on the
same evaluation observation set, but from wording-only paraphrase controls; it
does not use target-changing counterfactual deltas. The threshold-free AUC
tables are included precisely to reduce dependence on this calibrated cutoff.

## Diagnostic Mapping

| Diagnostic | Rules out |
|---|---|
| paraphrase threshold | wording-only variation |
| threshold-free AUC | threshold artifact |
| hidden-state probe | semantics absent from representation |
| random-label/mask/shuffle controls | spurious probe artifact |
| ActCheck attribution | pure grounding failure |
| K=50 rollout | one-step diagnostic irrelevance |
| Octo/OpenVLA-LIBERO90/BridgeData | OpenVLA-only artifact |
| VISA invalid deferral + normal pass-through | all-stop artifact |

## Release Checklist

- SAT-Bench/SAT-Bench++ prompt transformations.
- Gold labels and schema files.
- Policy adapter scripts for OpenVLA/Octo.
- ActCheck implementation and analysis scripts.
- VISA/VISA-R interface scripts.
- Threshold calibration and AUC scripts.
- Rollout scripts and analysis scripts.
- External annotation packet and scoring script.
