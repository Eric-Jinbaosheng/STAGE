# Oral-Focused Supplement Status

Generated outputs:

`outputs/linguistic_blindness/oral_final_supplements/`

## Completed Without New Model Runs

### 1. Threshold-Free Action Sensitivity

Files:

- `threshold_free_curve.csv`
- `action_delta_distribution_summary.csv`
- `threshold_free_auc_summary.csv`

Key values:

| Setting | N | Mean target delta | Mean control delta | Target-vs-control AUC |
|---|---:|---:|---:|---:|
| OpenVLA target-name swap | 600 | 1.268 | 1.108 | 0.573 |
| OpenVLA pixel-relation swap | 600 | 1.454 | 1.237 | 0.583 |
| BridgeData V2 Octo target/task change | 165 | 1.912 | 0.220 | 0.929 |

Interpretation:

- OpenVLA target-change deltas are only weakly separated from paraphrase controls, so the low sensitivity result is not just one arbitrary threshold.
- Bridge/Octo has strong target-vs-control separation, showing the metric can distinguish settings and is not designed to make every policy look bad.

### 2. Schema Quality vs Action Failure Quadrants

Files:

- `schema_action_quadrant_summary.csv`
- `schema_action_quadrant_predictions.jsonl`

OpenVLA pixel-grounded relation split:

| Case type | Count | Rate |
|---|---:|---:|
| Schema correct + action consistent | 311 | 0.518 |
| Schema correct + action wrong | 264 | 0.440 |
| Schema wrong + action consistent | 16 | 0.027 |
| Schema wrong + action wrong | 9 | 0.015 |

Main conclusion:

Among relation examples, 44.0% are schema-correct but action-wrong. This directly supports the action-interface bottleneck interpretation: many failures are not caused by schema grounding failure.

### 3. Normal Command Utility Breakdown

Files:

- `normal_utility_breakdown.csv`
- `normal_utility_breakdown_predictions.jsonl`

| Command type | N | Normal pass | False block | Target valid |
|---|---:|---:|---:|---:|
| Overall | 600 | 0.940 | 0.060 | 1.000 |
| Spatial relation | 468 | 1.000 | 0.000 | 1.000 |
| Drawer/cabinet | 132 | 0.727 | 0.273 | 1.000 |

Main conclusion:

VISA is not an all-stop rule: it preserves all spatial-relation normal commands in this split. Remaining normal false blocks concentrate in drawer/cabinet commands, which should be discussed as a limitation / domain-specific schema-policy issue.

### 4. Policy / Domain Matrix

File:

- `policy_domain_matrix.csv`

Key rows:

| Policy | Setting | N | Schema/oracle sens | Action sens | SAG |
|---|---|---:|---:|---:|---:|
| OpenVLA base | LIBERO target-name | 600 | 1.000 | 0.068 | 0.932 |
| OpenVLA LIBERO-90 | LIBERO target-name | 600 | 1.000 | 0.057 | 0.943 |
| Octo-small-1.5 | LIBERO target-name | 600 | 1.000 | 0.427 | 0.573 |
| OpenVLA base | LIBERO pixel relation | 600 | 0.958 | 0.077 | 0.882 |
| Octo-small-1.5 | LIBERO pixel relation | 600 | 0.958 | 0.375 | 0.583 |
| Octo-small-1.5 | Bridge target/task change | 165 | 1.000 | 0.661 | 0.339 |

Conclusion:

The gap is graded across policies/domains. Octo and Bridge are less severe than OpenVLA/LIBERO, but the action-interface gap remains nonzero.

### 5. Causal Interface / Action Repair Proxy

Files:

- `schema_conditioned_action_repair_summary.csv`
- `schema_conditioned_action_repair_predictions.jsonl`

| Method | N | Target-aligned rate | Wrong/ambiguous | Intervention rate |
|---|---:|---:|---:|---:|
| Native OpenVLA action | 600 | 0.545 | 0.455 | 0.000 |
| VISA-ActCheck directional candidate | 600 | 1.000 | 0.000 | 0.455 |

Caveat:

This is a post-hoc candidate-repair proxy, not a learned controller or full rollout. It shows that schema target information can causally define a target-consistent action candidate when native action fails ActCheck.

### 6. Negation Stress Benchmark Prepared

File:

- `negation_stress_benchmark.jsonl`

Stats:

- 200 base normal examples
- 4 negation/prohibition variants per base
- 800 examples total

Variants:

- `do_not`
- `avoid`
- `leave_alone`
- `except`

Model runs were not submitted because Slurm controller was temporarily unreachable.

### 7. Human-Written Invalid Annotation Packet Prepared

Files:

- `human_invalid_annotation_template.csv`
- `human_invalid_annotation_guidelines.md`
- Scoring script: `scripts/lb_score_human_invalid_annotations.py`

Status:

This still requires real annotators. The package is ready, but labels/agreement cannot be fabricated.

## Pending New Runs

Slurm submission failed with:

`Unable to contact slurm controller (connect failure)`

Ready-to-run commands once Slurm recovers:

```bash
OUT_DIR=outputs/linguistic_blindness/short_horizon_target_approach_k20 \
TASK_IDS=1,3,4,5,6 NUM_TRIALS=5 HORIZON=20 \
sbatch slurm/lb_short_horizon_target_approach.sbatch
```

```bash
BENCHMARK=outputs/linguistic_blindness/oral_final_supplements/negation_stress_benchmark.jsonl \
OUT_DIR=outputs/linguistic_blindness/negation_stress_octo_inhibition_800 \
MAX_PER_PERTURBATION=0 \
sbatch slurm/lb_octo_invalid_inhibition.sbatch
```

```bash
BENCHMARK=outputs/linguistic_blindness/oral_final_supplements/negation_stress_benchmark.jsonl \
OUT_DIR=outputs/linguistic_blindness/negation_stress_direct_qwen_gate_800 \
MAX_EXAMPLES=0 \
sbatch slurm/lb_direct_qwen_gate.sbatch
```

```bash
BENCHMARK=outputs/linguistic_blindness/oral_final_supplements/negation_stress_benchmark.jsonl \
OUT_DIR=outputs/linguistic_blindness/negation_stress_qwen_schema_800 \
MAX_EXAMPLES=0 USE_GATE=1 LOG_EVERY=50 SAVE_EVERY=100 \
sbatch slurm/lb_qwen_schema_eval.sbatch
```

## Completed New Runs After External Slurm Submission

### 8. K=20 Short-Horizon Target Approach

Output:

`outputs/linguistic_blindness/short_horizon_target_approach_k20/`

| Method | N | CF-target approach | Original-target approach | Mean Δdist to CF target | Mean target preference |
|---|---:|---:|---:|---:|---:|
| Original instruction | 25 | 0.840 | 0.640 | 0.000730 | 0.000296 |
| Counterfactual instruction | 25 | 0.560 | 0.480 | -0.000539 | 0.000530 |

Paired result:

| Metric | Value |
|---|---:|
| N pairs | 25 |
| Mean CF-minus-original counterfactual approach | -0.001269 |

Interpretation:

At K=20, counterfactual rollouts still do not self-correct toward the counterfactual target. The counterfactual instruction produces lower mean approach toward the counterfactual target than the original-instruction rollout.

### 9. Negation-Specific Stress Test

Benchmark:

`outputs/linguistic_blindness/oral_final_supplements/negation_stress_benchmark.jsonl`

Outputs:

- Octo: `outputs/linguistic_blindness/negation_stress_octo_inhibition_800/`
- Direct Gate: `outputs/linguistic_blindness/negation_stress_direct_qwen_gate_800/`
- VISA schema gate combined: `outputs/linguistic_blindness/negation_stress_qwen_schema_combined_800/`

Overall:

| Method | N | Safe/Inhibited ↑ | Blind/Allow ↓ |
|---|---:|---:|---:|
| Octo action inhibition | 800 | 0.298 | 0.703 blind execution |
| Direct Qwen Gate | 800 | 0.365 safe deferral | 0.635 allow |
| VISA / Qwen Schema Gate | 800 | 1.000 safe deferral | 0.000 allow |

By variant:

| Variant | Octo inhibited ↑ | Direct safe deferral ↑ | VISA safe deferral ↑ |
|---|---:|---:|---:|
| avoid | 0.205 | 1.000 | 1.000 |
| do_not | 0.150 | 0.000 | 1.000 |
| except | 0.150 | 0.460 | 1.000 |
| leave_alone | 0.685 | 0.000 | 1.000 |

Interpretation:

Negation/prohibition is a strong language-specific stressor. Octo often remains close to the original action under prohibition, and Direct Qwen Gate is brittle: it handles `avoid` but fails on `do_not` and `leave_alone`. VISA's schema-gated interface consistently defers all variants.
