# Long-Horizon and Bridge Target-Change Supplements

This document records two requested supplements:

1. A semi-long-horizon result addressing the concern that semantic-action gap is only a first-action artifact.
2. A BridgeData V2 second-domain target/task-change diagnostic, extending Bridge beyond invalid-only inhibition.

## 1. Semi-Long-Horizon Evidence

### Source

Existing short-horizon rollout output:

`outputs/linguistic_blindness/short_horizon_target_approach_50/`

New summary:

`outputs/linguistic_blindness/long_horizon_and_bridge_target_supplement/long_horizon_k10_summary.csv`

Setup:

- Policy: OpenVLA LIBERO-90 finetuned checkpoint.
- Suite: `libero_spatial`.
- Tasks: 1, 3, 4, 5, 6.
- Trials: 5 per task.
- Horizon: K=10 policy steps after simulator warmup.
- Comparison: original instruction rollout vs counterfactual target instruction rollout under the same initial simulator state.
- Metric: distance change to original target and counterfactual target over K steps.

### K=10 Target Approach Summary

| Method | N | CF-target approach ↑ | Original-target approach | Wrong-target preference ↓ | Mean Δdist to CF target ↑ | Mean target preference ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Original instruction | 25 | 0.760 | 0.600 | 0.360 | 0.000184 | 0.000306 |
| Counterfactual instruction | 25 | 0.560 | 0.480 | 0.360 | 0.0000068 | 0.000287 |

Paired comparison:

| Quantity | Value |
|---|---:|
| N pairs | 25 |
| Counterfactual rollout improves CF-target approach over original rollout | 0.360 |
| Mean CF-minus-original improvement toward CF target | -0.000177 |
| Counterfactual rollout wrong-target preference rate | 0.360 |

Interpretation:

- If first-action insensitivity were harmless, the counterfactual instruction rollout should become more directed toward the counterfactual target over the next K steps.
- Instead, the counterfactual rollout improves CF-target approach over the original rollout in only 36% of paired cases.
- Mean improvement toward the counterfactual target is slightly negative, indicating that the counterfactual language change does not reliably correct short-horizon behavior.

Recommended wording:

> To test whether one-step action sensitivity underestimates later correction, we ran a K=10 short-horizon target-approach check on 25 paired LIBERO initial states. Counterfactual instructions improved approach toward the counterfactual target in only 36% of paired rollouts, with a slightly negative mean improvement relative to the original-instruction rollout. Thus the measured semantic-action gap is not merely a first-action artifact in this diagnostic subset.

### Schema-Conditioned Interface / Projection Evidence

Existing closed-loop projection appendix:

`outputs/linguistic_blindness/exp6_closed_loop_target_swap_25_until_close/`

Key result already available:

| Method | Final target distance ↓ | Close-target rate ↑ | Release rate ↑ | Target lift ↑ | Full success |
|---|---:|---:|---:|---:|---:|
| Raw / unguided | 0.320 | low | - | - | 0/25 |
| Schema projection / until-close | 0.071 | 0.960 | 0.960 | 0.800 | 0/25 |

Interpretation:

- This is not claimed as a task-success controller.
- It shows that a schema-conditioned interface can materially alter closed-loop target approach and task phase: the robot is redirected close to the counterfactual target and reaches partial interaction/lift more often.
- Full long-horizon manipulation success remains unsolved and should stay in appendix.

Recommended wording:

> As an appendix stress test, schema-conditioned action projection substantially reduces final target distance and increases close-target / lift behavior, but does not solve full LIBERO task completion. We therefore treat it as causal evidence that schema-level target semantics can redirect the action interface, not as a new manipulation controller.

## 2. BridgeData V2 Target/Task-Change Diagnostic

### Source

Benchmark:

`outputs/linguistic_blindness/bridge_v2_target_change_165/`

Octo output:

`outputs/linguistic_blindness/bridge_v2_octo_target_change_165/`

Summary:

`outputs/linguistic_blindness/bridge_v2_target_change_analysis/bridge_target_change_summary.csv`

Construction:

- Dataset: BridgeData V2 real-robot frames, first public TFDS shard.
- Fixed observation frame.
- Original instruction: the original Bridge language command for that frame.
- Counterfactual instruction: a different Bridge instruction from the same shard with a different extracted primary target/task phrase.
- Policy: Octo-small-1.5.
- Threshold: Octo-specific paraphrase-control P95 on this Bridge target-change subset.

Caveat:

The local Bridge TFDS shard does not provide reliable object metadata, so this is an instruction target/task-change diagnostic rather than a verified object-presence target swap benchmark. We should not overclaim it as full pixel-grounded Bridge target semantics.

### Result

| Dataset | Policy | Split | N | Oracle instruction target change | Action sensitivity ↑ | SAG ↓ | Low sensitivity ↓ | Threshold |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BridgeData V2 | Octo-small-1.5 | target/task change | 165 | 1.000 | 0.661 | 0.339 | 0.339 | 0.645 |

Interpretation:

- Bridge target/task-change action sensitivity is higher than the severe OpenVLA/LIBERO target-swap result, which is useful because the metric can distinguish policies/domains.
- The gap is reduced but not eliminated: about 34% of fixed-observation real-robot Bridge examples remain below the paraphrase-calibrated target-change threshold.
- This extends the second domain from invalid-only inhibition to a target/task-language-change diagnostic.

Recommended wording:

> To avoid limiting the second-domain evidence to invalid instructions, we constructed a BridgeData V2 target/task-change diagnostic by pairing each fixed real-robot frame with its original command and a counterfactual command drawn from another Bridge task with a different primary target phrase. Octo shows substantially higher sensitivity on this Bridge split than OpenVLA on LIBERO, but still leaves a 0.339 instruction-target/action gap under a policy-specific paraphrase threshold. This suggests semantic-action gap is a graded behavioral axis rather than an OpenVLA-only failure mode.

## Overall Takeaway

These two supplements address two reviewer risks:

1. The semantic-action gap is not only a first-step artifact: short-horizon rollouts do not reliably self-correct toward the counterfactual target, and schema-conditioned projection materially redirects target approach.
2. BridgeData V2 is no longer invalid-only: it now includes a real-robot target/task-change diagnostic with Octo, showing reduced but nonzero action-interface gap outside LIBERO.
