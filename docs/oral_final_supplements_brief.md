# Oral Supplement Brief

Use this as a compact checklist for the paper.

## Threshold-Free Action Sensitivity

- OpenVLA target-name target-vs-control AUC: 0.573.
- OpenVLA pixel-relation target-vs-control AUC: 0.583.
- BridgeData V2 Octo target/task target-vs-control AUC: 0.929.

Conclusion: OpenVLA target-change deltas are weakly separated from paraphrase controls; Bridge/Octo shows the metric can distinguish a more responsive setting.

## Schema vs Action Failure Attribution

OpenVLA pixel-grounded relation:

- Schema correct + action wrong: 264/600 = 44.0%.
- Schema wrong + action wrong: 9/600 = 1.5%.

Conclusion: many failures are not schema/perception failures; they are action-interface failures.

## Normal Utility

- Overall normal pass: 94.0%.
- Spatial relation normal pass: 100.0%.
- Drawer/cabinet normal pass: 72.7%.

Conclusion: VISA is not all-stop; residual false blocks are concentrated in drawer/cabinet commands.

## Policy / Domain Matrix

- OpenVLA base target-name SAG: 0.932.
- OpenVLA LIBERO-90 target-name SAG: 0.943.
- Octo LIBERO target-name SAG: 0.573.
- OpenVLA pixel-relation SAG: 0.882.
- Octo pixel-relation SAG: 0.583.
- BridgeData V2 Octo target/task SAG: 0.339.

Conclusion: semantic-action gap is graded across policies/domains, not OpenVLA-only.

## K=20 Long-Horizon Check

- Counterfactual rollout CF-target approach rate: 0.560.
- Original-instruction rollout CF-target approach rate: 0.840.
- Mean CF-minus-original approach to CF target: -0.001269.

Conclusion: counterfactual rollouts do not self-correct over 20 steps; the issue is not just first-action artifact.

## Negation Stress

- Octo blind execution: 70.3%.
- Direct Qwen Gate safe deferral: 36.5%.
- VISA safe deferral: 100.0%.

By variant:

- `do_not`: Octo inhibited 15.0%, Direct Gate defers 0.0%, VISA defers 100.0%.
- `leave_alone`: Octo inhibited 68.5%, Direct Gate defers 0.0%, VISA defers 100.0%.

Conclusion: negation is a strong NLP-style stress test; schema-gated VISA handles it much more reliably than direct gate and native action output.
