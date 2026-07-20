# Oral Remaining Execution Plan

This file translates the current P0/P1/P2 checklist into concrete actions based on the experiments already completed in this repository.

## Current State

The strongest completed evidence is:

- OpenVLA base target-name swap: schema sensitivity 1.000, action sensitivity 0.068, SAG 0.932.
- OpenVLA LIBERO-90 target-name swap: action sensitivity 0.057, SAG 0.943.
- Octo-small-1.5 target-name swap: action sensitivity 0.427, SAG 0.573.
- OpenVLA pixel-grounded relation split: Qwen grounding 0.958; on grounding-correct examples, action sensitivity 0.077, SAG 0.923.
- Octo pixel-grounded relation split: on grounding-correct examples, action sensitivity 0.350, SAG 0.650.
- Exp3 invalid instructions: OpenVLA blind execution 0.927.
- Exp4 VISA v3: invalid blind execution reduced to 0.028, normal pass 0.940.
- Direct Qwen Gate vs VISA: Direct preserves normal better, VISA has much better invalid safety, Hybrid is safest but more conservative.
- VISA-ActCheck: flags wrong-target or ambiguous native actions rather than only measuring action delta.
- Short-horizon K=10 check: counterfactual instruction does not produce stronger approach toward the counterfactual target.
- Closed-loop projection appendix: improves target redirection, but full task success remains 0/25, so it should stay appendix.

## P0: Direct Oral-Level Risks

### 1. Policy / Benchmark Coverage

Status: partially satisfied.

Completed:

- OpenVLA base.
- OpenVLA LIBERO-90 finetuned.
- Octo-small-1.5.
- LIBERO target-name and relation-defined splits.
- Small handover safety schema domain.

Remaining risk:

- Still mostly LIBERO-style tabletop manipulation.
- Octo is a different VLA architecture, but the benchmark domain remains close to LIBERO.

Recommended next action:

- Do not keep adding more OpenVLA checkpoints unless easy.
- If a second non-LIBERO manipulation dataset is available locally, add a 100--300 example diagnostic subset.
- If no second dataset is available, frame the current result as "cross-policy within manipulation" rather than "universal VLA failure."

Minimum paper-safe claim:

> Semantic-action sensitivity varies across policies: Octo is more sensitive than OpenVLA, but a substantial gap remains on relation-defined targets. We therefore treat the gap as a measurable evaluation axis rather than claiming all policies fail equally.

### 2. Closed-Loop / Long-Horizon Evidence

Status: partially satisfied.

Completed:

- Short-horizon K=10 target approach check.
- Appendix closed-loop projection redirection.

Remaining risk:

- Full LIBERO task success is still 0/25 for projection.
- This cannot be sold as controller improvement.

Recommended next action:

- Keep closed-loop in appendix.
- Use short-horizon result in main paper only as evidence against the "one-step artifact" objection.
- Do not prioritize full success unless a clean controller adapter is already available.

Paper-safe claim:

> The short-horizon diagnostic shows that the lack of counterfactual sensitivity is not confined to a single action step. The projection appendix further suggests schema-level targets can redirect motion, but full manipulation success remains outside the scope of this paper.

## P1: Method Contribution Risks

### 3. VISA as More Than a Gate

Status: substantially improved.

Completed:

- VISA checker/gate for invalid instruction mitigation.
- VISA-ActCheck for semantic-action consistency.
- Relation-defined action consistency analysis.
- Schema field ablation.

Remaining risk:

- VISA does not synthesize high-quality low-level actions.
- Projection helps target redirection but not full success.

Recommended next action:

- Present VISA as an interface-level verifier, not a controller.
- Put ActCheck in the method contribution: it checks whether native continuous actions are semantically consistent with the schema target.
- Keep projection as optional downstream use, not the central method.

Paper-safe claim:

> VISA externalizes action-relevant semantics and verifies whether native action recommendations are consistent with them. It can defer unsafe/invalid actions and flag wrong-target actions; it is not intended to replace low-level control.

### 4. Non-Template Invalid Instructions

Status: not fully satisfied.

Completed:

- 300 hard invalid programmatic split.
- Direct Qwen Gate / VISA / Hybrid comparison.
- Human-written annotation packet prepared.

Remaining risk:

- No completed human-written invalid labels yet.
- Hard invalid split is still category-based.

Recommended next action:

- Highest-value remaining data task: fill the human-written invalid sheet with 300--500 approved examples.
- Run Direct Qwen Gate, VISA, and Hybrid on that set.
- Report agreement or at minimum double-review approval.

Do not claim:

- Do not call the prepared annotation sheet "human-written" until the `human_instruction`, `gold_decision`, and review fields are filled.

## P2: Reviewer Safety

### 5. Strong Baselines and Ablations

Status: mostly satisfied.

Completed:

- All-stop baseline.
- Direct Qwen Gate.
- Hybrid Direct+VISA.
- Schema field ablation.
- Threshold robustness.
- Qwen 3B schema probe.
- Hidden-state probe ablations.

Remaining optional additions:

- Direct Qwen Gate with a second prompt profile.
- More confidence intervals for newer Octo relation / short-horizon tables.

Recommended next action:

- Add bootstrap CI only for final main tables if time permits.
- Do not spend more time improving prompt variants unless reviewer-critical.

### 6. Target-Consistency Analysis

Status: substantially satisfied.

Completed:

- ActCheck for target-name swap.
- ActCheck for relation-defined swap.
- Octo relation ActCheck.
- Pair and angle breakdowns.

Remaining optional additions:

- Visual trajectory figures.
- Failure-case panels with image, instruction, schema target, action vector, and ActCheck flag.

Recommended next action:

- Generate 4--6 qualitative case panels for the paper or appendix.
- Prioritize cases where Qwen relation grounding is correct but OpenVLA or Octo action is wrong-target or ambiguous.

## Recommended Immediate Order

1. Finish the human-written invalid set.
2. Produce 4--6 qualitative ActCheck/trajectory figures.
3. Add bootstrap CI for Octo relation and ActCheck if these enter the main table.
4. Only if a non-LIBERO dataset is readily available, add a 100--300 example second-domain diagnostic subset.

## What Not To Do

- Do not keep optimizing projection until full success unless the goal is a robotics-control paper.
- Do not claim VISA solves manipulation.
- Do not claim universal VLA failure; write "semantic-action sensitivity is a measurable axis, with severity varying by policy."
- Do not report the human-written invalid set before annotation is complete.

