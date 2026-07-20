# Paper Tables

Generated run artifacts are written under `outputs/linguistic_blindness/`.

Common files:

- `benchmark_statistics.json/csv`
- `predictions.jsonl`
- `main_results.json/csv`
- `ablation_results.json/csv`
- `failure_taxonomy.json/csv`
- `qualitative_cases.json`
- `paper_tables.md`
- `latex_tables.tex`

## Table 1: Benchmark Statistics

Columns:

- Split / domain
- # observations
- # original instructions
- # counterfactual instructions
- # schema labels
- perturbation types

Purpose: show the evaluation suite is systematic, not anecdotal.

## Table 2: Semantic-Action Gap

Generated with:

```bash
scripts/lb_semantic_action_gap.sh <schema_predictions.jsonl> <action_predictions.jsonl> <out_dir>
```

Columns:

- Method / model
- VLM Target Accuracy ↑
- VLM Schema Sensitivity ↑
- Action Sensitivity ↑
- Semantic-Action Sensitivity Gap ↓
- Semantic-to-Action Transfer Failure ↓

Purpose: support the central attribution claim that VLM-level semantics can be correct while action output is wrong.

## Table 3: Linguistic Blindness Main Results

Rows:

- Direct VLM Schema
- Prompted VLM Schema
- VLA Action Head
- Structured Parser
- Parser + Checker
- Parser + Checker + Gate

Columns:

- Target Sensitivity ↑
- Blank Non-Execution ↑
- Impossible Rejection ↑
- Negation Consistency ↑
- Phase-State Consistency ↑
- Safety Violation ↓
- Blind Execution ↓

Purpose: show counterfactual instructions expose agent-level linguistic blindness.

## Table 4: Task Success vs Counterfactual Sensitivity

Columns:

- Method
- Normal Instruction Accuracy ↑
- Counterfactual Sensitivity ↑
- Instruction Sensitivity Gap ↓

Purpose: show standard task/action success can overestimate instruction following.

## Table 5: Verification and Mitigation

Rows:

- Schema Only
- Schema + Checker
- Schema + Checker + Gate

Columns:

- Schema Validity ↑
- Action Constraint Consistency ↑
- Blocked Action Violation ↓
- Safety Violation ↓
- Safe Deferral ↑
- Blind Execution ↓

Purpose: show checker/gate reduce invalid, blind, or unsafe action recommendations.

## Table 6: Ablation Study

Rows:

- Full Method
- w/o Checker
- w/o Gate
- w/o Blocked Actions
- w/o Phase Field
- w/o Counterfactual Data
- Direct JSON Prompt Only

Columns:

- Target Sensitivity ↑
- Schema Validity ↑
- Safety Violation ↓
- Blind Execution ↓
- Safe Deferral ↑

Purpose: show which schema/checker/gate components matter.

## Table 7: Replay Safety

Columns:

- Method
- Premature Retract ↓
- Blank Execution ↓
- Target-not-found Execution ↓
- Negation Violation ↓
- Overall Safety Violation ↓

Purpose: connect schema checking to embodied safety without requiring full rollout.

## Table 8: Failure Taxonomy

Columns:

- Failure Type
- Count
- Percentage
- Representative Example

Rows:

- Target Fixation
- Default Execution
- Hallucinated Target
- Negation Failure
- Premature Retract
- Phase Confusion
- Constraint Contradiction
- Semantic-to-Action Transfer Failure

Purpose: make the failures interpretable and support qualitative analysis.

## Reporting Rule

If a benchmark or run is marked `smoke_test`, do not report those numbers as final experimental results.
