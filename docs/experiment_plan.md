# Experiment Plan

## Active Assets

- Processed data: `data/processed/unified_index.parquet`, `schema_labels.parquet`, `instr_wrong.parquet`.
- Current benchmark/eval outputs: `outputs/linguistic_blindness/`.
- Qwen schema-probe runner: `scripts/lb_run_qwen_schema.py` and `slurm/lb_qwen_schema_eval.sbatch`.
- Semantic-action paired analyzer: `src/linguistic_blindness/evaluation/semantic_action_gap.py`.

Old handover/robotics artifacts are not part of the active paper pipeline.

## MVP Pipeline

1. Build counterfactual benchmark:

```bash
scripts/lb_build_benchmark.sh --max-observations 1000 --split val --out-dir outputs/linguistic_blindness/benchmark_val_1k
```

2. Run parser/checker/gate evaluation:

```bash
scripts/lb_run_eval.sh --benchmark outputs/linguistic_blindness/benchmark_val_1k/counterfactual_examples.jsonl --out-dir outputs/linguistic_blindness/eval_val_1k
```

3. Run ablation-oriented subset:

```bash
scripts/lb_run_ablation.sh
```

4. Run Qwen2.5-VL schema probe:

```bash
sbatch slurm/lb_qwen_schema_eval.sbatch
```

5. Run semantic-action gap analysis after both schema and action predictions exist:

```bash
scripts/lb_semantic_action_gap.sh <schema_predictions.jsonl> <action_predictions.jsonl> outputs/linguistic_blindness/semantic_action_gap/<run_name>
```

## Methods

- `VLM schema probe`: Qwen2.5-VL or another VLM outputs the explicit interaction schema.
- `VLA/action-head output`: model produces an action label, symbolic action, or converted action-target pair.
- `structured_parser`: deterministic parser using instruction text and replay state.
- `parser_checker`: structured parser plus deterministic checker flags.
- `parser_checker_gate`: parser plus checker plus safe action gate.
- `blind_structured_parser`: diagnostic baseline that intentionally uses the original instruction under counterfactual changes.

Unavailable VLM/VLA methods must be marked unavailable, not fabricated.

## Experiment 1: Counterfactual Interaction Schema Benchmark

Purpose: build a systematic offline suite where observation is fixed and instruction is counterfactually changed.

Input:

- processed observations from `data/processed/`
- original instruction
- target/phase labels
- optional wrong-instruction or target-swap metadata

Perturbation types:

- `target_swap`: target should change.
- `blank_instruction`: agent should defer, not execute.
- `impossible_instruction`: nonexistent target should be rejected.
- `safety_conflict`: unsafe state should block unsafe actions.
- `negation`: prohibited action/object should be blocked.
- `phase_conflict`: predicted phase should match observable state.

Outputs:

- `counterfactual_examples.jsonl`
- `benchmark_statistics.json/csv`

Table:

- Benchmark Statistics: domain/split, observations, original instructions, counterfactual instructions, schema labels, perturbation types.

## Experiment 2: VLM Semantic Probe vs Action Head

Purpose: test the revised attribution claim: VLM semantics may be correct while action output is wrong.

Input:

- same `example_id`
- same observation
- same counterfactual instruction
- one VLM schema-probe prediction
- one VLA/action-head prediction

Outputs compared:

- VLM schema probe: target, phase, allowed actions, blocked actions, next symbolic action.
- Action head: action label, action target if available, or schema-converted action output.

Case taxonomy:

| VLM Schema | Action Head | Interpretation |
|---|---|---|
| correct | correct | normal |
| wrong | wrong | VLM semantic failure |
| correct | wrong | semantic-to-action transfer failure |
| wrong | correct | possible visual shortcut or label bias |

Metrics:

- VLM Schema Sensitivity
- Action Sensitivity
- Semantic-Action Sensitivity Gap: `SASG = VLM Schema Sensitivity - Action Sensitivity`
- count/percentage of semantic-to-action transfer failures

Expected trend:

- VLM Schema Sensitivity is higher than Action Sensitivity for action-head baselines.
- Semantic-to-action transfer failure is a substantial failure category.

Output files:

- `semantic_action_pairs.jsonl`
- `semantic_action_gap.json`
- `semantic_action_taxonomy.csv`
- `semantic_action_by_perturbation.csv`
- `semantic_action_cases.json`

## Experiment 3: Linguistic Blindness Under Counterfactual Instructions

Purpose: show that fixed-observation instruction changes expose instruction blindness.

Methods:

- VLM schema probe
- VLA/action-head baseline
- structured parser
- parser + checker
- parser + checker + gate
- blind structured parser

Metrics:

- Target Sensitivity
- Blank Non-Execution
- Impossible Rejection
- Negation Consistency
- Phase-State Consistency
- Safety Violation
- Blind Execution

Expected trend:

- Action-head baselines have lower counterfactual/action sensitivity.
- Schema/checker/gate variants reduce unsafe and blind execution.

## Experiment 4: Task Success vs Counterfactual Sensitivity

Purpose: show that normal task/action success overestimates instruction following.

Metrics:

- Normal Instruction Accuracy
- Counterfactual Sensitivity
- Instruction Sensitivity Gap: `ISG = Normal Instruction Accuracy - Counterfactual Sensitivity`

Expected trend:

- Methods can look good on normal instructions but fail under counterfactual instructions.
- A lower ISG means better language-faithful behavior.

## Experiment 5: Schema / Checker / Gate Mitigation

Purpose: show that VISA is not only diagnostic; checker/gate reduce invalid or unsafe recommendations.

Variants:

- schema only
- schema + checker
- schema + checker + gate

Metrics:

- Schema Validity
- Action Constraint Consistency
- Blocked Action Violation
- Safety Violation
- Safe Deferral
- Blind Execution

Expected trend:

- Checker exposes contradictions.
- Gate reduces unsafe actions and blind execution by replacing risky actions with safe deferrals.

## Experiment 6: Ablation Study

Variants:

- Full Method
- w/o Checker
- w/o Gate
- w/o Blocked Actions
- w/o Phase Field
- w/o Counterfactual Data
- Direct JSON Prompt Only

Metrics:

- Target Sensitivity
- Schema Validity
- Safety Violation
- Blind Execution
- Safe Deferral

Expected trend:

- Removing checker increases undetected inconsistency.
- Removing gate increases unsafe/blind execution.
- Removing blocked actions hurts negation/prohibition handling.
- Removing phase hurts phase-state consistency.
- Removing counterfactual data hurts target sensitivity.

## Experiment 7: Replay-Based Safety Evaluation

Purpose: connect the schema/checker/gate to embodied safety without requiring full simulation rollout.

Input:

- replay states with contact/release/phase information
- model schema or action recommendation

Violation types:

- Premature Retract
- Blank Execution
- Target-not-found Execution
- Negation Violation
- Phase-State Conflict
- Blocked Action Violation

Metrics:

- Premature Retract rate
- Blank Execution rate
- Target-not-found Execution rate
- Negation Violation rate
- Overall Safety Violation rate

Expected trend:

- Parser + checker + gate should reduce safety violations and increase safe deferrals.

## Experiment 8: Failure Taxonomy and Qualitative Cases

Purpose: make the failure modes interpretable and show systematic patterns.

Categories:

- Target Fixation
- Default Execution
- Hallucinated Target
- Negation Failure
- Premature Retract
- Phase Confusion
- Constraint Contradiction
- Semantic-to-Action Transfer Failure

Outputs:

- `failure_taxonomy.json/csv`
- `qualitative_cases.json`
- `semantic_action_cases.json`

## Experiment 9: Optional Small Rollout Sanity Check

Purpose: small grounding sanity check only, not the main contribution.

Setup:

- 10-30 scenes
- 2-3 instruction variants per scene
- normal, target swap, and safety conflict cases

Metrics:

- Task success
- Unsafe action rate
- Safe abort/deferral rate
- Timeout
- Gate intervention count

Interpretation:

The goal is not to maximize aggressive completion. The goal is to show that schema gating avoids blind or unsafe execution under language conflict.

## Main Tables

- Table 1: Benchmark Statistics.
- Table 2: Semantic-Action Gap.
- Table 3: Linguistic Blindness Main Results.
- Table 4: Task Success vs Counterfactual Sensitivity.
- Table 5: Verification and Mitigation.
- Table 6: Ablation Study.
- Table 7: Replay Safety.
- Table 8: Failure Taxonomy.

## Reproducibility

Each run should save:

- predictions or paired predictions
- JSON summaries
- CSV summaries
- markdown/LaTeX tables
- qualitative cases
- run metadata with command, timestamp, benchmark path, and method availability
