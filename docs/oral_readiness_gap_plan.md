# Oral-Readiness Gap Plan

This note records what has been added after Exp5/Exp6 and what still separates the current paper from an oral-level submission. It is intentionally conservative: no unavailable model or domain result is fabricated.

## Added Now

### 1. Bootstrap Confidence Intervals

Generated with:

```bash
scripts/lb_run_oral_readiness_additions.py
```

Outputs:

- `outputs/linguistic_blindness/oral_readiness_additions/bootstrap_ci_main_metrics.csv`
- `outputs/linguistic_blindness/oral_readiness_additions/bootstrap_ci_main_metrics.json`
- `outputs/linguistic_blindness/oral_readiness_additions/latex_table_main_ci.tex`

Main numbers:

| Experiment | Metric | Mean [95% CI] |
|---|---:|---:|
| Exp2 | VLM schema sensitivity | 1.000 [1.000, 1.000] |
| Exp2 | OpenVLA action sensitivity | 0.068 [0.048, 0.088] |
| Exp2 | semantic-action gap case rate | 0.932 [0.910, 0.952] |
| Exp3 | OpenVLA blind execution | 0.927 [0.915, 0.939] |
| Exp4 | gated blind execution | 0.028 [0.020, 0.036] |
| Exp4 | safe deferral | 0.972 [0.963, 0.979] |
| Exp4 | normal pass | 0.940 [0.920, 0.958] |
| Exp4 | false block | 0.060 [0.042, 0.080] |

Interpretation: the main effects are not borderline. The gap between schema sensitivity and action sensitivity is large, and the checker/gate mitigation remains strong under bootstrap resampling.

### 2. Stronger Baseline Status Table

Output:

- `outputs/linguistic_blindness/oral_readiness_additions/stronger_baseline_status.csv`

Completed baselines:

- OpenVLA raw
- All-stop baseline
- Schema monitor only
- Checker without gate
- Checker + Gate v3
- Prompt rewrite OpenVLA negative result
- Domain-aware schema prompt v3 ablation

Recommended next baselines:

- VLM-as-critic gate
- Target-position heuristic gate

Why these matter:

- All-stop is useful but weak; it proves the gate is not simply stopping everything because normal pass would be zero.
- Prompt rewrite is useful as a negative result; it shows ordinary language rewriting does not close the action-head gap.
- VLM-as-critic and target-position heuristic are stronger because they are plausible alternative interfaces.

### 3. Second Model / Second Domain Plan

Output:

- `outputs/linguistic_blindness/oral_readiness_additions/second_model_domain_plan.csv`

Priority order:

1. Second schema probe on the existing 600 LIBERO examples.
2. Second domain/split, preferably 100-200 LIBERO-10 or LIBERO-90 held-out examples.
3. Second VLA/action policy if a released checkpoint is locally available.

Minimal acceptable version:

- Run another schema probe or VLM critic on 100-200 examples.
- Reuse existing OpenVLA actions when possible.
- Report whether schema-level sensitivity/rejection is robust beyond Qwen2.5-VL-7B.

## What Still Separates This From Oral-Level Evidence

### Must Add 1: Second Model or Second Domain

Current claim is strongest as:

> OpenVLA on the LIBERO counterfactual setting exhibits a semantic-action gap.

After a second model/domain, the claim can become:

> Semantic-action gaps appear across VLA settings or schema probes.

This is the highest-priority missing piece.

### Must Add 2: Stronger Baseline

Best next baseline:

- VLM-as-critic gate: ask a VLM whether executing the current action is allowed, but do not expose the full schema fields.

Second-best baseline:

- Target-position heuristic gate: use object positions and action direction to block target-inconsistent actions.

These baselines answer whether the full Verifiable Interaction Schema is necessary.

### Must Add 3: Confidence Intervals

This is now implemented. Use the generated LaTeX table in the paper's main experiments or appendix.

## Recommended Next Command

If a second schema model is available locally, run the schema probe on a 100-200 example subset first. If not, implement the VLM-as-critic gate because it reuses Qwen and existing data.
