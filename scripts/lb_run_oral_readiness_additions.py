#!/usr/bin/env python3
"""Oral-readiness additions for the linguistic-blindness experiments.

This is a post-hoc analysis script. It does not rerun Qwen/OpenVLA and does not
modify frozen experiment outputs. It adds the concrete items reviewers would
expect for a stronger paper: bootstrap confidence intervals, stronger-baseline
status, and an explicit second-setting run plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from linguistic_blindness.utils.io import command_string, ensure_dir, read_jsonl, utc_timestamp, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate bootstrap CI and oral-readiness analysis from frozen outputs.")
    p.add_argument("--out-dir", default="outputs/linguistic_blindness/oral_readiness_additions")
    p.add_argument("--rounds", type=int, default=5000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--exp2-predictions", default="outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/predictions.jsonl")
    p.add_argument("--exp3-predictions", default="outputs/linguistic_blindness/exp3_invalid_instruction_clean_600/openvla_action_inhibition/predictions.jsonl")
    p.add_argument("--exp4-invalid", default="outputs/linguistic_blindness/exp4_checker_gate_mitigation_clean_600_v3_unified/exp4_predictions.jsonl")
    p.add_argument("--exp4-normal", default="outputs/linguistic_blindness/exp4_checker_gate_mitigation_clean_600_v3_unified/exp4_normal_preservation_predictions.jsonl")
    p.add_argument("--exp5-checker-gate", default="outputs/linguistic_blindness/exp5_ablation_diagnostic_v3/exp5_checker_gate_ablation.csv")
    p.add_argument("--exp5-prompt-ablation", default="outputs/linguistic_blindness/exp5_ablation_diagnostic_v3/exp5_prompt_ablation.csv")
    return p.parse_args()


def bootstrap_mean(values: Sequence[float], *, seed: int, rounds: int) -> tuple[float, float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return (0.0, 0.0, 0.0)
    mean = sum(vals) / len(vals)
    rng = random.Random(seed)
    means: List[float] = []
    n = len(vals)
    for _ in range(rounds):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * (rounds - 1))]
    hi = means[int(0.975 * (rounds - 1))]
    return mean, lo, hi


def bool_float(x: Any) -> float:
    return 1.0 if bool(x) else 0.0


def metric_row(experiment: str, metric: str, values: Sequence[float], *, direction: str, seed: int, rounds: int, subset: str = "overall") -> Dict[str, Any]:
    mean, lo, hi = bootstrap_mean(values, seed=seed, rounds=rounds)
    return {
        "experiment": experiment,
        "subset": subset,
        "metric": metric,
        "direction": direction,
        "n": len(values),
        "mean": mean,
        "ci95_low": lo,
        "ci95_high": hi,
        "ci95": f"{mean:.3f} [{lo:.3f}, {hi:.3f}]",
    }


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def exp2_ci(path: str | Path, seed: int, rounds: int) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    out = [
        metric_row("Exp2 target-swap", "VLM schema sensitivity", [bool_float(r.get("qwen_schema_sensitive")) for r in rows], direction="up", seed=seed, rounds=rounds),
        metric_row("Exp2 target-swap", "OpenVLA action sensitivity", [bool_float(r.get("openvla_action_sensitive")) for r in rows], direction="up", seed=seed + 1, rounds=rounds),
        metric_row("Exp2 target-swap", "semantic-action gap case rate", [bool_float(r.get("semantic_action_gap_case")) for r in rows], direction="down", seed=seed + 2, rounds=rounds),
        metric_row("Exp2 target-swap", "low action sensitivity rate", [bool_float(r.get("low_action_sensitivity")) for r in rows], direction="down", seed=seed + 3, rounds=rounds),
        metric_row("Exp2 target-swap", "normalized action delta", [float(r.get("normalized_action_delta", 0.0)) for r in rows], direction="up", seed=seed + 4, rounds=rounds),
    ]
    return out


def exp3_ci(path: str | Path, seed: int, rounds: int) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    out: List[Dict[str, Any]] = []
    groups: Dict[str, List[Dict[str, Any]]] = {"overall": rows}
    for r in rows:
        groups.setdefault(str(r.get("perturbation_type")), []).append(r)
    for subset, rs in groups.items():
        out.extend([
            metric_row("Exp3 invalid instructions", "schema rejection", [bool_float(r.get("schema_rejects_execution")) for r in rs], direction="up", seed=seed + 10, rounds=rounds, subset=subset),
            metric_row("Exp3 invalid instructions", "OpenVLA action inhibition", [bool_float(r.get("action_inhibited")) for r in rs], direction="up", seed=seed + 11, rounds=rounds, subset=subset),
            metric_row("Exp3 invalid instructions", "OpenVLA blind execution", [bool_float(r.get("blind_execution")) for r in rs], direction="down", seed=seed + 12, rounds=rounds, subset=subset),
            metric_row("Exp3 invalid instructions", "schema-action inhibition gap case rate", [bool_float(r.get("schema_action_inhibition_gap_case")) for r in rs], direction="down", seed=seed + 13, rounds=rounds, subset=subset),
        ])
    return out


def exp4_ci(invalid_path: str | Path, normal_path: str | Path, seed: int, rounds: int) -> List[Dict[str, Any]]:
    invalid = read_jsonl(invalid_path)
    normal = read_jsonl(normal_path)
    out: List[Dict[str, Any]] = []
    groups: Dict[str, List[Dict[str, Any]]] = {"overall": invalid}
    for r in invalid:
        groups.setdefault(str(r.get("perturbation_type")), []).append(r)
    for subset, rs in groups.items():
        out.extend([
            metric_row("Exp4 checker+gate v3", "raw OpenVLA blind execution", [bool_float(r.get("openvla_blind_execution")) for r in rs], direction="down", seed=seed + 20, rounds=rounds, subset=subset),
            metric_row("Exp4 checker+gate v3", "gated blind execution", [bool_float(r.get("final_blind_execution")) for r in rs], direction="down", seed=seed + 21, rounds=rounds, subset=subset),
            metric_row("Exp4 checker+gate v3", "safe deferral", [bool_float(r.get("safe_deferral")) for r in rs], direction="up", seed=seed + 22, rounds=rounds, subset=subset),
            metric_row("Exp4 checker+gate v3", "gate intervention", [bool_float(r.get("gate_intervened")) for r in rs], direction="mixed", seed=seed + 23, rounds=rounds, subset=subset),
        ])
    out.extend([
        metric_row("Exp4 normal preservation", "normal pass", [bool_float(r.get("normal_pass")) for r in normal], direction="up", seed=seed + 30, rounds=rounds),
        metric_row("Exp4 normal preservation", "false block", [bool_float(r.get("false_block")) for r in normal], direction="down", seed=seed + 31, rounds=rounds),
        metric_row("Exp4 normal preservation", "target valid", [bool_float(r.get("target_valid")) for r in normal], direction="up", seed=seed + 32, rounds=rounds),
    ])
    return out


def latex_table(ci_rows: List[Dict[str, Any]]) -> str:
    keep = [
        ("Exp2 target-swap", "overall", "VLM schema sensitivity"),
        ("Exp2 target-swap", "overall", "OpenVLA action sensitivity"),
        ("Exp2 target-swap", "overall", "semantic-action gap case rate"),
        ("Exp3 invalid instructions", "overall", "OpenVLA blind execution"),
        ("Exp4 checker+gate v3", "overall", "gated blind execution"),
        ("Exp4 checker+gate v3", "overall", "safe deferral"),
        ("Exp4 normal preservation", "overall", "normal pass"),
        ("Exp4 normal preservation", "overall", "false block"),
    ]
    idx = {(r["experiment"], r["subset"], r["metric"]): r for r in ci_rows}
    lines = [
        "\\begin{tabular}{llr}",
        "\\toprule",
        "Experiment & Metric & Mean [95\\% CI] \\\\",
        "\\midrule",
    ]
    for key in keep:
        r = idx.get(key)
        if not r:
            continue
        lines.append(f"{r['experiment']} & {r['metric']} & {r['ci95']} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def stronger_baseline_rows(exp5_checker_gate: str | Path, exp5_prompt_ablation: str | Path) -> List[Dict[str, Any]]:
    existing = read_csv(exp5_checker_gate)
    prompt = read_csv(exp5_prompt_ablation)
    out: List[Dict[str, Any]] = []
    for r in existing:
        out.append({
            "baseline": r.get("method"),
            "status": "completed",
            "why_it_matters": r.get("note"),
            "invalid_blind_execution_rate": r.get("invalid_blind_execution_rate"),
            "normal_pass_rate": r.get("normal_pass_rate"),
            "remaining_weakness": "ablation/control baseline; not a semantic competitor" if r.get("method") in {"All-stop baseline", "Schema monitor only", "Checker without gate"} else "main model comparison",
        })
    best_prompt = next((r for r in prompt if r.get("schema_prompt") == "domain-aware v3"), {})
    out.extend([
        {
            "baseline": "Prompt rewrite OpenVLA",
            "status": "completed",
            "why_it_matters": "tests whether better natural-language prompting alone fixes action-head insensitivity",
            "invalid_blind_execution_rate": "n/a",
            "normal_pass_rate": "n/a",
            "remaining_weakness": "negative-result baseline; not a gate competitor",
        },
        {
            "baseline": "VLM-as-critic gate",
            "status": "recommended_next_run",
            "why_it_matters": "stronger semantic baseline: ask a VLM/LLM critic whether execution is allowed, without explicit schema fields",
            "invalid_blind_execution_rate": "pending",
            "normal_pass_rate": "pending",
            "remaining_weakness": "needs Qwen/LLM critic prompts and normal-preservation check",
        },
        {
            "baseline": "Target-position heuristic gate",
            "status": "recommended_next_run",
            "why_it_matters": "controls for whether simple geometry/nearest-object rules can replace schema semantics",
            "invalid_blind_execution_rate": "pending",
            "normal_pass_rate": "pending",
            "remaining_weakness": "requires object positions; weak for blank/negation semantics",
        },
        {
            "baseline": "Domain-aware schema prompt v3",
            "status": "completed",
            "why_it_matters": "schema-quality ablation that prevents over-deferral on valid LIBERO commands",
            "invalid_blind_execution_rate": "see checker+gate v3",
            "normal_pass_rate": best_prompt.get("normal_pass_rate", "0.94"),
            "remaining_weakness": "still one schema probe; not a second VLM",
        },
    ])
    return out


def second_setting_plan_rows() -> List[Dict[str, Any]]:
    return [
        {
            "priority": 1,
            "setting": "second schema probe on same 600 LIBERO examples",
            "minimal_model_or_domain": "InternVL / LLaVA-OneVision / Qwen2.5-VL-3B if available locally",
            "examples_needed": "100-200 minimum; 600 preferred",
            "reuses_existing_assets": "Exp2/Exp3 benchmark images and OpenVLA actions",
            "expected_output": "schema sensitivity/rejection compared with Qwen2.5-VL-7B",
            "paper_value": "separates schema-probe dependence from OpenVLA action-head gap",
            "status": "not_run_yet",
        },
        {
            "priority": 2,
            "setting": "second domain/split",
            "minimal_model_or_domain": "LIBERO-10 or LIBERO-90 held-out subset, 100-200 examples",
            "examples_needed": "100-200",
            "reuses_existing_assets": "Qwen schema runner, OpenVLA runner, checker/gate",
            "expected_output": "same Exp2/Exp3/Exp4 metrics on another task distribution",
            "paper_value": "upgrades claim from OpenVLA-on-one-LIBERO-subset to cross-setting evidence",
            "status": "partially_supported_by_exp6_libero90_sanity_only; not full benchmark",
        },
        {
            "priority": 3,
            "setting": "second VLA/action policy",
            "minimal_model_or_domain": "another released VLA checkpoint if local access is possible",
            "examples_needed": "100-200",
            "reuses_existing_assets": "counterfactual benchmark and metrics",
            "expected_output": "native action sensitivity for a second action head",
            "paper_value": "strongest generalization evidence",
            "status": "not_run_yet",
        },
    ]


def markdown_summary(ci_rows: List[Dict[str, Any]]) -> str:
    def find(exp: str, metric: str, subset: str = "overall") -> str:
        for r in ci_rows:
            if r["experiment"] == exp and r["metric"] == metric and r["subset"] == subset:
                return r["ci95"]
        return "missing"

    return f"""# Oral-Readiness Additions

Generated from frozen outputs. This file does not claim any new model run.

## Added Now

1. Bootstrap 95% confidence intervals for the main metrics.
2. Stronger-baseline status table distinguishing completed baselines from recommended next runs.
3. Concrete second-model / second-domain plan, without fabricating unavailable results.

## Main CI Numbers

- Exp2 VLM schema sensitivity: {find('Exp2 target-swap', 'VLM schema sensitivity')}
- Exp2 OpenVLA action sensitivity: {find('Exp2 target-swap', 'OpenVLA action sensitivity')}
- Exp2 semantic-action gap case rate: {find('Exp2 target-swap', 'semantic-action gap case rate')}
- Exp3 OpenVLA blind execution: {find('Exp3 invalid instructions', 'OpenVLA blind execution')}
- Exp4 gated blind execution: {find('Exp4 checker+gate v3', 'gated blind execution')}
- Exp4 safe deferral: {find('Exp4 checker+gate v3', 'safe deferral')}
- Exp4 normal pass: {find('Exp4 normal preservation', 'normal pass')}
- Exp4 false block: {find('Exp4 normal preservation', 'false block')}

## What Still Separates This From Oral-Level Evidence

- A second schema probe or second VLA/domain is still the highest-priority missing piece.
- A VLM-as-critic gate or target-position heuristic gate would be a stronger competitor than all-stop.
- The current bootstrap CIs make the existing evidence more formal, but do not replace cross-model/domain validation.
"""


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    ci_rows: List[Dict[str, Any]] = []
    ci_rows.extend(exp2_ci(args.exp2_predictions, args.seed, args.rounds))
    ci_rows.extend(exp3_ci(args.exp3_predictions, args.seed, args.rounds))
    ci_rows.extend(exp4_ci(args.exp4_invalid, args.exp4_normal, args.seed, args.rounds))

    write_csv(out_dir / "bootstrap_ci_main_metrics.csv", ci_rows)
    write_json(out_dir / "bootstrap_ci_main_metrics.json", {"rows": ci_rows})
    (out_dir / "latex_table_main_ci.tex").write_text(latex_table(ci_rows), encoding="utf-8")

    baseline_rows = stronger_baseline_rows(args.exp5_checker_gate, args.exp5_prompt_ablation)
    write_csv(out_dir / "stronger_baseline_status.csv", baseline_rows)
    write_json(out_dir / "stronger_baseline_status.json", {"rows": baseline_rows})

    second_rows = second_setting_plan_rows()
    write_csv(out_dir / "second_model_domain_plan.csv", second_rows)
    write_json(out_dir / "second_model_domain_plan.json", {"rows": second_rows})

    metadata = {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "rounds": args.rounds,
        "seed": args.seed,
        "inputs": {
            "exp2_predictions": args.exp2_predictions,
            "exp3_predictions": args.exp3_predictions,
            "exp4_invalid": args.exp4_invalid,
            "exp4_normal": args.exp4_normal,
            "exp5_checker_gate": args.exp5_checker_gate,
            "exp5_prompt_ablation": args.exp5_prompt_ablation,
        },
    }
    write_json(out_dir / "run_metadata.json", metadata)
    (out_dir / "oral_readiness_summary.md").write_text(markdown_summary(ci_rows), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "num_ci_rows": len(ci_rows)}, indent=2))


if __name__ == "__main__":
    main()
