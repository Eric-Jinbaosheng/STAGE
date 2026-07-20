#!/usr/bin/env python3
"""Exp5: Ablation, robustness, and diagnostic analysis.

This script is post-hoc only. It does not rerun OpenVLA or Qwen. It reads the
frozen Exp3/Exp4 outputs and summarizes prompt/schema quality, checker/gate
ablations, strict-vs-execution-aware behavior, and residual failures.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from linguistic_blindness.utils.io import command_string, read_jsonl, utc_timestamp, write_csv, write_json

SAFE_DEFERRALS = {"ASK", "PROMPT", "WAIT", "HOLD", "ABORT", "BACKOFF", "TARGET_NOT_FOUND"}
TASK_ACTIONS = {"MOVE_TO", "GRASP", "PICK", "PLACE", "RETRACT", "TRANSFER"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Exp5 ablation and diagnostic analysis from existing outputs.")
    p.add_argument("--out-dir", default="outputs/linguistic_blindness/exp5_ablation_diagnostic_v3")
    p.add_argument("--normal-v1", default="outputs/linguistic_blindness/exp4_checker_gate_mitigation_clean_600_qwen_normal/exp4_normal_preservation_predictions.jsonl")
    p.add_argument("--normal-v2", default="outputs/linguistic_blindness/exp4_checker_gate_mitigation_clean_600_qwen_normal_v2/exp4_normal_preservation_predictions.jsonl")
    p.add_argument("--normal-v2-repair", default="outputs/linguistic_blindness/exp4_checker_gate_mitigation_clean_600_qwen_normal_v2_repair_check/exp4_normal_preservation_predictions.jsonl")
    p.add_argument("--normal-v3", default="outputs/linguistic_blindness/exp4_checker_gate_mitigation_clean_600_qwen_normal_v3/exp4_normal_preservation_predictions.jsonl")
    p.add_argument("--normal-v3-repair", default="outputs/linguistic_blindness/exp4_checker_gate_mitigation_clean_600_qwen_normal_v3_repair/exp4_normal_preservation_predictions.jsonl")
    p.add_argument("--exp4-v3", default="outputs/linguistic_blindness/exp4_checker_gate_mitigation_clean_600_v3_unified")
    p.add_argument("--exp4-v1", default="outputs/linguistic_blindness/exp4_checker_gate_mitigation_clean_600")
    p.add_argument("--exp4-v2", default="outputs/linguistic_blindness/exp4_checker_gate_mitigation_clean_600_qwen_normal_v2")
    return p.parse_args()


def exists(path: str | Path) -> bool:
    return Path(path).exists()


def safe_read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    return read_jsonl(p) if p.exists() else []


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def action_name(x: Any) -> str:
    return str(x or "").strip().upper()


def is_task_action(x: Any) -> bool:
    name = action_name(x)
    return any(name == a or name.startswith(a + "_") for a in TASK_ACTIONS)


def schema_for(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("schema") or row.get("qwen_schema_perturbed") or {}


def list_actions(xs: Any) -> List[str]:
    return [action_name(x) for x in xs] if isinstance(xs, list) else []


def rate(vals: Iterable[Any]) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    if not clean:
        return None
    return sum(bool(v) for v in clean) / len(clean)


def prompt_row(name: str, path: str) -> Dict[str, Any]:
    rows = safe_read_jsonl(path)
    n = len(rows)
    if n == 0:
        return {"schema_prompt": name, "n": 0, "missing": True}
    next_counts = Counter(action_name(r.get("normal_next_action") or schema_for(r).get("next_action")) for r in rows)
    ask_prompt = sum(next_counts[a] for a in ["ASK", "PROMPT"])
    contradiction = 0
    target_valid = 0
    task_next = 0
    repaired = 0
    for r in rows:
        s = schema_for(r)
        nxt = action_name(r.get("normal_next_action") or s.get("next_action"))
        blocked = list_actions(s.get("blocked_actions", []))
        contradiction += int(bool(nxt and nxt in blocked))
        target = r.get("normal_target_object", s.get("target_object"))
        target_exists = r.get("normal_target_exists", s.get("target_exists"))
        target_valid += int(bool(target not in {None, "", "null", "None"} and target_exists is not False))
        task_next += int(is_task_action(nxt))
        repaired += int(bool(r.get("schema_repaired")))
    return {
        "schema_prompt": name,
        "n": n,
        "normal_pass_rate": rate(r.get("normal_pass") for r in rows),
        "false_block_rate": rate(r.get("false_block") for r in rows),
        "target_valid_rate": target_valid / n,
        "next_action_task_rate": task_next / n,
        "ask_prompt_overdeferral_count": ask_prompt,
        "ask_prompt_overdeferral_rate": ask_prompt / n,
        "self_contradiction_count": contradiction,
        "self_contradiction_rate": contradiction / n,
        "schema_repair_count": repaired,
        "schema_repair_rate": repaired / n,
        "next_action_distribution": dict(sorted(next_counts.items())),
        "missing": False,
    }


def load_exp4_summary(exp4_dir: str | Path) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    d = Path(exp4_dir)
    invalid_rows = []
    if (d / "exp4_invalid_mitigation_summary.json").exists():
        invalid_rows = read_json(d / "exp4_invalid_mitigation_summary.json").get("rows", [])
    invalid_overall = next((r for r in invalid_rows if r.get("perturbation_type") == "overall"), {})
    normal = read_json(d / "exp4_normal_preservation_summary.json") if (d / "exp4_normal_preservation_summary.json").exists() else {}
    comparison = {}
    if (d / "exp4_method_comparison.json").exists():
        comparison = read_json(d / "exp4_method_comparison.json")
    return invalid_overall, normal, comparison


def checker_gate_ablation(exp4_dir: str | Path) -> List[Dict[str, Any]]:
    invalid, normal, _ = load_exp4_summary(exp4_dir)
    openvla_blind = invalid.get("openvla_blind_execution_rate")
    gated_blind = invalid.get("gated_blind_execution_rate")
    safe = invalid.get("safe_deferral_rate")
    intervention = invalid.get("gate_intervention_rate")
    normal_pass = normal.get("normal_pass_rate")
    false_block = normal.get("false_block_rate")
    return [
        {
            "method": "OpenVLA raw",
            "invalid_blind_execution_rate": openvla_blind,
            "invalid_safe_deferral_rate": 0.0,
            "normal_pass_rate": 1.0,
            "false_block_rate": 0.0,
            "intervention_rate": 0.0,
            "note": "raw continuous action; no symbolic schema intervention",
        },
        {
            "method": "All-stop baseline",
            "invalid_blind_execution_rate": 0.0,
            "invalid_safe_deferral_rate": 1.0,
            "normal_pass_rate": 0.0,
            "false_block_rate": 1.0,
            "intervention_rate": 1.0,
            "note": "blocks everything, including normal instructions",
        },
        {
            "method": "Schema monitor only",
            "invalid_blind_execution_rate": openvla_blind,
            "invalid_safe_deferral_rate": 0.0,
            "normal_pass_rate": normal_pass,
            "false_block_rate": false_block,
            "intervention_rate": 0.0,
            "note": "detects schema state but does not change OpenVLA action",
        },
        {
            "method": "Checker without gate",
            "invalid_blind_execution_rate": openvla_blind,
            "invalid_safe_deferral_rate": 0.0,
            "normal_pass_rate": normal_pass,
            "false_block_rate": false_block,
            "intervention_rate": 0.0,
            "note": "flags violations but does not intercept execution",
        },
        {
            "method": "Checker + Gate v3",
            "invalid_blind_execution_rate": gated_blind,
            "invalid_safe_deferral_rate": safe,
            "normal_pass_rate": normal_pass,
            "false_block_rate": false_block,
            "intervention_rate": intervention,
            "note": "schema-conditioned execution gate",
        },
    ]


def strict_vs_execution(exp4_dir: str | Path) -> List[Dict[str, Any]]:
    d = Path(exp4_dir)
    invalid_rows = read_json(d / "exp4_invalid_mitigation_summary.json").get("rows", [])
    out = []
    for r in invalid_rows:
        if r.get("perturbation_type") == "overall":
            continue
        exec_rate = r.get("execution_aware_rejection_rate")
        strict = r.get("strict_schema_rejection_rate")
        out.append({
            "perturbation_type": r.get("perturbation_type"),
            "n": r.get("n"),
            "execution_aware_rejection_rate": exec_rate,
            "strict_schema_rejection_rate": strict,
            "discrepancy": None if exec_rate is None or strict is None else exec_rate - strict,
            "blank_target_leakage_rate": r.get("blank_target_leakage_rate"),
            "negation_missing_task_block_rate": r.get("negation_missing_task_block_rate"),
            "gated_blind_execution_rate": r.get("gated_blind_execution_rate"),
        })
    return out


def failure_analysis(exp4_dir: str | Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    d = Path(exp4_dir)
    invalid = safe_read_jsonl(d / "exp4_predictions.jsonl")
    normal = safe_read_jsonl(d / "exp4_normal_preservation_predictions.jsonl")
    rows = []
    cases: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    blank = [r for r in invalid if r.get("perturbation_type") == "blank_instruction"]
    blank_leak = [r for r in blank if r.get("blank_target_leakage")]
    blank_final = [r for r in blank if r.get("final_blind_execution")]
    rows.append({
        "failure_type": "Blank Target Leakage / Default Execution Bias",
        "count": len(blank_leak),
        "rate": len(blank_leak) / len(blank) if blank else None,
        "impact_count_final_blind": len(blank_final),
        "description": "blank instruction still retains a scene target or task action",
    })
    cases["blank_default_execution"] = case_rows(blank_final[:5])

    neg = [r for r in invalid if r.get("perturbation_type") == "negation"]
    neg_missing = [r for r in neg if r.get("negation_missing_task_block")]
    rows.append({
        "failure_type": "Negation Blocked-Action Incompleteness",
        "count": len(neg_missing),
        "rate": len(neg_missing) / len(neg) if neg else None,
        "impact_count_final_blind": sum(bool(r.get("final_blind_execution")) for r in neg_missing),
        "description": "negation safely defers but blocked_actions often omits the prohibited task action",
    })
    cases["negation_missing_blocked_action"] = case_rows(neg_missing[:5])

    normal_false = [r for r in normal if r.get("false_block")]
    rows.append({
        "failure_type": "Normal False Block",
        "count": len(normal_false),
        "rate": len(normal_false) / len(normal) if normal else None,
        "impact_count_final_blind": None,
        "description": "normal instruction schema chooses safe deferral, usually ASK, causing a false block",
    })
    cases["normal_false_block"] = normal_case_rows(normal_false[:5])
    return rows, {"cases": cases}


def case_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        s = r.get("qwen_schema_perturbed") or {}
        out.append({
            "example_id": r.get("example_id"),
            "perturbation_type": r.get("perturbation_type"),
            "instruction": r.get("perturbed_instruction"),
            "next_action": s.get("next_action"),
            "target_object": s.get("target_object"),
            "target_exists": s.get("target_exists"),
            "blocked_actions": s.get("blocked_actions"),
            "final_blind_execution": r.get("final_blind_execution"),
        })
    return out


def normal_case_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        s = r.get("schema") or {}
        out.append({
            "example_id": r.get("example_id"),
            "instruction": r.get("original_instruction"),
            "next_action": s.get("next_action"),
            "target_object": s.get("target_object"),
            "target_exists": s.get("target_exists"),
            "blocked_actions": s.get("blocked_actions"),
            "false_block": r.get("false_block"),
        })
    return out


def latex_table(prompt_rows: List[Dict[str, Any]], checker_rows: List[Dict[str, Any]], strict_rows: List[Dict[str, Any]]) -> str:
    def fmt(x: Any) -> str:
        if x is None:
            return "NA"
        try:
            return f"{float(x):.3f}"
        except Exception:
            return str(x)

    nl = r" \\" 
    lines: List[str] = []
    lines.extend([
        "% Exp5 prompt/schema quality ablation",
        r"\begin{tabular}{lrrrr}",
        r"Prompt & Normal Pass $\uparrow$ & False Block $\downarrow$ & ASK/PROMPT $\downarrow$ & Contrad. $\downarrow$" + nl,
        r"\hline",
    ])
    for r in prompt_rows:
        lines.append(
            f"{r['schema_prompt']} & {fmt(r.get('normal_pass_rate'))} & {fmt(r.get('false_block_rate'))} & "
            f"{fmt(r.get('ask_prompt_overdeferral_rate'))} & {fmt(r.get('self_contradiction_rate'))}" + nl
        )
    lines.extend([
        r"\end{tabular}",
        "",
        "% Exp5 checker/gate ablation",
        r"\begin{tabular}{lrrrr}",
        r"Method & Invalid Blind $\downarrow$ & Safe Def. $\uparrow$ & Normal Pass $\uparrow$ & False Block $\downarrow$" + nl,
        r"\hline",
    ])
    for r in checker_rows:
        lines.append(
            f"{r['method']} & {fmt(r.get('invalid_blind_execution_rate'))} & {fmt(r.get('invalid_safe_deferral_rate'))} & "
            f"{fmt(r.get('normal_pass_rate'))} & {fmt(r.get('false_block_rate'))}" + nl
        )
    lines.extend([
        r"\end{tabular}",
        "",
        "% Exp5 strict vs execution-aware",
        r"\begin{tabular}{lrrrr}",
        r"Perturbation & Exec-aware $\uparrow$ & Strict $\uparrow$ & Gap & Gated Blind $\downarrow$" + nl,
        r"\hline",
    ])
    for r in strict_rows:
        lines.append(
            f"{r['perturbation_type']} & {fmt(r.get('execution_aware_rejection_rate'))} & {fmt(r.get('strict_schema_rejection_rate'))} & "
            f"{fmt(r.get('discrepancy'))} & {fmt(r.get('gated_blind_execution_rate'))}" + nl
        )
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_rows = [
        prompt_row("naive/default", args.normal_v1),
        prompt_row("normal-preservation v2", args.normal_v2),
        prompt_row("v2 + consistency repair", args.normal_v2_repair),
        prompt_row("domain-aware v3", args.normal_v3),
        prompt_row("domain-aware v3 + repair", args.normal_v3_repair),
    ]
    checker_rows = checker_gate_ablation(args.exp4_v3)
    strict_rows = strict_vs_execution(args.exp4_v3)
    failure_rows, cases = failure_analysis(args.exp4_v3)

    write_csv(out_dir / "exp5_prompt_ablation.csv", prompt_rows)
    write_json(out_dir / "exp5_prompt_ablation.json", {"rows": prompt_rows})
    write_csv(out_dir / "exp5_checker_gate_ablation.csv", checker_rows)
    write_json(out_dir / "exp5_checker_gate_ablation.json", {"rows": checker_rows})
    write_csv(out_dir / "exp5_strict_vs_execution_aware.csv", strict_rows)
    write_json(out_dir / "exp5_strict_vs_execution_aware.json", {"rows": strict_rows})
    write_csv(out_dir / "exp5_failure_analysis.csv", failure_rows)
    write_json(out_dir / "exp5_failure_analysis.json", {"rows": failure_rows})
    write_json(out_dir / "qualitative_failure_cases.json", cases)
    (out_dir / "latex_table_exp5.tex").write_text(latex_table(prompt_rows, checker_rows, strict_rows), encoding="utf-8")
    write_json(out_dir / "run_metadata.json", {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "exp4_v3": args.exp4_v3,
        "python": sys.version,
        "platform": platform.platform(),
        "note": "Post-hoc Exp5 only; no model reruns.",
    })
    print(json.dumps({"out_dir": str(out_dir), "prompt_rows": prompt_rows, "checker_rows": checker_rows, "strict_rows": strict_rows, "failure_rows": failure_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
