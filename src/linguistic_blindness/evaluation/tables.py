from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def fmt(v: Any, inverse: bool = False) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):.3f}"
    except Exception:
        return str(v)


def make_markdown_tables(main_rows: List[Dict[str, Any]], failure_rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Linguistic Blindness Tables")
    lines.append("")
    lines.append("## Table 2: Linguistic Blindness Main Results")
    lines.append("| Method | Target Sens. ↑ | Blank Non-Exec ↑ | Impossible Rej. ↑ | Negation Cons. ↑ | CF Cons. ↑ | Safety Viol. ↓ | Blind Exec. ↓ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in main_rows:
        lines.append(f"| {r['method']} | {fmt(r.get('target_sensitivity'))} | {fmt(r.get('blank_nonexecution'))} | {fmt(r.get('impossible_rejection'))} | {fmt(r.get('negation_consistency'))} | {fmt(r.get('counterfactual_consistency'))} | {fmt(r.get('safety_violation'))} | {fmt(r.get('blind_execution'))} |")
    lines.append("")
    lines.append("## Table 3: Task Success vs Counterfactual Sensitivity")
    lines.append("| Method | Normal Instruction Accuracy ↑ | Counterfactual Sensitivity ↑ | Instruction Sensitivity Gap ↓ |")
    lines.append("|---|---:|---:|---:|")
    for r in main_rows:
        lines.append(f"| {r['method']} | {fmt(r.get('normal_instruction_accuracy'))} | {fmt(r.get('counterfactual_consistency'))} | {fmt(r.get('instruction_sensitivity_gap'))} |")
    lines.append("")
    lines.append("## Table 4: Verification / Mitigation Results")
    lines.append("| Method | Schema Validity ↑ | Action Constraint Cons. ↑ | Blocked Action Viol. ↓ | Safety Viol. ↓ | Safe Deferral ↑ |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in main_rows:
        lines.append(f"| {r['method']} | {fmt(r.get('schema_validity'))} | {fmt(r.get('action_constraint_consistency'))} | {fmt(r.get('blocked_action_violation'))} | {fmt(r.get('safety_violation'))} | {fmt(r.get('safe_deferral'))} |")
    lines.append("")
    lines.append("## Table 6: Failure Taxonomy")
    lines.append("| Failure Type | Count | Percentage | Example ID |")
    lines.append("|---|---:|---:|---|")
    for r in failure_rows:
        lines.append(f"| {r['category']} | {r['count']} | {fmt(r.get('percentage'))} | {r.get('example_id', '')} |")
    lines.append("")
    return "\n".join(lines)


def make_latex_table(main_rows: List[Dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated MVP table. Smoke-test rows are not final experimental results.",
        "\\begin{tabular}{lrrrrrrr}",
        "Method & Target & Blank & Missing & Negation & CF & Safety & Blind \\\\",
        "\\hline",
    ]
    for r in main_rows:
        lines.append(
            f"{r['method']} & {fmt(r.get('target_sensitivity'))} & {fmt(r.get('blank_nonexecution'))} & {fmt(r.get('impossible_rejection'))} & {fmt(r.get('negation_consistency'))} & {fmt(r.get('counterfactual_consistency'))} & {fmt(r.get('safety_violation'))} & {fmt(r.get('blind_execution'))} \\\\" 
        )
    lines.extend(["\\end{tabular}", ""])
    return "\n".join(lines)


def write_tables(out_dir: str | Path, main_rows: List[Dict[str, Any]], failure_rows: List[Dict[str, Any]]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "paper_tables.md").write_text(make_markdown_tables(main_rows, failure_rows), encoding="utf-8")
    (out / "latex_tables.tex").write_text(make_latex_table(main_rows), encoding="utf-8")
