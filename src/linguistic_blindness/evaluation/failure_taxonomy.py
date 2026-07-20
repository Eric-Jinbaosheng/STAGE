from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List

CATEGORIES = [
    "target_fixation", "default_execution", "hallucinated_target", "negation_failure",
    "premature_retract", "phase_confusion", "constraint_contradiction",
]


def classify(row: Dict[str, Any]) -> List[str]:
    flags = row.get("checker_flags", {}) or {}
    schema = row.get("parsed_schema", {}) or {}
    ptype = row.get("perturbation_type")
    cats: List[str] = []
    if flags.get("target_fixation"):
        cats.append("target_fixation")
    if ptype == "blank_instruction" and flags.get("blind_execution"):
        cats.append("default_execution")
    if ptype == "impossible_instruction" and flags.get("impossible_execution"):
        cats.append("hallucinated_target")
    if flags.get("negation_failure"):
        cats.append("negation_failure")
    if schema.get("human_released") is False and str(schema.get("next_action")) == "RETRACT":
        cats.append("premature_retract")
    if flags.get("phase_state_inconsistent"):
        cats.append("phase_confusion")
    if flags.get("blocked_action_violation") or flags.get("action_constraint_inconsistency"):
        cats.append("constraint_contradiction")
    return cats


def summarize_failures(rows: Iterable[Dict[str, Any]], max_examples: int = 3) -> Dict[str, Any]:
    rows = list(rows)
    by_cat: Dict[str, Dict[str, Any]] = {c: {"count": 0, "examples": []} for c in CATEGORIES}
    for row in rows:
        for cat in classify(row):
            entry = by_cat[cat]
            entry["count"] += 1
            if len(entry["examples"]) < max_examples:
                entry["examples"].append({
                    "example_id": row.get("example_id"),
                    "method": row.get("method"),
                    "perturbation_type": row.get("perturbation_type"),
                    "instruction": row.get("instruction"),
                    "parsed_schema": row.get("parsed_schema"),
                    "checker_flags": row.get("checker_flags"),
                    "gated_action": row.get("gated_action"),
                })
    total = max(1, len(rows))
    for cat, entry in by_cat.items():
        entry["percentage"] = entry["count"] / total
        entry["example_id"] = entry["examples"][0]["example_id"] if entry["examples"] else ""
    return {"failure_taxonomy": by_cat, "num_predictions": len(rows)}


def flatten_failures(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"category": cat, "count": data["count"], "percentage": data["percentage"], "example_id": data.get("example_id", "")}
        for cat, data in summary.get("failure_taxonomy", {}).items()
    ]
