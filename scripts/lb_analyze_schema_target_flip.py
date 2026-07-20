#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in [REPO_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from linguistic_blindness.utils.io import read_jsonl, write_csv, write_json


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze target identity and flip rates for schema-intervention candidate pools.")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budgets", default="2,4")
    ap.add_argument("--correct-condition", default="correct_schema")
    ap.add_argument("--swapped-condition", default="target_swapped_schema")
    return ap.parse_args()


def finite(x: Any) -> Optional[float]:
    try:
        y = float(x)
        if math.isnan(y):
            return None
        return y
    except Exception:
        return None


def mean(xs: Iterable[Any]) -> Optional[float]:
    vals = [finite(x) for x in xs]
    vals = [x for x in vals if x is not None]
    return None if not vals else float(sum(vals) / len(vals))


def rate(xs: Iterable[Any]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return None if not vals else float(sum(bool(x) for x in vals) / len(vals))


def block(row: Dict[str, Any], condition: str) -> Optional[Dict[str, Any]]:
    return next((b for b in row.get("condition_candidates", []) if b.get("condition") == condition), None)


def cands(row: Dict[str, Any], condition: str, budget: int) -> List[Dict[str, Any]]:
    b = block(row, condition)
    if not b:
        return []
    return list(b.get("candidates") or [])[:budget]


def target_identity(c: Dict[str, Any], margin: float = 0.05) -> str:
    ac = c.get("actcheck") or {}
    sac = c.get("schema_target_actcheck") or {}
    score = finite(ac.get("semantic_action_consistency_score"))
    schema_minus_correct = finite(sac.get("schema_minus_correct_margin"))
    schema_known = bool(sac.get("schema_target_known_in_scene"))
    if schema_known and schema_minus_correct is not None and schema_minus_correct > margin:
        return "schema_target"
    if score is not None and score > margin:
        return "correct_target"
    if score is not None and score < -margin:
        return "original_or_wrong_target"
    return "ambiguous"


def candidate_score(c: Dict[str, Any]) -> float:
    score = finite((c.get("actcheck") or {}).get("semantic_action_consistency_score"))
    return -1e9 if score is None else float(score)


def best_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    return max(candidates, key=candidate_score)


def groups(rows: List[Dict[str, Any]]) -> List[Tuple[str, str, List[Dict[str, Any]]]]:
    out: List[Tuple[str, str, List[Dict[str, Any]]]] = [("overall", "overall", rows)]
    for split in sorted({str(r.get("satbenchpp_split") or "target_swap") for r in rows}):
        out.append(("split", split, [r for r in rows if str(r.get("satbenchpp_split") or "target_swap") == split]))
    for fam in sorted({str(r.get("satbenchpp_family") or "target_swap") for r in rows}):
        out.append(("family", fam, [r for r in rows if str(r.get("satbenchpp_family") or "target_swap") == fam]))
    return out


def summarize_condition(rows: List[Dict[str, Any]], condition: str, budget: int) -> Dict[str, Any]:
    cand_ids: List[str] = []
    best_ids: List[Optional[str]] = []
    mixed_flags: List[Optional[bool]] = []
    for r in rows:
        cs = cands(r, condition, budget)
        if not cs:
            continue
        ids = [target_identity(c) for c in cs]
        cand_ids.extend(ids)
        best = best_candidate(cs)
        best_ids.append(target_identity(best) if best else None)
        mixed_flags.append(("correct_target" in ids) and ("schema_target" in ids))
    return {
        "condition": condition,
        "budget": budget,
        "n_examples": len(best_ids),
        "n_candidates": len(cand_ids),
        "candidate_correct_target_rate": rate(x == "correct_target" for x in cand_ids),
        "candidate_schema_target_rate": rate(x == "schema_target" for x in cand_ids),
        "candidate_original_or_wrong_rate": rate(x == "original_or_wrong_target" for x in cand_ids),
        "candidate_ambiguous_rate": rate(x == "ambiguous" for x in cand_ids),
        "best_correct_target_rate": rate(x == "correct_target" for x in best_ids),
        "best_schema_target_rate": rate(x == "schema_target" for x in best_ids),
        "best_original_or_wrong_rate": rate(x == "original_or_wrong_target" for x in best_ids),
        "best_ambiguous_rate": rate(x == "ambiguous" for x in best_ids),
        "mixed_correct_and_schema_pool_rate": rate(mixed_flags),
    }


def flip_summary(rows: List[Dict[str, Any]], correct_condition: str, swapped_condition: str, budget: int) -> Dict[str, Any]:
    pairs: List[Tuple[str, str]] = []
    for r in rows:
        cb = best_candidate(cands(r, correct_condition, budget))
        sb = best_candidate(cands(r, swapped_condition, budget))
        if cb and sb:
            pairs.append((target_identity(cb), target_identity(sb)))
    return {
        "budget": budget,
        "n_pairs": len(pairs),
        "correct_best_to_swapped_schema_best_rate": rate(a == "correct_target" and b == "schema_target" for a, b in pairs),
        "swapped_best_schema_target_rate": rate(b == "schema_target" for _, b in pairs),
        "correct_best_correct_target_rate": rate(a == "correct_target" for a, _ in pairs),
        "both_best_correct_target_rate": rate(a == "correct_target" and b == "correct_target" for a, b in pairs),
        "swapped_best_not_correct_rate": rate(b != "correct_target" for _, b in pairs),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    rows = read_jsonl(args.predictions)
    condition_rows: List[Dict[str, Any]] = []
    flip_rows: List[Dict[str, Any]] = []
    for group_type, group, g in groups(rows):
        conditions = sorted({b.get("condition") for r in g for b in r.get("condition_candidates", [])})
        for budget in budgets:
            for condition in conditions:
                row = summarize_condition(g, str(condition), budget)
                row.update({"group_type": group_type, "group": group})
                condition_rows.append(row)
            flip = flip_summary(g, args.correct_condition, args.swapped_condition, budget)
            flip.update({"group_type": group_type, "group": group, "correct_condition": args.correct_condition, "swapped_condition": args.swapped_condition})
            flip_rows.append(flip)
    write_csv(out_dir / "candidate_target_identity_summary.csv", condition_rows)
    write_csv(out_dir / "paired_best_target_flip_summary.csv", flip_rows)
    write_json(out_dir / "run_metadata.json", {
        "predictions": args.predictions,
        "budgets": budgets,
        "note": "Target identity uses ActCheck correct-target margin and schema-target-vs-correct margin. A pool can contain both correct-target and schema-target candidates.",
    })
    print(json.dumps({"out_dir": str(out_dir), "condition_rows": len(condition_rows), "flip_rows": len(flip_rows)}, indent=2))


if __name__ == "__main__":
    main()
