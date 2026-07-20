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
    ap = argparse.ArgumentParser(description="Compare candidate support for the intervened/swapped target across schema conditions.")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budgets", default="2,4")
    ap.add_argument("--margin", type=float, default=0.05)
    return ap.parse_args()


def finite(x: Any) -> Optional[float]:
    try:
        y = float(x)
        if math.isnan(y):
            return None
        return y
    except Exception:
        return None


def rate(xs: Iterable[Any]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return None if not vals else float(sum(bool(x) for x in vals) / len(vals))


def block(row: Dict[str, Any], condition: str) -> Optional[Dict[str, Any]]:
    return next((b for b in row.get("condition_candidates", []) if b.get("condition") == condition), None)


def cands(row: Dict[str, Any], condition: str, budget: int) -> List[Dict[str, Any]]:
    b = block(row, condition)
    return [] if not b else list(b.get("candidates") or [])[:budget]


def correct_margin(c: Dict[str, Any]) -> Optional[float]:
    return finite((c.get("actcheck") or {}).get("semantic_action_consistency_score"))


def best_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    return max(candidates, key=lambda c: -1e9 if correct_margin(c) is None else float(correct_margin(c)))


def is_correct(c: Dict[str, Any], margin: float) -> bool:
    m = correct_margin(c)
    return bool(m is not None and m > margin)


def is_intervened_target(c: Dict[str, Any], margin: float) -> bool:
    # In these counterfactual pairs, the swapped/intervened schema target is the
    # original target. ActCheck wrong_target is therefore support for that target.
    m = correct_margin(c)
    return bool(m is not None and m < -margin)


def groups(rows: List[Dict[str, Any]]) -> List[Tuple[str, str, List[Dict[str, Any]]]]:
    out: List[Tuple[str, str, List[Dict[str, Any]]]] = [("overall", "overall", rows)]
    for split in sorted({str(r.get("satbenchpp_split") or "target_swap") for r in rows}):
        out.append(("split", split, [r for r in rows if str(r.get("satbenchpp_split") or "target_swap") == split]))
    for fam in sorted({str(r.get("satbenchpp_family") or "target_swap") for r in rows}):
        out.append(("family", fam, [r for r in rows if str(r.get("satbenchpp_family") or "target_swap") == fam]))
    return out


def summarize_condition(rows: List[Dict[str, Any]], condition: str, budget: int, margin: float) -> Dict[str, Any]:
    cand_correct: List[bool] = []
    cand_intervened: List[bool] = []
    best_correct: List[Optional[bool]] = []
    best_intervened: List[Optional[bool]] = []
    mixed: List[Optional[bool]] = []
    any_intervened: List[Optional[bool]] = []
    any_correct: List[Optional[bool]] = []
    for r in rows:
        cs = cands(r, condition, budget)
        if not cs:
            continue
        corr = [is_correct(c, margin) for c in cs]
        inter = [is_intervened_target(c, margin) for c in cs]
        cand_correct.extend(corr)
        cand_intervened.extend(inter)
        b = best_candidate(cs)
        best_correct.append(is_correct(b, margin) if b else None)
        best_intervened.append(is_intervened_target(b, margin) if b else None)
        any_correct.append(any(corr))
        any_intervened.append(any(inter))
        mixed.append(any(corr) and any(inter))
    return {
        "condition": condition,
        "budget": budget,
        "n_examples": len(best_correct),
        "n_candidates": len(cand_correct),
        "candidate_correct_target_rate": rate(cand_correct),
        "candidate_intervened_target_rate": rate(cand_intervened),
        "pool_correct_target_presence": rate(any_correct),
        "pool_intervened_target_presence": rate(any_intervened),
        "mixed_correct_and_intervened_pool_rate": rate(mixed),
        "best_correct_target_rate": rate(best_correct),
        "best_intervened_target_rate": rate(best_intervened),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    rows = read_jsonl(args.predictions)
    out_rows: List[Dict[str, Any]] = []
    for group_type, group, g in groups(rows):
        conditions = sorted({b.get("condition") for r in g for b in r.get("condition_candidates", [])})
        for budget in budgets:
            for condition in conditions:
                row = summarize_condition(g, str(condition), budget, args.margin)
                row.update({"group_type": group_type, "group": group})
                out_rows.append(row)
    write_csv(out_dir / "intervened_target_support_summary.csv", out_rows)
    write_json(out_dir / "run_metadata.json", {
        "predictions": args.predictions,
        "budgets": budgets,
        "margin": args.margin,
        "note": "Intervened target is the original target used in target-swapped schema conditions; support is ActCheck wrong-target margin.",
    })
    print(json.dumps({"out_dir": str(out_dir), "rows": len(out_rows)}, indent=2))


if __name__ == "__main__":
    main()
