#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in [REPO_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from linguistic_blindness.utils.io import read_jsonl, write_csv, write_json


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Paired bootstrap/McNemar stats for schema intervention candidate pools.")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budgets", default="2,4")
    ap.add_argument("--condition-a", default="correct_schema")
    ap.add_argument("--condition-b", default="generic")
    ap.add_argument("--condition-drop", default="field_dropped_schema")
    ap.add_argument("--condition-swapped", default="target_swapped_schema")
    ap.add_argument("--bootstrap-iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    return ap.parse_args()


def condition_block(row: Dict[str, Any], condition: str) -> Optional[Dict[str, Any]]:
    return next((b for b in row.get("condition_candidates", []) if b.get("condition") == condition), None)


def cands(row: Dict[str, Any], condition: str, budget: int) -> List[Dict[str, Any]]:
    block = condition_block(row, condition)
    if not block:
        return []
    return list(block.get("candidates") or [])[:budget]


def any_correct(row: Dict[str, Any], condition: str, budget: int) -> Optional[bool]:
    cs = cands(row, condition, budget)
    if not cs:
        return None
    return any((c.get("actcheck") or {}).get("target_aligned") for c in cs)


def any_schema_redirect(row: Dict[str, Any], condition: str, budget: int) -> Optional[bool]:
    cs = cands(row, condition, budget)
    vals = [
        bool((c.get("schema_target_actcheck") or {}).get("aligned_to_schema_target_over_correct"))
        for c in cs
        if (c.get("schema_target_actcheck") or {}).get("schema_target_known_in_scene")
    ]
    if not vals:
        return None
    return any(vals)


def rate(vals: Iterable[Optional[bool]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return sum(bool(v) for v in xs) / len(xs)


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mcnemar_midp(a: List[bool], b: List[bool]) -> Dict[str, Any]:
    # a/b are paired binary outcomes. b01: a wrong, b right; b10: a right, b wrong.
    b01 = sum((not x) and y for x, y in zip(a, b))
    b10 = sum(x and (not y) for x, y in zip(a, b))
    n = b01 + b10
    if n == 0:
        p = 1.0
    else:
        # Exact two-sided binomial sign test approximation for paired discordants.
        k = min(b01, b10)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)
    return {"discordant_a0_b1": b01, "discordant_a1_b0": b10, "discordant_n": n, "mcnemar_exact_p": p}


def bootstrap_ci(diffs: List[float], iters: int, seed: int) -> Tuple[Optional[float], Optional[float]]:
    if not diffs:
        return None, None
    rng = random.Random(seed)
    n = len(diffs)
    vals = []
    for _ in range(iters):
        vals.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    vals.sort()
    lo = vals[int(0.025 * (iters - 1))]
    hi = vals[int(0.975 * (iters - 1))]
    return float(lo), float(hi)


def group_rows(rows: List[Dict[str, Any]]) -> List[Tuple[str, str, List[Dict[str, Any]]]]:
    out: List[Tuple[str, str, List[Dict[str, Any]]]] = [("overall", "overall", rows)]
    for split in sorted({str(r.get("satbenchpp_split") or "target_swap") for r in rows}):
        out.append(("split", split, [r for r in rows if str(r.get("satbenchpp_split") or "target_swap") == split]))
    for fam in sorted({str(r.get("satbenchpp_family") or "target_swap") for r in rows}):
        out.append(("family", fam, [r for r in rows if str(r.get("satbenchpp_family") or "target_swap") == fam]))
    return out


def paired_compare(rows: List[Dict[str, Any]], cond_a: str, cond_b: str, budget: int, iters: int, seed: int) -> Dict[str, Any]:
    pairs = []
    for r in rows:
        va = any_correct(r, cond_a, budget)
        vb = any_correct(r, cond_b, budget)
        if va is not None and vb is not None:
            pairs.append((bool(va), bool(vb)))
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    diffs = [float(x) - float(y) for x, y in pairs]
    lo, hi = bootstrap_ci(diffs, iters, seed)
    return {
        "condition_a": cond_a,
        "condition_b": cond_b,
        "budget": budget,
        "n_pairs": len(pairs),
        "recall_a": rate(a),
        "recall_b": rate(b),
        "paired_diff_a_minus_b": None if not diffs else sum(diffs) / len(diffs),
        "bootstrap_ci_low": lo,
        "bootstrap_ci_high": hi,
        **mcnemar_midp(a, b),
    }


def redirect_summary(rows: List[Dict[str, Any]], condition: str, budget: int) -> Dict[str, Any]:
    vals = [any_schema_redirect(r, condition, budget) for r in rows]
    return {
        "condition": condition,
        "budget": budget,
        "n_known": len([v for v in vals if v is not None]),
        "schema_target_redirection_rate": rate(vals),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    rows = read_jsonl(args.predictions)
    comp_rows: List[Dict[str, Any]] = []
    redir_rows: List[Dict[str, Any]] = []
    for group_type, group, g in group_rows(rows):
        for budget in budgets:
            for a, b in [
                (args.condition_a, args.condition_b),
                (args.condition_a, args.condition_drop),
            ]:
                row = paired_compare(g, a, b, budget, args.bootstrap_iters, args.seed + budget)
                row.update({"group_type": group_type, "group": group})
                comp_rows.append(row)
            redir = redirect_summary(g, args.condition_swapped, budget)
            redir.update({"group_type": group_type, "group": group})
            redir_rows.append(redir)
    write_csv(out_dir / "paired_candidate_recall_stats.csv", comp_rows)
    write_csv(out_dir / "schema_redirection_stats.csv", redir_rows)
    write_json(out_dir / "run_metadata.json", {
        "predictions": args.predictions,
        "budgets": budgets,
        "bootstrap_iters": args.bootstrap_iters,
        "note": "Paired binary recall uses per-example any-correct-candidate@N. McNemar is exact binomial sign-test over discordant pairs.",
    })
    print(json.dumps({"out_dir": str(out_dir), "paired_rows": len(comp_rows), "redirection_rows": len(redir_rows)}, indent=2))


if __name__ == "__main__":
    main()
