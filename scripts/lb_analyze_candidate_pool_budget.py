#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

for p in [REPO_ROOT / "src", REPO_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from linguistic_blindness.utils.io import read_jsonl, write_csv, write_json


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute-matched pre-rerank candidate pool analysis for VISA-Rerank outputs.")
    ap.add_argument("--schema-predictions", required=True)
    ap.add_argument("--generic-predictions", required=True)
    ap.add_argument("--metadata-jsonl", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budgets", default="1,2,4,6,8,16")
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


def mean(xs: Iterable[Any]) -> Optional[float]:
    vals = [finite(x) for x in xs]
    vals = [x for x in vals if x is not None]
    return None if not vals else float(sum(vals) / len(vals))


def rate(xs: Iterable[Any]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return None if not vals else float(sum(bool(x) for x in vals) / len(vals))


def get_score(c: Dict[str, Any]) -> Optional[float]:
    return finite((c.get("actcheck") or {}).get("semantic_action_consistency_score"))


def is_aligned(c: Dict[str, Any]) -> bool:
    return bool((c.get("actcheck") or {}).get("target_aligned"))


def is_wrong_or_amb(c: Dict[str, Any]) -> bool:
    ac = c.get("actcheck") or {}
    return bool(ac.get("wrong_target") or ac.get("ambiguous"))


def best(cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cands:
        return None
    return max(cands, key=lambda c: -1e9 if get_score(c) is None else float(get_score(c)))


def load_meta(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    return {str(r.get("example_id")): r for r in read_jsonl(path) if r.get("example_id")}


def family_for(row: Dict[str, Any], meta: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    m = meta.get(str(row.get("example_id")), {})
    split = row.get("satbenchpp_split") or m.get("satbenchpp_split") or "target_swap"
    fam = row.get("satbenchpp_family") or m.get("satbenchpp_family") or "target_swap"
    return str(split), str(fam)


def summarize(rows: List[Dict[str, Any]], generator: str, budgets: List[int], meta: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    groups: List[Tuple[str, str, List[Dict[str, Any]]]] = [("overall", "overall", rows)]
    splits = sorted({family_for(r, meta)[0] for r in rows})
    for split in splits:
        groups.append(("split", split, [r for r in rows if family_for(r, meta)[0] == split]))
    families = sorted({family_for(r, meta)[1] for r in rows})
    for fam in families:
        groups.append(("family", fam, [r for r in rows if family_for(r, meta)[1] == fam]))

    for group_type, group, g in groups:
        for n in budgets:
            per_example = []
            for r in g:
                cands = list(r.get("candidate_actions") or [])[:n]
                if not cands:
                    continue
                b = best(cands)
                scores = [get_score(c) for c in cands]
                per_example.append({
                    "any_good": any(is_aligned(c) for c in cands),
                    "all_bad": all(is_wrong_or_amb(c) for c in cands),
                    "best_good": is_aligned(b) if b else None,
                    "best_score": get_score(b) if b else None,
                    "mean_score": mean(scores),
                    "wrong_rate": rate(is_wrong_or_amb(c) for c in cands),
                    "aligned_fraction": rate(is_aligned(c) for c in cands),
                    "num_candidates": len(cands),
                })
            out.append({
                "generator": generator,
                "group_type": group_type,
                "group": group,
                "budget": n,
                "n_examples": len(per_example),
                "mean_realized_candidates": mean(p["num_candidates"] for p in per_example),
                "candidate_recall_at_n": rate(p["any_good"] for p in per_example),
                "best_candidate_aligned_rate": rate(p["best_good"] for p in per_example),
                "mean_best_target_margin": mean(p["best_score"] for p in per_example),
                "mean_candidate_target_margin": mean(p["mean_score"] for p in per_example),
                "mean_wrong_or_ambiguous_candidate_rate": mean(p["wrong_rate"] for p in per_example),
                "mean_aligned_candidate_fraction": mean(p["aligned_fraction"] for p in per_example),
                "all_candidates_wrong_or_ambiguous_rate": rate(p["all_bad"] for p in per_example),
            })
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    meta = load_meta(args.metadata_jsonl)
    schema = read_jsonl(args.schema_predictions)
    generic = read_jsonl(args.generic_predictions)
    rows = summarize(schema, "schema_conditioned", budgets, meta) + summarize(generic, "generic_no_schema", budgets, meta)
    write_csv(out_dir / "candidate_budget_summary.csv", rows)
    write_json(out_dir / "run_metadata.json", {
        "schema_predictions": args.schema_predictions,
        "generic_predictions": args.generic_predictions,
        "metadata_jsonl": args.metadata_jsonl,
        "budgets": budgets,
        "note": "Pre-rerank candidate-pool analysis. Recall@N is whether at least one candidate is ActCheck target-aligned before selector/defer is applied.",
    })
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
