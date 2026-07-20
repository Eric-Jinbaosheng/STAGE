#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def norm(x: Any) -> str:
    return " ".join(str(x or "").strip().lower().split())


def pair_agreement(vals: Iterable[str]) -> Optional[float]:
    xs = [norm(v) for v in vals if norm(v)]
    if len(xs) < 2:
        return None
    total = 0
    agree = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            total += 1
            agree += int(xs[i] == xs[j])
    return agree / total if total else None


def majority(vals: Iterable[str]) -> str:
    xs = [norm(v) for v in vals if norm(v)]
    return Counter(xs).most_common(1)[0][0] if xs else ""


def internal_execute(row: Dict[str, str]) -> str:
    v = norm(row.get("internal_should_execute"))
    if v in {"allow", "execute", "yes", "true", "1"}:
        return "execute"
    if v in {"defer", "ask", "hold", "abort", "no", "false", "0"}:
        return "ask"
    executable = norm(row.get("internal_executable_label"))
    if executable in {"true", "1", "yes"}:
        return "execute"
    if executable in {"false", "0", "no"}:
        return "ask"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Score completed external validation sheets.")
    ap.add_argument("--annotations", required=True, help="CSV containing repeated rows from three annotators.")
    ap.add_argument("--out-dir", default="outputs/linguistic_blindness/external_validation_scored")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.annotations)))
    by_item: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_item[row["annotation_id"]].append(row)

    item_rows: List[Dict[str, Any]] = []
    for aid, group in sorted(by_item.items()):
        first = group[0]
        exec_vals = [r.get("is_instruction_executable", "") for r in group]
        target_vals = [r.get("intended_target", "") for r in group]
        subgoal_vals = [r.get("first_required_subgoal", "") for r in group]
        decision_vals = [r.get("robot_decision_execute_ask_hold_abort", "") for r in group]
        maj_exec = majority(exec_vals)
        maj_target = majority(target_vals)
        maj_subgoal = majority(subgoal_vals)
        maj_decision = majority(decision_vals)
        internal_target = norm(first.get("internal_target"))
        internal_subgoal = norm(first.get("internal_first_subgoal"))
        internal_exec = internal_execute(first)
        item_rows.append({
            "annotation_id": aid,
            "subset": first.get("subset", ""),
            "n_annotators": len(group),
            "execute_pair_agreement": pair_agreement(exec_vals),
            "target_pair_agreement": pair_agreement(target_vals),
            "subgoal_pair_agreement": pair_agreement(subgoal_vals),
            "decision_pair_agreement": pair_agreement(decision_vals),
            "majority_execute": maj_exec,
            "majority_target": maj_target,
            "majority_subgoal": maj_subgoal,
            "majority_decision": maj_decision,
            "internal_execute": internal_exec,
            "internal_target": internal_target,
            "internal_subgoal": internal_subgoal,
            "execute_matches_internal": None if not internal_exec or not maj_exec else (maj_exec == internal_exec or (internal_exec == "ask" and maj_exec in {"no", "ask", "hold", "abort", "ambiguous"})),
            "target_matches_internal": None if not internal_target or not maj_target else (maj_target == internal_target),
            "subgoal_matches_internal": None if not internal_subgoal or not maj_subgoal else (internal_subgoal in maj_subgoal or maj_subgoal in internal_subgoal),
        })

    def avg(key: str, subset: Optional[str] = None) -> Optional[float]:
        vals = []
        for r in item_rows:
            if subset is not None and r["subset"] != subset:
                continue
            v = r.get(key)
            if v is None:
                continue
            vals.append(float(v))
        return None if not vals else sum(vals) / len(vals)

    subsets = ["overall"] + sorted({r["subset"] for r in item_rows})
    summary = []
    for subset in subsets:
        s = None if subset == "overall" else subset
        summary.append({
            "subset": subset,
            "n_items": len(item_rows) if s is None else sum(1 for r in item_rows if r["subset"] == s),
            "execute_pair_agreement": avg("execute_pair_agreement", s),
            "target_pair_agreement": avg("target_pair_agreement", s),
            "subgoal_pair_agreement": avg("subgoal_pair_agreement", s),
            "decision_pair_agreement": avg("decision_pair_agreement", s),
            "execute_matches_internal": avg("execute_matches_internal", s),
            "target_matches_internal": avg("target_matches_internal", s),
            "subgoal_matches_internal": avg("subgoal_matches_internal", s),
        })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "item_agreement.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(item_rows[0].keys()) if item_rows else ["annotation_id"])
        w.writeheader()
        w.writerows(item_rows)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ["subset"])
        w.writeheader()
        w.writerows(summary)
    (out / "summary.json").write_text(json.dumps({"summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
