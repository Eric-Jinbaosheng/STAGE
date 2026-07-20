#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_rows(run_dir: Path) -> List[Dict[str, Any]]:
    return read_jsonl(run_dir / "visa_closed_loop_rollouts.jsonl") or read_jsonl(run_dir / "visa_closed_loop_rollouts.partial.jsonl")


def mean(vals: Iterable[Any]) -> Optional[float]:
    xs: List[float] = []
    for v in vals:
        try:
            if v is not None and not math.isnan(float(v)):
                xs.append(float(v))
        except Exception:
            pass
    return None if not xs else sum(xs) / len(xs)


def rate(vals: Iterable[Any]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return None if not xs else sum(bool(v) for v in xs) / len(xs)


def summarize(rows: List[Dict[str, Any]], methods: List[str], competent_keys: Optional[set[Tuple[int, int]]] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for method in methods:
        g = [r for r in rows if r.get("method") == method]
        if competent_keys is not None:
            g = [r for r in g if (int(r["task_id"]), int(r["trial_idx"])) in competent_keys]
        out.append({
            "method": method,
            "n": len(g),
            "correct_contact_rate": rate(r.get("correct_contact") for r in g),
            "correct_lift_rate": rate(r.get("correct_lift") for r in g),
            "wrong_contact_rate": rate(r.get("wrong_contact") for r in g),
            "wrong_lift_rate": rate(r.get("wrong_lift") for r in g),
            "correct_first_contact_rate": rate(r.get("correct_first_contact") for r in g),
            "wrong_first_contact_rate": rate(r.get("wrong_first_contact") for r in g),
            "mean_correct_contact_step": mean(r.get("correct_contact_step") for r in g),
            "mean_final_target_preference": mean(r.get("target_preference_score") for r in g),
            "mean_integrated_target_preference": mean(r.get("integrated_target_preference_score") for r in g),
            "mean_target_approach_auc": mean(r.get("target_approach_auc") for r in g),
            "mean_wrong_object_approach_auc": mean(r.get("wrong_object_approach_auc") for r in g),
            "wrong_object_approach_rate": rate(r.get("wrong_object_approach") for r in g),
            "mean_residual_correction_rate": mean(r.get("residual_correction_rate") for r in g),
            "mean_residual_alpha": mean(r.get("mean_residual_alpha") for r in g),
        })
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counterfactual-run-dir", required=True)
    ap.add_argument("--original-validation-run-dir", default="")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cf_rows = read_rows(Path(args.counterfactual_run_dir))
    orig_rows = read_rows(Path(args.original_validation_run_dir)) if args.original_validation_run_dir else []
    methods = sorted({str(r.get("method")) for r in cf_rows})
    competent_keys: Optional[set[Tuple[int, int]]] = None
    if orig_rows:
        competent_keys = {
            (int(r["task_id"]), int(r["trial_idx"]))
            for r in orig_rows
            if r.get("method") == "native_openvla" and bool(r.get("success"))
        }

    out_dir = Path(args.out_dir)
    summary_all = summarize(cf_rows, methods)
    summary_competent = summarize(cf_rows, methods, competent_keys) if competent_keys is not None else []
    write_csv(out_dir / "summary_all.csv", summary_all)
    write_csv(out_dir / "summary_base_policy_competent.csv", summary_competent)
    (out_dir / "summary.json").write_text(json.dumps({
        "counterfactual_run_dir": args.counterfactual_run_dir,
        "original_validation_run_dir": args.original_validation_run_dir,
        "n_counterfactual_rows": len(cf_rows),
        "n_original_validation_rows": len(orig_rows),
        "base_policy_competent_keys": sorted([list(x) for x in competent_keys]) if competent_keys is not None else None,
        "summary_all": summary_all,
        "summary_base_policy_competent": summary_competent,
    }, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
