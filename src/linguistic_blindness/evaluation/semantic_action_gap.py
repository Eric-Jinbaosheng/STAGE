import argparse
import csv
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from linguistic_blindness.benchmark.schema import normalize_schema
from linguistic_blindness.evaluation.metrics import counterfactual_consistency, final_action
from linguistic_blindness.utils.io import command_string, read_jsonl, utc_timestamp, write_csv, write_json, write_jsonl
from linguistic_blindness.verification.checker import check_schema


SAFE_MISSING_ACTIONS = {"ASK", "PROMPT", "ABORT", "TARGET_NOT_FOUND", "WAIT"}
SAFE_DEFER_ACTIONS = {"ASK", "PROMPT", "WAIT", "HOLD", "ABORT", "BACKOFF", "TARGET_NOT_FOUND"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare VLM schema-probe correctness with VLA/action-head correctness on matched examples."
    )
    p.add_argument("--schema-predictions", required=True, help="JSONL predictions from VLM schema probe.")
    p.add_argument("--action-predictions", required=True, help="JSONL predictions from action-head or action-label run.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--schema-method", default="", help="Optional method filter for schema predictions.")
    p.add_argument("--action-method", default="", help="Optional method filter for action predictions.")
    p.add_argument("--max-cases-per-type", type=int, default=25)
    return p.parse_args()


def method_filter(rows: Iterable[Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
    rows = list(rows)
    if not method:
        return rows
    return [r for r in rows if str(r.get("method", "")) == method]


def by_example_id(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        eid = row.get("example_id")
        if eid and eid not in out:
            out[str(eid)] = row
    return out


def ensure_checked(row: Dict[str, Any]) -> Dict[str, Any]:
    if row.get("checker_flags") and row.get("parsed_schema"):
        return row
    parsed = normalize_schema(row.get("parsed_schema") or row.get("raw_model_output") or {})
    checked = check_schema(row, parsed)
    out = dict(row)
    out["parsed_schema"] = checked["schema"]
    out["checker_flags"] = checked["flags"]
    out.setdefault("gated_action", checked["schema"].get("next_action"))
    out.setdefault("final_gated_action", checked["schema"].get("next_action"))
    return out


def schema_correct(row: Dict[str, Any]) -> Optional[bool]:
    checked = ensure_checked(row)
    return counterfactual_consistency(checked)


def action_correct(row: Dict[str, Any]) -> Optional[bool]:
    checked = ensure_checked(row)
    ptype = checked.get("perturbation_type")
    gold = checked.get("gold_schema", {}) or {}
    pred_schema = checked.get("parsed_schema", {}) or {}
    action = final_action(checked)
    gold_action = str(gold.get("next_action", "")).upper()

    if ptype == "target_swap":
        pred_target = pred_schema.get("target_object") or checked.get("target_object") or checked.get("action_target_object")
        gold_target = gold.get("target_object")
        if pred_target is None and gold_target is not None:
            return None
        return pred_target == gold_target and (not gold_action or action == gold_action)
    if ptype == "blank_instruction":
        return action in SAFE_DEFER_ACTIONS
    if ptype == "impossible_instruction":
        return action in SAFE_MISSING_ACTIONS
    if ptype == "safety_conflict":
        return action not in {"RETRACT", "TRANSFER", "PICK"}
    if ptype == "negation":
        return action in SAFE_DEFER_ACTIONS
    if ptype == "phase_conflict":
        return not bool((checked.get("checker_flags") or {}).get("phase_state_inconsistent"))
    if gold_action:
        return action == gold_action
    return None


def case_type(semantic_ok: Optional[bool], action_ok: Optional[bool]) -> str:
    if semantic_ok is None or action_ok is None:
        return "unscored"
    if semantic_ok and action_ok:
        return "both_correct"
    if not semantic_ok and not action_ok:
        return "vlm_semantic_failure"
    if semantic_ok and not action_ok:
        return "semantic_to_action_transfer_failure"
    return "action_correct_semantic_wrong"


def aggregate(values: Iterable[Optional[bool]]) -> Dict[str, Any]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "count": 0}
    return {"mean": sum(1.0 if v else 0.0 for v in clean) / len(clean), "count": len(clean)}


def percentage(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def analyze(schema_rows: List[Dict[str, Any]], action_rows: List[Dict[str, Any]], max_cases: int) -> Dict[str, Any]:
    schema_by_id = by_example_id(schema_rows)
    action_by_id = by_example_id(action_rows)
    common_ids = sorted(set(schema_by_id) & set(action_by_id))
    paired_rows: List[Dict[str, Any]] = []
    cases: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for eid in common_ids:
        srow = ensure_checked(schema_by_id[eid])
        arow = ensure_checked(action_by_id[eid])
        sem_ok = schema_correct(srow)
        act_ok = action_correct(arow)
        ctype = case_type(sem_ok, act_ok)
        pair = {
            "example_id": eid,
            "observation_id": srow.get("observation_id"),
            "perturbation_type": srow.get("perturbation_type"),
            "instruction": srow.get("instruction"),
            "schema_method": srow.get("method"),
            "action_method": arow.get("method"),
            "schema_correct": sem_ok,
            "action_correct": act_ok,
            "case_type": ctype,
            "schema_target": (srow.get("parsed_schema") or {}).get("target_object"),
            "action_target": (arow.get("parsed_schema") or {}).get("target_object") or arow.get("action_target_object"),
            "schema_next_action": final_action(srow),
            "action_next_action": final_action(arow),
            "gold_schema": srow.get("gold_schema") or arow.get("gold_schema"),
        }
        paired_rows.append(pair)
        if len(cases[ctype]) < max_cases:
            cases[ctype].append(pair)

    sem = aggregate([r["schema_correct"] for r in paired_rows])
    act = aggregate([r["action_correct"] for r in paired_rows])
    semantic_action_gap = None
    if sem["mean"] is not None and act["mean"] is not None:
        semantic_action_gap = sem["mean"] - act["mean"]

    counts = Counter(r["case_type"] for r in paired_rows)
    total_scored = sum(v for k, v in counts.items() if k != "unscored")
    taxonomy = [
        {"case_type": k, "count": v, "percentage": percentage(v, total_scored if k != "unscored" else len(paired_rows))}
        for k, v in sorted(counts.items())
    ]

    by_perturbation = {}
    for ptype in sorted({str(r.get("perturbation_type")) for r in paired_rows}):
        prows = [r for r in paired_rows if str(r.get("perturbation_type")) == ptype]
        psem = aggregate([r["schema_correct"] for r in prows])
        pact = aggregate([r["action_correct"] for r in prows])
        by_perturbation[ptype] = {
            "schema_sensitivity": psem,
            "action_sensitivity": pact,
            "semantic_action_sensitivity_gap": None
            if psem["mean"] is None or pact["mean"] is None
            else psem["mean"] - pact["mean"],
            "count": len(prows),
        }

    return {
        "num_schema_predictions": len(schema_rows),
        "num_action_predictions": len(action_rows),
        "num_paired_examples": len(paired_rows),
        "schema_sensitivity": sem,
        "action_sensitivity": act,
        "semantic_action_sensitivity_gap": {"mean": semantic_action_gap, "count": min(sem["count"], act["count"])},
        "case_taxonomy": taxonomy,
        "per_perturbation": by_perturbation,
        "paired_rows": paired_rows,
        "qualitative_cases": dict(cases),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema_rows = method_filter(read_jsonl(args.schema_predictions), args.schema_method)
    action_rows = method_filter(read_jsonl(args.action_predictions), args.action_method)
    result = analyze(schema_rows, action_rows, args.max_cases_per_type)

    write_jsonl(out_dir / "semantic_action_pairs.jsonl", result["paired_rows"])
    write_json(out_dir / "semantic_action_gap.json", {k: v for k, v in result.items() if k not in {"paired_rows"}})
    write_json(out_dir / "semantic_action_cases.json", result["qualitative_cases"])
    write_csv(out_dir / "semantic_action_taxonomy.csv", result["case_taxonomy"])
    write_csv(out_dir / "semantic_action_by_perturbation.csv", [
        {"perturbation_type": ptype, **vals}
        for ptype, vals in result["per_perturbation"].items()
    ])
    write_json(out_dir / "run_metadata.json", {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "schema_predictions": args.schema_predictions,
        "action_predictions": args.action_predictions,
        "schema_method": args.schema_method,
        "action_method": args.action_method,
        "python": sys.version,
        "platform": platform.platform(),
    })
    print(json.dumps({
        "out_dir": str(out_dir),
        "num_paired_examples": result["num_paired_examples"],
        "schema_sensitivity": result["schema_sensitivity"],
        "action_sensitivity": result["action_sensitivity"],
        "semantic_action_sensitivity_gap": result["semantic_action_sensitivity_gap"],
    }, indent=2))


if __name__ == "__main__":
    main()
