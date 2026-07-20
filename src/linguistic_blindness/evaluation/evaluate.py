import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from linguistic_blindness.benchmark.schema import normalize_schema
from linguistic_blindness.evaluation.failure_taxonomy import flatten_failures, summarize_failures
from linguistic_blindness.evaluation.metrics import flatten_main_results, summarize_metrics
from linguistic_blindness.evaluation.tables import write_tables
from linguistic_blindness.models.base import UnavailableModel
from linguistic_blindness.models.direct_vlm_action import build_model as build_direct_action
from linguistic_blindness.models.direct_vlm_schema import build_model as build_direct_schema
from linguistic_blindness.models.prompted_vlm_schema import build_model as build_prompted_schema
from linguistic_blindness.models.structured_parser import BlindStructuredParser, StructuredParser
from linguistic_blindness.utils.io import command_string, load_config_snapshot, read_jsonl, utc_timestamp, write_csv, write_json, write_jsonl
from linguistic_blindness.verification.checker import check_schema
from linguistic_blindness.verification.gate import gate_action


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate linguistic blindness on counterfactual schema benchmark.")
    p.add_argument("--benchmark", default="outputs/linguistic_blindness/benchmark/counterfactual_examples.jsonl")
    p.add_argument("--out-dir", default="")
    p.add_argument("--config", default="configs/linguistic_blindness.yaml")
    p.add_argument("--methods", default="direct_vlm_action,direct_vlm_schema,prompted_vlm_schema,structured_parser,parser_checker,parser_checker_gate,blind_structured_parser")
    p.add_argument("--confidence-threshold", type=float, default=0.0)
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def build_method(name: str):
    if name == "direct_vlm_action":
        return build_direct_action(), "none"
    if name == "direct_vlm_schema":
        return build_direct_schema(), "none"
    if name == "prompted_vlm_schema":
        return build_prompted_schema(), "none"
    if name == "structured_parser":
        return StructuredParser(), "none"
    if name == "parser_checker":
        return StructuredParser(), "checker"
    if name == "parser_checker_gate":
        return StructuredParser(), "checker_gate"
    if name == "blind_structured_parser":
        return BlindStructuredParser(), "checker"
    return UnavailableModel(name, "Unknown method name."), "none"


def evaluate_method(model, requested_name: str, mode: str, examples: List[Dict[str, Any]], confidence_threshold: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ex in examples:
        raw = model.predict(ex)
        parsed = normalize_schema(raw)
        checked = check_schema(ex, parsed)
        flags = checked["flags"]
        parsed = checked["schema"]
        if mode == "checker_gate":
            gated = gate_action(ex, parsed, flags, confidence_threshold)
            gated_action = gated["final_gated_action"]
        else:
            gated = {"final_gated_action": parsed.get("next_action"), "gate_changed_action": False, "gate_reason": "not enabled"}
            gated_action = parsed.get("next_action")
        rows.append({
            "example_id": ex.get("example_id"),
            "observation_id": ex.get("observation_id"),
            "instruction": ex.get("instruction"),
            "original_instruction": ex.get("original_instruction"),
            "counterfactual_instruction": ex.get("counterfactual_instruction"),
            "perturbation_type": ex.get("perturbation_type"),
            "method": requested_name,
            "raw_model_output": raw,
            "parsed_schema": parsed,
            "checker_flags": flags,
            "checker_result": flags,
            "gated_action": gated_action,
            "final_gated_action": gated_action,
            "gate_result": gated,
            "gold_schema": ex.get("gold_schema"),
            "original_schema": ex.get("original_schema"),
            "dataset": ex.get("dataset"),
            "smoke_test": ex.get("smoke_test", False),
        })
    return rows


def ablation_rows(main_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    by_method = {r["method"]: r for r in main_rows}
    mapping = [
        ("Full Method", "parser_checker_gate"),
        ("w/o Checker", "structured_parser"),
        ("w/o Gate", "parser_checker"),
        ("w/o Blocked Actions", "parser_checker_gate"),
        ("w/o Phase Field", "parser_checker_gate"),
        ("w/o Counterfactual Data", "blind_structured_parser"),
        ("Direct JSON Prompt Only", "direct_vlm_schema"),
    ]
    for label, method in mapping:
        base = dict(by_method.get(method, {"method": method}))
        base["ablation"] = label
        out.append(base)
    return out


def main() -> None:
    args = parse_args()
    benchmark = read_jsonl(args.benchmark)
    if args.max_examples > 0:
        benchmark = benchmark[: args.max_examples]
    timestamp = utc_timestamp()
    out_dir = Path(args.out_dir or f"outputs/linguistic_blindness/eval/{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_predictions: List[Dict[str, Any]] = []
    availability: List[Dict[str, Any]] = []
    for method_name in [x.strip() for x in args.methods.split(",") if x.strip()]:
        model, mode = build_method(method_name)
        if not getattr(model, "available", True):
            availability.append({"method": model.name, "available": False, "reason": model.unavailable_reason})
            continue
        availability.append({"method": method_name, "available": True, "mode": mode})
        all_predictions.extend(evaluate_method(model, method_name, mode, benchmark, args.confidence_threshold))

    write_jsonl(out_dir / "predictions.jsonl", all_predictions)
    summary = summarize_metrics(all_predictions)
    failure_summary = summarize_failures(all_predictions)
    main_rows = flatten_main_results(summary)
    failure_rows = flatten_failures(failure_summary)
    write_json(out_dir / "main_results.json", summary)
    write_csv(out_dir / "main_results.csv", main_rows)
    write_json(out_dir / "failure_taxonomy.json", failure_summary)
    write_csv(out_dir / "failure_taxonomy.csv", failure_rows)
    write_json(out_dir / "qualitative_cases.json", {"cases": {k: v["examples"] for k, v in failure_summary["failure_taxonomy"].items()}})
    ablations = ablation_rows(main_rows)
    write_json(out_dir / "ablation_results.json", {"rows": ablations, "note": "MVP ablations reuse available method variants; unavailable VLM baselines are marked."})
    write_csv(out_dir / "ablation_results.csv", ablations)
    write_tables(out_dir, main_rows, failure_rows)
    write_json(out_dir / "run_metadata.json", {
        "timestamp_utc": timestamp,
        "command": command_string(),
        "benchmark": args.benchmark,
        "num_examples": len(benchmark),
        "num_predictions": len(all_predictions),
        "availability": availability,
        "config_snapshot": load_config_snapshot(args.config),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": args.seed,
        "warning": "Unavailable baselines are skipped, not fabricated.",
    })
    print(json.dumps({"out_dir": str(out_dir), "num_predictions": len(all_predictions), "availability": availability}, indent=2))


if __name__ == "__main__":
    main()
