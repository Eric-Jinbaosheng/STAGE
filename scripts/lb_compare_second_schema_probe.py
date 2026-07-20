#!/usr/bin/env python3
"""Compare a second schema probe against frozen Exp2 OpenVLA actions.

This is post-hoc: it reads a schema-probe prediction file and Exp2 action
sensitivity outputs, aligns by example_id, and reports whether the second probe
also shows schema-level target sensitivity while OpenVLA remains action
insensitive.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from linguistic_blindness.utils.io import command_string, ensure_dir, read_jsonl, utc_timestamp, write_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare second schema probe with Exp2 action sensitivity.")
    p.add_argument("--second-schema", required=True, help="Second probe predictions.jsonl")
    p.add_argument("--exp2-predictions", default="outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/predictions.jsonl")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--probe-name", default="second_schema_probe")
    p.add_argument("--rounds", type=int, default=5000)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


def norm(x: Any) -> str:
    return str(x or "").strip().lower().replace("_", " ")


def target_from_schema(row: Dict[str, Any]) -> str:
    schema = row.get("parsed_schema") or row.get("schema") or row.get("qwen_schema_counterfactual") or {}
    return norm(schema.get("target_object"))


def gold_target(row: Dict[str, Any], exp2_row: Dict[str, Any] | None = None) -> str:
    gs = row.get("gold_schema") or {}
    if gs.get("target_object"):
        return norm(gs.get("target_object"))
    if exp2_row and exp2_row.get("counterfactual_target_object"):
        return norm(exp2_row.get("counterfactual_target_object"))
    return ""


def original_target(row: Dict[str, Any], exp2_row: Dict[str, Any] | None = None) -> str:
    os = row.get("original_schema") or {}
    if os.get("target_object"):
        return norm(os.get("target_object"))
    if exp2_row and exp2_row.get("original_target_object"):
        return norm(exp2_row.get("original_target_object"))
    return ""


def bootstrap_mean(values: Sequence[float], seed: int, rounds: int) -> tuple[float, float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(vals)
    mean = sum(vals) / n
    means = []
    for _ in range(rounds):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return mean, means[int(0.025 * (rounds - 1))], means[int(0.975 * (rounds - 1))]


def metric_row(metric: str, values: Sequence[float], seed: int, rounds: int, direction: str) -> Dict[str, Any]:
    mean, lo, hi = bootstrap_mean(values, seed, rounds)
    return {
        "metric": metric,
        "direction": direction,
        "n": len(values),
        "mean": mean,
        "ci95_low": lo,
        "ci95_high": hi,
        "ci95": f"{mean:.3f} [{lo:.3f}, {hi:.3f}]",
    }


def latex(rows: List[Dict[str, Any]], probe_name: str) -> str:
    lines = [
        "\\begin{tabular}{lr}",
        "\\toprule",
        f"Metric ({probe_name}) & Mean [95\\% CI] \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(f"{r['metric']} & {r['ci95']} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    exp2 = {r["example_id"]: r for r in read_jsonl(args.exp2_predictions)}
    second = read_jsonl(args.second_schema)

    aligned = []
    missing_exp2 = 0
    for r in second:
        e = exp2.get(r.get("example_id"))
        if not e:
            missing_exp2 += 1
            continue
        pred_t = target_from_schema(r)
        gold_t = gold_target(r, e)
        orig_t = original_target(r, e)
        target_correct = bool(gold_t and pred_t == gold_t)
        target_not_fixed = bool(orig_t and gold_t and orig_t != gold_t and pred_t != orig_t)
        schema_sensitive = bool(target_correct and target_not_fixed)
        openvla_sensitive = bool(e.get("openvla_action_sensitive"))
        gap_case = bool(schema_sensitive and not openvla_sensitive)
        aligned.append({
            "example_id": r.get("example_id"),
            "observation_id": r.get("observation_id"),
            "original_instruction": r.get("original_instruction"),
            "counterfactual_instruction": r.get("counterfactual_instruction"),
            "original_target": orig_t,
            "counterfactual_target": gold_t,
            "second_probe_target": pred_t,
            "second_probe_target_correct": target_correct,
            "second_probe_schema_sensitive": schema_sensitive,
            "openvla_action_sensitive": openvla_sensitive,
            "semantic_action_gap_case": gap_case,
            "normalized_action_delta": e.get("normalized_action_delta"),
            "openvla_action_sensitivity_threshold": e.get("action_sensitivity_threshold"),
            "raw_model_output": r.get("raw_model_output"),
            "parsed_schema": r.get("parsed_schema"),
        })

    metrics = [
        metric_row("Second-probe target correctness", [x["second_probe_target_correct"] for x in aligned], args.seed, args.rounds, "up"),
        metric_row("Second-probe schema sensitivity", [x["second_probe_schema_sensitive"] for x in aligned], args.seed + 1, args.rounds, "up"),
        metric_row("OpenVLA action sensitivity", [x["openvla_action_sensitive"] for x in aligned], args.seed + 2, args.rounds, "up"),
        metric_row("Semantic-action gap case rate", [x["semantic_action_gap_case"] for x in aligned], args.seed + 3, args.rounds, "down"),
    ]
    if aligned:
        schema_sens = sum(x["second_probe_schema_sensitive"] for x in aligned) / len(aligned)
        action_sens = sum(x["openvla_action_sensitive"] for x in aligned) / len(aligned)
    else:
        schema_sens = action_sens = 0.0
    metrics.append({
        "metric": "Aggregate schema-action gap",
        "direction": "down",
        "n": len(aligned),
        "mean": schema_sens - action_sens,
        "ci95_low": None,
        "ci95_high": None,
        "ci95": f"{schema_sens - action_sens:.3f}",
    })

    pairs: Dict[str, List[Dict[str, Any]]] = {}
    for x in aligned:
        pairs.setdefault(f"{x['original_target']} -> {x['counterfactual_target']}", []).append(x)
    pair_rows = []
    for pair, xs in sorted(pairs.items()):
        n = len(xs)
        pair_rows.append({
            "target_pair": pair,
            "n": n,
            "second_probe_schema_sensitivity": sum(x["second_probe_schema_sensitive"] for x in xs) / n,
            "openvla_action_sensitivity": sum(x["openvla_action_sensitive"] for x in xs) / n,
            "semantic_action_gap_case_rate": sum(x["semantic_action_gap_case"] for x in xs) / n,
        })

    target_counts = Counter(x["second_probe_target"] for x in aligned)
    write_jsonl(out_dir / "aligned_predictions.jsonl", aligned)
    write_csv(out_dir / "second_probe_metrics.csv", metrics)
    write_json(out_dir / "second_probe_metrics.json", {"probe_name": args.probe_name, "rows": metrics})
    write_csv(out_dir / "target_pair_breakdown.csv", pair_rows)
    (out_dir / "latex_table_second_schema_probe.tex").write_text(latex(metrics, args.probe_name), encoding="utf-8")
    write_json(out_dir / "run_metadata.json", {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "probe_name": args.probe_name,
        "second_schema": args.second_schema,
        "exp2_predictions": args.exp2_predictions,
        "num_second_rows": len(second),
        "num_aligned": len(aligned),
        "missing_exp2": missing_exp2,
        "second_probe_target_distribution": dict(target_counts),
    })
    print(json.dumps({"out_dir": str(out_dir), "num_aligned": len(aligned), "missing_exp2": missing_exp2}, indent=2))


if __name__ == "__main__":
    main()
