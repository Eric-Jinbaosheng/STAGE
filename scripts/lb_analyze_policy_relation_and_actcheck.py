#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def rate(values: list[Any]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(bool(v) for v in vals) / len(vals)


def mean(values: list[Any]) -> float | None:
    vals: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(x):
            vals.append(x)
    if not vals:
        return None
    return sum(vals) / len(vals)


def vec3(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)[:3]


def unit(v: np.ndarray) -> np.ndarray | None:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    ua, ub = unit(a), unit(b)
    if ua is None or ub is None:
        return None
    return float(np.dot(ua, ub))


def summarize(rows: list[dict[str, Any]], subset: str, policy: str) -> dict[str, Any]:
    schema_sens = rate([r["target_correct"] for r in rows])
    action_sens = rate([r["action_sensitive"] for r in rows])
    return {
        "policy": policy,
        "subset": subset,
        "n": len(rows),
        "grounding_accuracy": schema_sens,
        "schema_sensitivity": schema_sens,
        "action_sensitivity": action_sens,
        "semantic_action_gap": (schema_sens or 0.0) - (action_sens or 0.0),
        "low_sensitivity_rate": rate([r["low_action_sensitivity"] for r in rows]),
        "mean_normalized_action_delta": mean([r["normalized_action_delta"] for r in rows]),
        "mean_target_angle_deg": mean([r["target_angle_deg"] for r in rows]),
        "target_aligned_action_rate": rate([r["cf_target_aligned_action"] for r in rows]),
        "wrong_target_action_rate": rate([r["cf_wrong_target_aligned_action"] for r in rows]),
        "ambiguous_alignment_rate": rate([r["cf_ambiguous_target_alignment"] for r in rows]),
        "actcheck_block_rate": rate([r["actcheck_blocks_cf_action"] for r in rows]),
        "valid_original_action_preservation": rate([r["orig_target_aligned_action"] for r in rows]),
        "mean_semantic_action_consistency_score": mean(
            [r["semantic_action_consistency_score"] for r in rows]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--policy-predictions", required=True)
    parser.add_argument("--qwen-predictions", required=True)
    parser.add_argument("--policy", default="octo")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--alignment-margin", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark = {r["example_id"]: r for r in read_jsonl(args.benchmark)}
    qwen = {r["example_id"]: r for r in read_jsonl(args.qwen_predictions)}
    policy_rows = read_jsonl(args.policy_predictions)

    action_orig_key = f"{args.policy}_action_original"
    action_cf_key = f"{args.policy}_action_counterfactual"
    action_sensitive_key = f"{args.policy}_action_sensitive"

    rows: list[dict[str, Any]] = []
    for pred in policy_rows:
        ex_id = pred["example_id"]
        if ex_id not in benchmark:
            continue
        bench = benchmark[ex_id]
        q = qwen.get(ex_id, {})
        state = bench.get("observation_state") or {}
        if not all(k in state for k in ["ee_pos", "original_target_pos", "counterfactual_target_pos"]):
            continue

        ee = vec3(state["ee_pos"])
        pos_orig = vec3(state["original_target_pos"])
        pos_cf = vec3(state["counterfactual_target_pos"])
        action_orig = vec3(pred[action_orig_key])
        action_cf = vec3(pred[action_cf_key])

        orig_to_orig = cosine(action_orig, pos_orig - ee)
        orig_to_cf = cosine(action_orig, pos_cf - ee)
        cf_to_cf = cosine(action_cf, pos_cf - ee)
        cf_to_orig = cosine(action_cf, pos_orig - ee)

        margin = args.alignment_margin
        orig_aligned = (
            orig_to_orig is not None and orig_to_cf is not None and orig_to_orig > orig_to_cf + margin
        )
        cf_aligned = (
            cf_to_cf is not None and cf_to_orig is not None and cf_to_cf > cf_to_orig + margin
        )
        wrong_target = (
            cf_to_orig is not None and cf_to_cf is not None and cf_to_orig >= cf_to_cf + margin
        )
        ambiguous = (
            cf_to_orig is not None and cf_to_cf is not None and abs(cf_to_cf - cf_to_orig) <= margin
        )

        target_correct = bool(q.get("target_correct", pred.get("qwen_schema_sensitive", True)))
        rows.append(
            {
                "example_id": ex_id,
                "policy": args.policy,
                "target_pair": f"{bench.get('original_target_object')} -> {bench.get('counterfactual_target_object')}",
                "original_target_object": bench.get("original_target_object"),
                "counterfactual_target_object": bench.get("counterfactual_target_object"),
                "target_correct": target_correct,
                "pred_target": (q.get("parsed_schema") or {}).get("target_object"),
                "gold_target": bench.get("counterfactual_target_object"),
                "target_distance": bench.get("target_distance"),
                "target_angle_deg": bench.get("target_angle_deg"),
                "action_sensitive": bool(pred.get(action_sensitive_key)),
                "low_action_sensitivity": bool(pred.get("low_action_sensitivity")),
                "normalized_action_delta": pred.get("normalized_action_delta"),
                "orig_action_cos_to_orig_target": orig_to_orig,
                "orig_action_cos_to_cf_target": orig_to_cf,
                "cf_action_cos_to_cf_target": cf_to_cf,
                "cf_action_cos_to_orig_target": cf_to_orig,
                "orig_target_aligned_action": orig_aligned,
                "cf_target_aligned_action": cf_aligned,
                "cf_wrong_target_aligned_action": wrong_target,
                "cf_ambiguous_target_alignment": ambiguous,
                "actcheck_blocks_cf_action": wrong_target or ambiguous,
                "semantic_action_consistency_score": None
                if cf_to_cf is None or cf_to_orig is None
                else cf_to_cf - cf_to_orig,
            }
        )

    summaries: list[dict[str, Any]] = [
        summarize(rows, "all", args.policy),
        summarize([r for r in rows if r["target_correct"]], "grounding_correct_only", args.policy),
        summarize([r for r in rows if (r.get("target_angle_deg") or 0) >= 45], "hard_angle_ge_45", args.policy),
        summarize(
            [r for r in rows if r["target_correct"] and (r.get("target_angle_deg") or 0) >= 45],
            "grounding_correct_and_angle_ge_45",
            args.policy,
        ),
    ]
    for pair in sorted({r["target_pair"] for r in rows}):
        summaries.append(summarize([r for r in rows if r["target_pair"] == pair], pair, args.policy))

    write_jsonl(out_dir / "relation_actcheck_predictions.jsonl", rows)
    write_csv(out_dir / "relation_actcheck_summary.csv", summaries)

    md = [
        f"# {args.policy} Relation + ActCheck",
        "",
        "| Subset | N | Grounding acc | Action sens | SAG | Target-aligned | Wrong-target | Ambiguous | ActCheck flag |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        md.append(
            f"| {s['subset']} | {s['n']} | {s['grounding_accuracy']:.3f} | "
            f"{s['action_sensitivity']:.3f} | {s['semantic_action_gap']:.3f} | "
            f"{s['target_aligned_action_rate']:.3f} | {s['wrong_target_action_rate']:.3f} | "
            f"{s['ambiguous_alignment_rate']:.3f} | {s['actcheck_block_rate']:.3f} |"
        )
    (out_dir / "relation_actcheck_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "overall": summaries[0]}, indent=2))


if __name__ == "__main__":
    main()
