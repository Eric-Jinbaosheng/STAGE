#!/usr/bin/env python3
"""Prompt-rewrite probe for schema-conditioned OpenVLA actions.

This experiment does not gate or stop execution. It reuses cached Qwen schemas
to rewrite the instruction into a more explicit target/action prompt, then
measures whether OpenVLA's native 7-DoF action changes relative to the raw
counterfactual prompt.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lb_run_exp2_action_sensitivity import (  # noqa: E402
    action_stats,
    load_image_for_example,
    load_index_maps,
    load_model_and_processor,
    predict_action,
)
from linguistic_blindness.utils.io import (  # noqa: E402
    command_string,
    read_jsonl,
    utc_timestamp,
    write_csv,
    write_json,
    write_jsonl,
)


TASK_ACTIONS = {"MOVE_TO", "APPROACH", "ALIGN", "GRASP", "PICK", "PLACE", "OPEN", "CLOSE", "TRANSFER"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp6 prompt rewrite probe using cached Qwen schemas.")
    p.add_argument("--exp2-predictions", required=True, help="Frozen Exp2 predictions with raw original/counterfactual actions.")
    p.add_argument("--qwen-predictions", default="", help="Cached Qwen schema predictions for the same examples.")
    p.add_argument("--index", default="data/processed/unified_index.parquet")
    p.add_argument("--model-path", default="/scratch/bj2410/models/openvla-7b")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-examples", type=int, default=100)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--sample-seed", type=int, default=7)
    p.add_argument("--threshold", type=float, default=2.2851185083448717, help="Exp2 paraphrase-control P95 threshold.")
    p.add_argument("--rewrite-profile", choices=["concise", "schema"], default="concise")
    p.add_argument("--unnorm-key", default="bridge_orig")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--attn-implementation", default="")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--allow-placeholder-image", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=50)
    return p.parse_args()


def by_example_id(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        eid = row.get("example_id")
        if eid:
            out[str(eid)] = row
    return out


def as_action_name(x: Any) -> str:
    return str(x or "").strip().upper()


def safe_list(xs: Any) -> List[str]:
    if not isinstance(xs, list):
        return []
    return [str(x).strip() for x in xs if str(x).strip()]


def choose_action(schema: Dict[str, Any]) -> str:
    next_action = as_action_name(schema.get("next_action"))
    allowed = [as_action_name(x) for x in safe_list(schema.get("allowed_actions"))]
    if next_action in TASK_ACTIONS:
        return next_action
    for action in allowed:
        if action in TASK_ACTIONS:
            return action
    return "MOVE_TO"


def schema_for_row(row: Dict[str, Any], qwen: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    eid = str(row.get("example_id"))
    if eid in qwen:
        return qwen[eid].get("parsed_schema") or {}
    return row.get("qwen_schema_counterfactual") or {}


def rewrite_instruction(row: Dict[str, Any], schema: Dict[str, Any], profile: str) -> str:
    raw_instruction = str(row.get("counterfactual_instruction") or row.get("instruction") or "").strip()
    target = str(schema.get("target_object") or row.get("counterfactual_target_object") or "").strip()
    original_target = str(row.get("original_target_object") or "").strip()
    action = choose_action(schema)
    allowed = [a for a in safe_list(schema.get("allowed_actions")) if as_action_name(a) in TASK_ACTIONS]
    blocked = safe_list(schema.get("blocked_actions"))

    if profile == "schema":
        parts = [
            raw_instruction,
            f"Explicit interaction schema: target_object={target or 'unknown'}; next_action={action}.",
        ]
        if allowed:
            parts.append("Allowed robot actions: " + ", ".join(allowed[:5]) + ".")
        if original_target and target and original_target.lower() != target.lower():
            parts.append(f"The old target was {original_target}; do not interact with {original_target}.")
        elif blocked:
            parts.append("Avoid blocked actions: " + ", ".join(blocked[:5]) + ".")
        parts.append(f"Now take the next robot action for {action.lower()} toward {target or 'the specified target'}.")
        return " ".join(x for x in parts if x).strip()

    # Concise is intentionally closer to the natural OpenVLA instruction format.
    if target:
        rewritten = f"{raw_instruction}. The target object is {target}. Move toward and manipulate only {target}."
    else:
        rewritten = raw_instruction
    if original_target and target and original_target.lower() != target.lower():
        rewritten += f" Do not target {original_target}."
    return rewritten.strip()


def bool_mean(vals: List[bool]) -> float:
    return float(sum(bool(v) for v in vals) / len(vals)) if vals else float("nan")


def float_summary(vals: List[float]) -> Dict[str, Optional[float]]:
    clean = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not clean:
        return {"mean": None, "std": None, "median": None, "p75": None, "p90": None, "p95": None}
    arr = np.asarray(clean, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def summarize(rows: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    raw_sens = [bool(r["raw_counterfactual_sensitive"]) for r in rows]
    rewrite_sens = [bool(r["rewrite_sensitive"]) for r in rows]
    improved = [bool(r["rewrite_improves_over_raw"]) for r in rows]
    return {
        "num_examples": len(rows),
        "threshold": threshold,
        "raw_counterfactual_action_sensitivity": bool_mean(raw_sens),
        "schema_rewrite_action_sensitivity": bool_mean(rewrite_sens),
        "rewrite_sensitivity_gain": bool_mean(rewrite_sens) - bool_mean(raw_sens) if rows else None,
        "rewrite_improves_over_raw_rate": bool_mean(improved),
        "raw_delta_from_original": float_summary([r["raw_cf_delta_from_original"] for r in rows]),
        "rewrite_delta_from_original": float_summary([r["rewrite_delta_from_original"] for r in rows]),
        "rewrite_delta_from_raw_cf": float_summary([r["rewrite_delta_from_raw_cf"] for r in rows]),
    }


def group_summaries(rows: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    out = []
    pairs = sorted({str(r.get("target_pair")) for r in rows})
    for pair in pairs + ["Overall"]:
        group = rows if pair == "Overall" else [r for r in rows if str(r.get("target_pair")) == pair]
        if not group:
            continue
        s = summarize(group, threshold)
        out.append({
            "target_pair": pair,
            "n": len(group),
            "raw_counterfactual_action_sensitivity": s["raw_counterfactual_action_sensitivity"],
            "schema_rewrite_action_sensitivity": s["schema_rewrite_action_sensitivity"],
            "rewrite_sensitivity_gain": s["rewrite_sensitivity_gain"],
            "rewrite_improves_over_raw_rate": s["rewrite_improves_over_raw_rate"],
            "raw_mean_delta": s["raw_delta_from_original"]["mean"],
            "rewrite_mean_delta": s["rewrite_delta_from_original"]["mean"],
            "rewrite_vs_raw_mean_delta": s["rewrite_delta_from_raw_cf"]["mean"],
        })
    return out


def latex_table(rows: List[Dict[str, Any]]) -> str:
    def fmt(x: Any) -> str:
        if x is None:
            return "NA"
        try:
            return f"{float(x):.3f}"
        except Exception:
            return str(x)

    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "Target Pair & N & Raw Sens. & Rewrite Sens. & Gain & Rewrite Improves & Rewrite $\\Delta$ \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['target_pair']} & {row['n']} & {fmt(row['raw_counterfactual_action_sensitivity'])} "
            f"& {fmt(row['schema_rewrite_action_sensitivity'])} & {fmt(row['rewrite_sensitivity_gain'])} "
            f"& {fmt(row['rewrite_improves_over_raw_rate'])} & {fmt(row['rewrite_mean_delta'])} \\\\"
        )
    lines.append("\\end{tabular}\n")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.exp2_predictions)
    qwen = by_example_id(read_jsonl(args.qwen_predictions)) if args.qwen_predictions else {}
    rows = [r for r in rows if r.get("perturbation_type") == "target_swap"]
    if args.shuffle:
        random.Random(args.sample_seed).shuffle(rows)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]

    obs_map, action_std = load_index_maps(args.index)
    model, processor = load_model_and_processor(args)

    out_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        schema = schema_for_row(row, qwen)
        rewritten = rewrite_instruction(row, schema, args.rewrite_profile)
        image_row = dict(row)
        if image_row.get("image_source") and not image_row.get("obs_ptr"):
            image_row["obs_ptr"] = image_row["image_source"]
        image, image_source, placeholder = load_image_for_example(image_row, obs_map, args.allow_placeholder_image)

        action_orig = row.get("openvla_action_original")
        action_cf = row.get("openvla_action_counterfactual")
        if not action_orig or not action_cf:
            raise ValueError(f"Exp2 row lacks cached OpenVLA actions: {row.get('example_id')}")
        action_rewrite = predict_action(model, processor, image, rewritten, args)

        raw_stats = action_stats(action_orig, action_cf, action_std)
        rewrite_stats = action_stats(action_orig, action_rewrite, action_std)
        rewrite_vs_raw = action_stats(action_cf, action_rewrite, action_std)

        raw_delta = float(raw_stats["normalized_action_delta"])
        rewrite_delta = float(rewrite_stats["normalized_action_delta"])
        raw_sensitive = raw_delta >= args.threshold
        rewrite_sensitive = rewrite_delta >= args.threshold
        target_pair = f"{row.get('original_target_object')} -> {row.get('counterfactual_target_object')}"

        out = {
            "example_id": row.get("example_id"),
            "observation_id": row.get("observation_id"),
            "target_pair": target_pair,
            "original_instruction": row.get("original_instruction"),
            "counterfactual_instruction": row.get("counterfactual_instruction"),
            "rewritten_instruction": rewritten,
            "rewrite_profile": args.rewrite_profile,
            "qwen_schema_counterfactual": schema,
            "qwen_target_counterfactual": schema.get("target_object"),
            "image_source": image_source,
            "placeholder_image": placeholder,
            "openvla_action_original": action_orig,
            "openvla_action_counterfactual": action_cf,
            "openvla_action_schema_rewrite": action_rewrite,
            "raw_cf_delta_from_original": raw_delta,
            "rewrite_delta_from_original": rewrite_delta,
            "rewrite_delta_from_raw_cf": float(rewrite_vs_raw["normalized_action_delta"]),
            "raw_counterfactual_sensitive": bool(raw_sensitive),
            "rewrite_sensitive": bool(rewrite_sensitive),
            "rewrite_improves_over_raw": bool(rewrite_delta > raw_delta),
            "rewrite_crosses_threshold_when_raw_does_not": bool((not raw_sensitive) and rewrite_sensitive),
            "raw_delta_pos": raw_stats["delta_pos"],
            "raw_delta_rot": raw_stats["delta_rot"],
            "raw_delta_gripper": raw_stats["delta_gripper"],
            "rewrite_delta_pos": rewrite_stats["delta_pos"],
            "rewrite_delta_rot": rewrite_stats["delta_rot"],
            "rewrite_delta_gripper": rewrite_stats["delta_gripper"],
            "rewrite_action_cosine_to_original": rewrite_stats["action_cosine"],
            "rewrite_action_cosine_to_raw_cf": rewrite_vs_raw["action_cosine"],
            "action_sensitivity_threshold": args.threshold,
        }
        out_rows.append(out)

        if args.log_every > 0 and (i % args.log_every == 0 or i == len(rows)):
            print(f"progress {i}/{len(rows)}", flush=True)
        if args.save_every > 0 and i % args.save_every == 0:
            write_jsonl(out_dir / "predictions.partial.jsonl", out_rows)

    summary = summarize(out_rows, args.threshold)
    pair_rows = group_summaries(out_rows, args.threshold)
    write_jsonl(out_dir / "predictions.jsonl", out_rows)
    write_json(out_dir / "metrics.json", summary)
    write_csv(out_dir / "target_pair_summary.csv", pair_rows)
    (out_dir / "latex_table_exp6_prompt_rewrite.tex").write_text(latex_table(pair_rows), encoding="utf-8")
    write_json(out_dir / "run_metadata.json", {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "exp2_predictions": args.exp2_predictions,
        "qwen_predictions": args.qwen_predictions,
        "model_path": args.model_path,
        "unnorm_key": args.unnorm_key,
        "rewrite_profile": args.rewrite_profile,
        "threshold": args.threshold,
        "note": "This is prompt rewrite, not hard gating. It reuses cached Qwen schemas and only reruns OpenVLA on rewritten prompts.",
        "python": sys.version,
        "platform": platform.platform(),
    })
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
