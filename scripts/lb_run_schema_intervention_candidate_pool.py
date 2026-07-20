#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in [REPO_ROOT / "src", REPO_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lb_run_exp2_action_sensitivity import load_image_for_example, load_index_maps, load_model_and_processor, predict_action
from lb_run_visa_rerank_action_probe import actcheck, has_position_fields, row_pos, vec3
from linguistic_blindness.utils.io import command_string, read_jsonl, utc_timestamp, write_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Schema intervention candidate-pool experiment before reranking.")
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--index", default="data/processed/unified_index.parquet")
    ap.add_argument("--model-path", default="/scratch/bj2410/models/openvla-7b")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-examples", type=int, default=120)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--sample-seed", type=int, default=13)
    ap.add_argument("--split-filter", default="", help="Optional comma-separated satbenchpp_split values to keep.")
    ap.add_argument("--family-filter", default="", help="Optional comma-separated satbenchpp_family values to keep.")
    ap.add_argument("--budgets", default="1,2,4,8,16")
    ap.add_argument("--conditions", default="generic,correct_schema,target_swapped_schema,shuffled_schema,field_dropped_schema")
    ap.add_argument("--alignment-margin", type=float, default=0.05)
    ap.add_argument("--unnorm-key", default="bridge_orig")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--attn-implementation", default="")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--allow-placeholder-image", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=20)
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


def get_schema_target(row: Dict[str, Any], source: str, shuffled_row: Optional[Dict[str, Any]] = None) -> str:
    if source == "field_dropped_schema":
        return ""
    if source == "target_swapped_schema":
        return str(row.get("original_target_object") or "").strip()
    src = shuffled_row if source == "shuffled_schema" and shuffled_row is not None else row
    target = (
        src.get("qwen_target_counterfactual")
        or (src.get("qwen_schema_counterfactual") or {}).get("target_object")
        or (src.get("gold_schema") or {}).get("target_object")
        or src.get("counterfactual_target_object")
        or src.get("target_object")
        or ""
    )
    return str(target).strip()


def get_next_action(row: Dict[str, Any]) -> str:
    schema = row.get("qwen_schema_counterfactual") or row.get("gold_schema") or {}
    return str(schema.get("next_action") or schema.get("next_symbolic_action") or "move_to_target").strip()


def get_relation(row: Dict[str, Any]) -> str:
    schema = row.get("qwen_schema_counterfactual") or row.get("gold_schema") or {}
    return str(schema.get("relation") or row.get("relation_changed") or row.get("perturbation_type") or "").strip()


def get_anchor(row: Dict[str, Any]) -> str:
    schema = row.get("qwen_schema_counterfactual") or row.get("gold_schema") or {}
    return str(schema.get("anchor_object") or row.get("anchor_object") or "").strip()


def schema_prompt_variants(row: Dict[str, Any], target: str, condition: str) -> List[Tuple[str, str]]:
    raw = str(row.get("counterfactual_instruction") or row.get("instruction") or "").strip()
    old = str(row.get("original_target_object") or "").strip()
    target_phrase = target or "the task-specified object"
    next_action = get_next_action(row) or "move_to_target"
    relation = get_relation(row)
    anchor = get_anchor(row)
    relation_clause = f" relation={relation}" if relation else ""
    anchor_clause = f" anchor={anchor}" if anchor else ""
    if condition == "generic":
        prompts = [
            raw,
            "do the task",
            "follow the instruction",
            "move toward the object",
            "pick up the object",
            "complete the manipulation task",
            "move to the relevant object",
            "execute the next robot action",
            "approach the object mentioned by the instruction",
            "manipulate the object",
            "reach toward the task object",
            "perform the requested manipulation",
            "continue the task",
            "take the next low-level action",
            "move the end effector toward the correct object",
            "act according to the language command",
        ]
    elif condition == "field_dropped_schema":
        prompts = [
            raw,
            f"schema fields: target=<dropped>; next_action={next_action}{relation_clause}{anchor_clause}. Execute the next action.",
            f"target field is unavailable. Use relation/context only{relation_clause}{anchor_clause}.",
            f"next robot action: {next_action}. The target field is omitted.",
            "use the recovered schema, but the target identity is missing",
            "move toward the object specified by the non-target schema fields",
            "execute the task with target identity hidden",
            "follow the instruction using only action constraints",
            "the schema omits target_object; choose the next safe motion",
            "target_object is unknown; infer from context and move",
            "perform next_symbolic_action without explicit target",
            "use relation and anchor fields only",
            "continue using the procedural constraint, target dropped",
            "act on the schema while ignoring explicit object names",
            "take a low-level action from schema without target field",
            "move toward the most likely task object",
        ]
    else:
        prompts = [
            raw,
            f"target object: {target_phrase}. Move toward it.",
            f"move toward the {target_phrase}",
            f"pick up the {target_phrase}",
            f"{raw}. The correct target is {target_phrase}.",
            f"schema target_object={target_phrase}; next_action={next_action}.",
            f"next robot action: {next_action} toward {target_phrase}.",
            f"for the next low-level action, move the end effector toward {target_phrase}.",
            f"ignore {old}; target {target_phrase}.",
            f"the object to interact with is {target_phrase}.",
            f"approach {target_phrase} before any other object.",
            f"execute only the first subgoal: move to {target_phrase}.",
            f"schema fields: target={target_phrase}; action={next_action}{relation_clause}{anchor_clause}.",
            f"robot hand should go toward {target_phrase}.",
            f"prepare to grasp or interact with {target_phrase}.",
            f"select the action whose endpoint moves closer to {target_phrase}.",
        ]
    out: List[Tuple[str, str]] = []
    seen = set()
    for i, p in enumerate(prompts, start=1):
        pp = " ".join(str(p).split())
        if pp and pp not in seen:
            out.append((f"{condition}_{i:02d}", pp))
            seen.add(pp)
    return out


def unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return None
    return v / n


def cos(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    ua, ub = unit(a), unit(b)
    if ua is None or ub is None:
        return None
    return float(np.dot(ua, ub))


def schema_target_actcheck(action: List[float], row: Dict[str, Any], schema_target: str, margin: float) -> Dict[str, Any]:
    cf = str(row.get("counterfactual_target_object") or row.get("target_object") or "")
    orig = str(row.get("original_target_object") or "")
    if not schema_target:
        return {"schema_target_known_in_scene": False}
    if schema_target == cf:
        pos_key = "counterfactual_target_pos"
    elif schema_target == orig:
        pos_key = "original_target_pos"
    else:
        return {"schema_target_known_in_scene": False}
    ee = vec3(row_pos(row, "ee_pos"))
    ps = vec3(row_pos(row, pos_key))
    pc = vec3(row_pos(row, "counterfactual_target_pos"))
    a = vec3(action)
    c_schema = cos(a, ps - ee)
    c_correct = cos(a, pc - ee)
    score = None if c_schema is None or c_correct is None else c_schema - c_correct
    return {
        "schema_target_known_in_scene": True,
        "cos_to_schema_target": c_schema,
        "schema_minus_correct_margin": score,
        "aligned_to_schema_target_over_correct": bool(score is not None and score > margin),
    }


def keep_by_filter(row: Dict[str, Any], split_filter: set[str], family_filter: set[str]) -> bool:
    if split_filter and str(row.get("satbenchpp_split") or "target_swap") not in split_filter:
        return False
    if family_filter and str(row.get("satbenchpp_family") or "target_swap") not in family_filter:
        return False
    return True


def select_rows(
    rows: List[Dict[str, Any]],
    max_examples: int,
    seed: int,
    shuffle: bool,
    split_filter: set[str],
    family_filter: set[str],
) -> List[Dict[str, Any]]:
    rows = [r for r in rows if r.get("position_extraction_status") in {None, "ok"} and has_position_fields(r)]
    rows = [r for r in rows if keep_by_filter(r, split_filter, family_filter)]
    if shuffle:
        random.Random(seed).shuffle(rows)
    if max_examples > 0:
        rows = rows[:max_examples]
    return rows


def summarize(out_rows: List[Dict[str, Any]], budgets: List[int]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    groups: List[Tuple[str, str, List[Dict[str, Any]]]] = [("overall", "overall", out_rows)]
    for split in sorted({str(r.get("satbenchpp_split") or "target_swap") for r in out_rows}):
        groups.append(("split", split, [r for r in out_rows if str(r.get("satbenchpp_split") or "target_swap") == split]))
    for fam in sorted({str(r.get("satbenchpp_family") or "target_swap") for r in out_rows}):
        groups.append(("family", fam, [r for r in out_rows if str(r.get("satbenchpp_family") or "target_swap") == fam]))

    for group_type, group, rows in groups:
        conditions = sorted({c["condition"] for r in rows for c in r.get("condition_candidates", [])})
        for condition in conditions:
            for n in budgets:
                per = []
                for r in rows:
                    block = next((c for c in r.get("condition_candidates", []) if c["condition"] == condition), None)
                    if not block:
                        continue
                    cands = block.get("candidates", [])[:n]
                    if not cands:
                        continue
                    best = max(cands, key=lambda c: -1e9 if finite((c.get("actcheck") or {}).get("semantic_action_consistency_score")) is None else float((c.get("actcheck") or {}).get("semantic_action_consistency_score")))
                    per.append({
                        "realized": len(cands),
                        "any_correct": any((c.get("actcheck") or {}).get("target_aligned") for c in cands),
                        "best_correct": bool((best.get("actcheck") or {}).get("target_aligned")),
                        "best_margin": (best.get("actcheck") or {}).get("semantic_action_consistency_score"),
                        "mean_margin": mean((c.get("actcheck") or {}).get("semantic_action_consistency_score") for c in cands),
                        "wrong_rate": rate((c.get("actcheck") or {}).get("wrong_target") or (c.get("actcheck") or {}).get("ambiguous") for c in cands),
                        "any_schema": any((c.get("schema_target_actcheck") or {}).get("aligned_to_schema_target_over_correct") for c in cands if (c.get("schema_target_actcheck") or {}).get("schema_target_known_in_scene")),
                        "schema_known": any((c.get("schema_target_actcheck") or {}).get("schema_target_known_in_scene") for c in cands),
                    })
                result.append({
                    "group_type": group_type,
                    "group": group,
                    "condition": condition,
                    "budget": n,
                    "n_examples": len(per),
                    "mean_realized_candidates": mean(p["realized"] for p in per),
                    "candidate_recall_at_n": rate(p["any_correct"] for p in per),
                    "best_candidate_aligned_rate": rate(p["best_correct"] for p in per),
                    "mean_best_target_margin": mean(p["best_margin"] for p in per),
                    "mean_candidate_target_margin": mean(p["mean_margin"] for p in per),
                    "mean_wrong_or_ambiguous_candidate_rate": mean(p["wrong_rate"] for p in per),
                    "schema_target_known_rate": rate(p["schema_known"] for p in per),
                    "schema_target_redirection_rate": rate(p["any_schema"] for p in per if p["schema_known"]),
                })
    return result


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    split_filter = {x.strip() for x in args.split_filter.split(",") if x.strip()}
    family_filter = {x.strip() for x in args.family_filter.split(",") if x.strip()}
    rows = select_rows(read_jsonl(args.input_jsonl), args.max_examples, args.sample_seed, args.shuffle, split_filter, family_filter)
    shuffled_rows = list(rows)
    random.Random(args.sample_seed + 99).shuffle(shuffled_rows)

    obs_map, _action_std = load_index_maps(args.index)
    model, processor = load_model_and_processor(args)
    out_rows: List[Dict[str, Any]] = []
    max_budget = max(budgets)

    for i, row in enumerate(rows, start=1):
        image_row = dict(row)
        if image_row.get("image_source") and not image_row.get("obs_ptr"):
            image_row["obs_ptr"] = image_row["image_source"]
        image, image_source, placeholder = load_image_for_example(image_row, obs_map, args.allow_placeholder_image)
        shuffled_row = shuffled_rows[(i - 1) % len(shuffled_rows)] if shuffled_rows else None
        blocks = []
        for condition in conditions:
            target = get_schema_target(row, condition, shuffled_row)
            variants = schema_prompt_variants(row, target, condition)[:max_budget]
            cands = []
            for name, prompt in variants:
                action = predict_action(model, processor, image, prompt, args)
                ac = actcheck(action, row, args.alignment_margin)
                sac = schema_target_actcheck(action, row, target, args.alignment_margin)
                cands.append({
                    "candidate_name": name,
                    "instruction": prompt,
                    "schema_target": target,
                    "action": action,
                    "actcheck": ac,
                    "schema_target_actcheck": sac,
                })
            blocks.append({"condition": condition, "schema_target": target, "candidates": cands})

        out_rows.append({
            "example_id": row.get("example_id"),
            "observation_id": row.get("observation_id"),
            "original_instruction": row.get("original_instruction"),
            "counterfactual_instruction": row.get("counterfactual_instruction") or row.get("instruction"),
            "original_target_object": row.get("original_target_object"),
            "counterfactual_target_object": row.get("counterfactual_target_object") or row.get("target_object"),
            "satbenchpp_split": row.get("satbenchpp_split"),
            "satbenchpp_family": row.get("satbenchpp_family"),
            "image_source": image_source,
            "placeholder_image": placeholder,
            "condition_candidates": blocks,
        })
        if args.log_every and (i % args.log_every == 0 or i == len(rows)):
            print(f"progress {i}/{len(rows)}", flush=True)
        if args.save_every and i % args.save_every == 0:
            write_jsonl(out_dir / "predictions.partial.jsonl", out_rows)
            write_csv(out_dir / "summary.partial.csv", summarize(out_rows, budgets))

    summary = summarize(out_rows, budgets)
    write_jsonl(out_dir / "predictions.jsonl", out_rows)
    write_csv(out_dir / "summary.csv", summary)
    write_json(out_dir / "run_metadata.json", {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "input_jsonl": args.input_jsonl,
        "model_path": args.model_path,
        "max_examples": args.max_examples,
        "budgets": budgets,
        "conditions": conditions,
        "split_filter": sorted(split_filter),
        "family_filter": sorted(family_filter),
        "python": sys.version,
        "platform": platform.platform(),
    })
    print(json.dumps({"out_dir": str(out_dir), "num_examples": len(out_rows), "summary_rows": len(summary)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
