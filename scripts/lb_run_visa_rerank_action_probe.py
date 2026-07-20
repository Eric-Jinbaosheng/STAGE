#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in [REPO_ROOT / "src", REPO_ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lb_run_exp2_action_sensitivity import action_stats, load_image_for_example, load_index_maps, load_model_and_processor, predict_action
from linguistic_blindness.utils.io import command_string, read_jsonl, utc_timestamp, write_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="VISA-Rerank: generate multiple schema-conditioned OpenVLA candidates and select by ActCheck.")
    ap.add_argument("--positioned-predictions", default="outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/analysis/predictions_with_positions.jsonl")
    ap.add_argument("--qwen-predictions", default="outputs/linguistic_blindness/qwen25vl7b_schema_libero_target_swap_clean_600/predictions.jsonl")
    ap.add_argument("--index", default="data/processed/unified_index.parquet")
    ap.add_argument("--model-path", default="/scratch/bj2410/models/openvla-7b")
    ap.add_argument("--out-dir", default="outputs/linguistic_blindness/visa_rerank_target_swap_600")
    ap.add_argument("--max-examples", type=int, default=600)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--sample-seed", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=2.2851185083448717)
    ap.add_argument("--alignment-margin", type=float, default=0.05)
    ap.add_argument("--unnorm-key", default="bridge_orig")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--attn-implementation", default="")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--allow-placeholder-image", action="store_true")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--candidate-set", choices=["schema", "no_schema"], default="schema")
    return ap.parse_args()


def by_example_id(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("example_id")): r for r in rows if r.get("example_id")}


def vec3(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)[:3]


def row_pos(row: Dict[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    state = row.get("observation_state") or {}
    return state[key]


def has_position_fields(row: Dict[str, Any]) -> bool:
    try:
        row_pos(row, "ee_pos")
        row_pos(row, "original_target_pos")
        row_pos(row, "counterfactual_target_pos")
        return True
    except Exception:
        return False


def unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    return None if n < 1e-12 else v / n


def cos(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    ua, ub = unit(a), unit(b)
    if ua is None or ub is None:
        return None
    return float(np.dot(ua, ub))


def actcheck(action: List[float], row: Dict[str, Any], margin: float) -> Dict[str, Any]:
    ee = vec3(row_pos(row, "ee_pos"))
    po = vec3(row_pos(row, "original_target_pos"))
    pc = vec3(row_pos(row, "counterfactual_target_pos"))
    a = vec3(action)
    c_cf = cos(a, pc - ee)
    c_orig = cos(a, po - ee)
    score = None if c_cf is None or c_orig is None else c_cf - c_orig
    aligned = score is not None and score > margin
    wrong = score is not None and score <= -margin
    ambiguous = score is not None and abs(score) <= margin
    return {
        "cos_to_counterfactual_target": c_cf,
        "cos_to_original_target": c_orig,
        "semantic_action_consistency_score": score,
        "target_aligned": bool(aligned),
        "wrong_target": bool(wrong),
        "ambiguous": bool(ambiguous),
        "label": "aligned" if aligned else ("wrong-target" if wrong else ("ambiguous" if ambiguous else "unknown")),
    }


def safe_schema(row: Dict[str, Any], qwen: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    q = qwen.get(str(row.get("example_id")), {})
    return q.get("parsed_schema") or row.get("qwen_schema_counterfactual") or {}


def candidate_prompts(row: Dict[str, Any], schema: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = str(row.get("counterfactual_instruction") or "").strip()
    target = str(schema.get("target_object") or row.get("counterfactual_target_object") or "").strip()
    original = str(row.get("original_target_object") or "").strip()
    next_action = str(schema.get("next_action") or "MOVE_TO").lower()
    target_phrase = target or "the specified target"
    prompts = [
        ("native_counterfactual", raw),
        ("target_only", f"move toward the {target_phrase}"),
        ("target_manipulate", f"pick up the {target_phrase}" if "drawer" not in target_phrase.lower() else f"open the {target_phrase}"),
        ("explicit_exclude_old", f"{raw}. The target object is {target_phrase}. Do not target {original}."),
        ("schema_next_action", f"next robot action: {next_action} toward {target_phrase}"),
        ("boundary_handoff", f"for the next low-level action, move the end effector toward {target_phrase} and ignore {original}"),
    ]
    out = []
    seen = set()
    for name, prompt in prompts:
        p = " ".join(prompt.split())
        if p and p not in seen:
            out.append({"candidate_name": name, "instruction": p})
            seen.add(p)
    return out


def no_schema_candidate_prompts(row: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = str(row.get("counterfactual_instruction") or "").strip()
    prompts = [
        ("native_counterfactual", raw),
        ("generic_do_task", "do the task"),
        ("generic_follow_instruction", "follow the instruction"),
        ("generic_move_to_object", "move toward the object"),
        ("generic_pick_object", "pick up the object"),
        ("generic_complete_task", "complete the manipulation task"),
    ]
    out = []
    seen = set()
    for name, prompt in prompts:
        p = " ".join(prompt.split())
        if p and p not in seen:
            out.append({"candidate_name": name, "instruction": p})
            seen.add(p)
    return out


def choose_candidate(candidates: List[Dict[str, Any]], margin: float) -> Dict[str, Any]:
    aligned = [c for c in candidates if c["actcheck"]["target_aligned"]]
    pool = aligned if aligned else candidates
    best = max(pool, key=lambda c: (-1e9 if c["actcheck"]["semantic_action_consistency_score"] is None else c["actcheck"]["semantic_action_consistency_score"]))
    return {
        **best,
        "rerank_decision": "ALLOW" if best["actcheck"]["target_aligned"] else "DEFER",
        "rerank_found_aligned": bool(aligned),
        "rerank_num_aligned_candidates": len(aligned),
        "rerank_alignment_margin": margin,
    }


def mean(xs: Iterable[Any]) -> Optional[float]:
    vals = []
    for x in xs:
        try:
            if x is not None and not math.isnan(float(x)):
                vals.append(float(x))
        except Exception:
            pass
    return None if not vals else float(sum(vals) / len(vals))


def rate(xs: Iterable[Any]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return None if not vals else float(sum(bool(x) for x in vals) / len(vals))


def summarize(rows: List[Dict[str, Any]], threshold: float, group_field: Optional[str] = None) -> List[Dict[str, Any]]:
    methods = ["native_counterfactual", "single_best_prompt_candidate", "visa_rerank"]
    out = []
    groups = [("overall", rows)]
    if group_field:
        for key in sorted({str(r.get(group_field) or "unknown") for r in rows}):
            groups.append((key, [r for r in rows if str(r.get(group_field) or "unknown") == key]))
    for group_name, g in groups:
      for method in methods:
        if method == "native_counterfactual":
            aligned = [r["native_actcheck"]["target_aligned"] for r in g]
            wrong_amb = [r["native_actcheck"]["wrong_target"] or r["native_actcheck"]["ambiguous"] for r in g]
            sens = [r["native_delta_from_original"] >= threshold for r in g]
            score = [r["native_actcheck"]["semantic_action_consistency_score"] for r in g]
            intervention = [False for _ in g]
            defer = [False for _ in g]
        elif method == "single_best_prompt_candidate":
            # Best single candidate by average prompt candidate selection is approximated
            # as the best non-native prompt per example, before the rerank allow/defer gate.
            aligned = [r["best_prompt_actcheck"]["target_aligned"] for r in g]
            wrong_amb = [r["best_prompt_actcheck"]["wrong_target"] or r["best_prompt_actcheck"]["ambiguous"] for r in g]
            sens = [r["best_prompt_delta_from_original"] >= threshold for r in g]
            score = [r["best_prompt_actcheck"]["semantic_action_consistency_score"] for r in g]
            intervention = [True for _ in g]
            defer = [False for _ in g]
        else:
            aligned = [r["rerank_selected_actcheck"]["target_aligned"] for r in g if r["rerank_decision"] == "ALLOW"]
            wrong_amb = [r["rerank_selected_actcheck"]["wrong_target"] or r["rerank_selected_actcheck"]["ambiguous"] for r in g if r["rerank_decision"] == "ALLOW"]
            sens = [r["rerank_delta_from_original"] >= threshold for r in g if r["rerank_decision"] == "ALLOW"]
            score = [r["rerank_selected_actcheck"]["semantic_action_consistency_score"] for r in g if r["rerank_decision"] == "ALLOW"]
            intervention = [r["rerank_selected_candidate_name"] != "native_counterfactual" for r in g]
            defer = [r["rerank_decision"] == "DEFER" for r in g]
        out.append({
            "group": group_name,
            "group_field": group_field or "overall",
            "method": method,
            "n": len(g),
            "allow_coverage": None if method != "visa_rerank" else rate([r["rerank_decision"] == "ALLOW" for r in g]),
            "defer_rate": rate(defer),
            "intervention_rate": rate(intervention),
            "action_sensitivity": rate(sens),
            "actcheck_aligned_rate": rate(aligned),
            "wrong_or_ambiguous_rate": rate(wrong_amb),
            "mean_semantic_action_consistency": mean(score),
        })
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        r for r in read_jsonl(args.positioned_predictions)
        if r.get("position_extraction_status") in {None, "ok"} and has_position_fields(r)
    ]
    qwen = by_example_id(read_jsonl(args.qwen_predictions))
    if args.shuffle:
        random.Random(args.sample_seed).shuffle(rows)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]

    obs_map, action_std = load_index_maps(args.index)
    model, processor = load_model_and_processor(args)

    out_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        schema = safe_schema(row, qwen)
        image_row = dict(row)
        if image_row.get("image_source") and not image_row.get("obs_ptr"):
            image_row["obs_ptr"] = image_row["image_source"]
        image, image_source, placeholder = load_image_for_example(image_row, obs_map, args.allow_placeholder_image)

        action_orig = row["openvla_action_original"]
        native_action = row["openvla_action_counterfactual"]
        native_ac = actcheck(native_action, row, args.alignment_margin)
        native_delta = float(action_stats(action_orig, native_action, action_std)["normalized_action_delta"])

        candidates: List[Dict[str, Any]] = []
        prompt_candidates = no_schema_candidate_prompts(row) if args.candidate_set == "no_schema" else candidate_prompts(row, schema)
        for cand in prompt_candidates:
            if cand["candidate_name"] == "native_counterfactual":
                action = native_action
            else:
                action = predict_action(model, processor, image, cand["instruction"], args)
            ac = actcheck(action, row, args.alignment_margin)
            delta = float(action_stats(action_orig, action, action_std)["normalized_action_delta"])
            candidates.append({**cand, "action": action, "actcheck": ac, "delta_from_original": delta})

        chosen = choose_candidate(candidates, args.alignment_margin)
        non_native = [c for c in candidates if c["candidate_name"] != "native_counterfactual"]
        best_prompt = choose_candidate(non_native, args.alignment_margin)

        out_rows.append({
            "example_id": row.get("example_id"),
            "observation_id": row.get("observation_id"),
            "target_pair": f"{row.get('original_target_object')} -> {row.get('counterfactual_target_object')}",
            "original_instruction": row.get("original_instruction"),
            "counterfactual_instruction": row.get("counterfactual_instruction"),
            "original_target_object": row.get("original_target_object"),
            "counterfactual_target_object": row.get("counterfactual_target_object"),
            "qwen_target_counterfactual": schema.get("target_object"),
            "image_source": image_source,
            "placeholder_image": placeholder,
            "native_delta_from_original": native_delta,
            "native_actcheck": native_ac,
            "candidate_actions": candidates,
            "best_prompt_candidate_name": best_prompt["candidate_name"],
            "best_prompt_instruction": best_prompt["instruction"],
            "best_prompt_delta_from_original": best_prompt["delta_from_original"],
            "best_prompt_actcheck": best_prompt["actcheck"],
            "rerank_decision": chosen["rerank_decision"],
            "rerank_found_aligned": chosen["rerank_found_aligned"],
            "rerank_num_aligned_candidates": chosen["rerank_num_aligned_candidates"],
            "rerank_selected_candidate_name": chosen["candidate_name"],
            "rerank_selected_instruction": chosen["instruction"],
            "rerank_selected_action": chosen["action"],
            "rerank_delta_from_original": chosen["delta_from_original"],
            "rerank_selected_actcheck": chosen["actcheck"],
            "action_sensitivity_threshold": args.threshold,
            "candidate_set": args.candidate_set,
        })

        if args.log_every and (i % args.log_every == 0 or i == len(rows)):
            print(f"progress {i}/{len(rows)}", flush=True)
        if args.save_every and i % args.save_every == 0:
            write_jsonl(out_dir / "predictions.partial.jsonl", out_rows)

    summary = summarize(out_rows, args.threshold)
    split_summary = summarize(out_rows, args.threshold, "satbenchpp_split")
    family_summary = summarize(out_rows, args.threshold, "satbenchpp_family")
    write_jsonl(out_dir / "predictions.jsonl", out_rows)
    write_csv(out_dir / "summary.csv", summary)
    if any(r.get("satbenchpp_split") for r in out_rows):
        write_csv(out_dir / "summary_by_split.csv", split_summary)
        write_csv(out_dir / "summary_by_family.csv", family_summary)
    write_json(out_dir / "metrics.json", {"summary": summary, "summary_by_split": split_summary, "summary_by_family": family_summary, "num_examples": len(out_rows), "threshold": args.threshold})
    write_json(out_dir / "run_metadata.json", {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "model_path": args.model_path,
        "positioned_predictions": args.positioned_predictions,
        "qwen_predictions": args.qwen_predictions,
        "alignment_margin": args.alignment_margin,
        "candidate_set": args.candidate_set,
        "note": "VISA-Rerank generates multiple schema-conditioned candidates and uses ActCheck at the action boundary. DEFER is counted as no unsafe exposure, not as low-level control success.",
        "python": sys.version,
        "platform": platform.platform(),
    })
    print(json.dumps({"out_dir": str(out_dir), "num_examples": len(out_rows), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
