#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in [REPO_ROOT / "third_party" / "LIBERO_pkg", REPO_ROOT / "third_party" / "openvla", REPO_ROOT / "scripts", REPO_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lb_run_exp2_action_sensitivity import predict_action
from lb_run_exp6_schema_action_projection import counterfactual_for_task, eef_pos, load_model_and_processor_with_diagnostics, site_pos
from lb_run_exp6_sim_gate_sanity import inject_local_dataset_statistics
from lb_run_libero10_smoke import dummy_action, get_image, get_libero_env, normalize_for_env
from linguistic_blindness.utils.io import command_string, utc_timestamp, write_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="K-step closed-loop rollout for VISA-Rerank action-boundary interface.")
    ap.add_argument("--task-suite-name", default="libero_spatial")
    ap.add_argument("--task-ids", default="1,3,4,5,6")
    ap.add_argument("--num-trials-per-task", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--model-path", default="/scratch/bj2410/models/openvla-7b-finetuned-libero-90")
    ap.add_argument("--unnorm-key", default="libero_90_no_noops")
    ap.add_argument("--out-dir", default="outputs/linguistic_blindness/short_horizon_visa_rerank_k50")
    ap.add_argument("--alignment-margin", type=float, default=0.05)
    ap.add_argument("--contact-distance", type=float, default=0.06)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--attn-implementation", default="")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--center-crop", action="store_true")
    ap.add_argument("--save-step-traces", action="store_true")
    return ap.parse_args()


def unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    return None if n < 1e-12 else v / n


def cos(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    ua, ub = unit(a), unit(b)
    if ua is None or ub is None:
        return None
    return float(np.dot(ua, ub))


def actcheck(action: List[float], ee: Optional[np.ndarray], po: Optional[np.ndarray], pc: Optional[np.ndarray], margin: float) -> Dict[str, Any]:
    if ee is None or po is None or pc is None:
        return {"score": None, "label": "unknown", "target_aligned": False, "wrong_target": False, "ambiguous": False}
    a = np.asarray(action, dtype=float).reshape(-1)[:3]
    c_cf = cos(a, pc - ee)
    c_orig = cos(a, po - ee)
    score = None if c_cf is None or c_orig is None else c_cf - c_orig
    aligned = score is not None and score > margin
    wrong = score is not None and score <= -margin
    ambiguous = score is not None and abs(score) <= margin
    return {
        "cos_to_counterfactual_target": c_cf,
        "cos_to_original_target": c_orig,
        "score": score,
        "target_aligned": bool(aligned),
        "wrong_target": bool(wrong),
        "ambiguous": bool(ambiguous),
        "label": "aligned" if aligned else ("wrong-target" if wrong else ("ambiguous" if ambiguous else "unknown")),
    }


def candidate_prompts(cf: Dict[str, Any]) -> List[Tuple[str, str]]:
    raw = str(cf["counterfactual_instruction"])
    target = str(cf["counterfactual_target_object"])
    original = str(cf["original_target_object"])
    next_action = str((cf.get("schema") or {}).get("next_action") or "MOVE_TO").lower()
    return [
        ("native_counterfactual", raw),
        ("target_only", f"move toward the {target}"),
        ("target_manipulate", f"pick up the {target}" if "drawer" not in target.lower() else f"open the {target}"),
        ("explicit_exclude_old", f"{raw}. The target object is {target}. Do not target {original}."),
        ("schema_next_action", f"next robot action: {next_action} toward {target}"),
        ("boundary_handoff", f"for the next low-level action, move the end effector toward {target} and ignore {original}"),
    ]


def no_schema_candidate_prompts(cf: Dict[str, Any]) -> List[Tuple[str, str]]:
    raw = str(cf["counterfactual_instruction"])
    return [
        ("native_counterfactual", raw),
        ("generic_do_task", "do the task"),
        ("generic_follow_instruction", "follow the instruction"),
        ("generic_move_to_object", "move toward the object"),
        ("generic_pick_object", "pick up the object"),
        ("generic_complete_task", "complete the manipulation task"),
    ]


def choose_candidate(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    aligned = [c for c in cands if c["actcheck"]["target_aligned"]]
    pool = aligned if aligned else cands
    best = max(pool, key=lambda c: -1e9 if c["actcheck"]["score"] is None else c["actcheck"]["score"])
    return {**best, "decision": "ALLOW" if aligned else "DEFER", "num_aligned_candidates": len(aligned)}


def action_for_method(
    method: str,
    obs: Dict[str, Any],
    cf: Dict[str, Any],
    model: Any,
    processor: Any,
    env: Any,
    args: argparse.Namespace,
) -> Tuple[Optional[List[float]], Dict[str, Any]]:
    img = get_image(obs, center_crop=args.center_crop)
    po, _, _ = site_pos(env, cf["original_target_object"])
    pc, _, _ = site_pos(env, cf["counterfactual_target_object"])
    ee = eef_pos(obs)

    if method == "native_counterfactual":
        raw = predict_action(model, processor, img, cf["counterfactual_instruction"], args)
        return raw, {"selected_candidate": "native_counterfactual", "decision": "ALLOW", "actcheck": actcheck(raw, ee, po, pc, args.alignment_margin)}
    if method == "single_target_prompt":
        prompt = f"pick up the {cf['counterfactual_target_object']}"
        raw = predict_action(model, processor, img, prompt, args)
        return raw, {"selected_candidate": "single_target_prompt", "selected_instruction": prompt, "decision": "ALLOW", "actcheck": actcheck(raw, ee, po, pc, args.alignment_margin)}
    if method == "primitive_handoff":
        if ee is None or pc is None:
            return None, {"selected_candidate": "primitive_handoff", "decision": "DEFER", "reason": "missing_position"}
        vec = pc - ee
        direction = unit(vec)
        if direction is None:
            return None, {"selected_candidate": "primitive_handoff", "decision": "DEFER", "reason": "zero_direction"}
        action = [float(x) for x in (direction * 0.02).tolist()] + [0.0, 0.0, 0.0, 0.0]
        return action, {"selected_candidate": "primitive_handoff", "decision": "ALLOW", "actcheck": actcheck(action, ee, po, pc, args.alignment_margin)}

    prompt_source = no_schema_candidate_prompts(cf) if method == "best_of_n_no_schema" else candidate_prompts(cf)
    cands = []
    for name, prompt in prompt_source:
        raw = predict_action(model, processor, img, prompt, args)
        cands.append({"candidate_name": name, "instruction": prompt, "action": raw, "actcheck": actcheck(raw, ee, po, pc, args.alignment_margin)})
    chosen = choose_candidate(cands)
    if chosen["decision"] != "ALLOW":
        return None, {"selected_candidate": chosen["candidate_name"], "decision": "DEFER", "num_aligned_candidates": 0, "candidates": cands}
    return chosen["action"], {"selected_candidate": chosen["candidate_name"], "decision": "ALLOW", "num_aligned_candidates": chosen["num_aligned_candidates"], "actcheck": chosen["actcheck"], "candidates": cands}


def rollout(env: Any, init_state: Any, cf: Dict[str, Any], method: str, model: Any, processor: Any, args: argparse.Namespace) -> Dict[str, Any]:
    env.reset()
    obs = env.set_init_state(init_state)
    distances: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    first_close: Optional[str] = None
    done = False
    err = None
    defer_steps = 0
    try:
        for t in range(args.horizon + args.num_steps_wait):
            po, _, _ = site_pos(env, cf["original_target_object"])
            pc, _, _ = site_pos(env, cf["counterfactual_target_object"])
            ee = eef_pos(obs)
            if t >= args.num_steps_wait and po is not None and pc is not None and ee is not None:
                do = float(np.linalg.norm(ee - po))
                dc = float(np.linalg.norm(ee - pc))
                if first_close is None:
                    if dc <= args.contact_distance and do > args.contact_distance:
                        first_close = "counterfactual"
                    elif do <= args.contact_distance and dc > args.contact_distance:
                        first_close = "original"
                    elif do <= args.contact_distance and dc <= args.contact_distance:
                        first_close = "both"
                distances.append({"step": t - args.num_steps_wait, "dist_original": do, "dist_counterfactual": dc})
            if t < args.num_steps_wait:
                env_action = dummy_action()
                meta = {"warmup": True}
            else:
                raw, meta = action_for_method(method, obs, cf, model, processor, env, args)
                if raw is None:
                    defer_steps += 1
                    env_action = dummy_action()
                else:
                    env_action = normalize_for_env(raw)
                if args.save_step_traces:
                    actions.append({"step": t - args.num_steps_wait, **meta})
            obs, _, done, _ = env.step(env_action)
            if done:
                break
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"

    def first_last(key: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if not distances:
            return None, None, None
        start = distances[0][key]
        end = distances[-1][key]
        return start, end, start - end

    so, eo, delta_o = first_last("dist_original")
    sc, ec, delta_c = first_last("dist_counterfactual")
    integrated_c = None if not distances else float(np.mean([d["dist_counterfactual"] for d in distances]))
    integrated_o = None if not distances else float(np.mean([d["dist_original"] for d in distances]))
    if distances:
        start_cf = distances[0]["dist_counterfactual"]
        target_approach_auc = float(np.mean([start_cf - d["dist_counterfactual"] for d in distances]))
        start_orig = distances[0]["dist_original"]
        wrong_object_approach_auc = float(np.mean([start_orig - d["dist_original"] for d in distances]))
    else:
        target_approach_auc = None
        wrong_object_approach_auc = None
    pref = None if delta_o is None or delta_c is None else delta_c - delta_o
    integrated_pref = None if integrated_c is None or integrated_o is None else integrated_o - integrated_c
    out = {
        "method": method,
        "done": bool(done),
        "error": err,
        "defer_steps": defer_steps,
        "defer_rate": None if args.horizon <= 0 else defer_steps / args.horizon,
        "start_dist_original": so,
        "final_dist_original": eo,
        "delta_dist_original": delta_o,
        "start_dist_counterfactual": sc,
        "final_dist_counterfactual": ec,
        "delta_dist_counterfactual": delta_c,
        "integrated_dist_counterfactual": integrated_c,
        "integrated_dist_original": integrated_o,
        "target_preference_score": pref,
        "integrated_target_preference_score": integrated_pref,
        "target_approach_auc": target_approach_auc,
        "wrong_object_approach_auc": wrong_object_approach_auc,
        "counterfactual_approach": None if delta_c is None else delta_c > 0,
        "original_approach": None if delta_o is None else delta_o > 0,
        "first_close_target": first_close or "none",
        "first_close_correct": first_close == "counterfactual",
        "wrong_object_first_close": first_close == "original",
    }
    if args.save_step_traces:
        out["trace"] = distances
        out["actions"] = actions
    return out


def mean(vals: Iterable[Any]) -> Optional[float]:
    xs = []
    for v in vals:
        try:
            if v is not None and not math.isnan(float(v)):
                xs.append(float(v))
        except Exception:
            pass
    return None if not xs else float(sum(xs) / len(xs))


def rate(vals: Iterable[Any]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return None if not xs else float(sum(bool(v) for v in xs) / len(xs))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from libero.libero import benchmark

    model, processor = load_model_and_processor_with_diagnostics(args)
    inject_local_dataset_statistics(model, args.model_path, args.unnorm_key)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()]
    methods = ["native_counterfactual", "single_target_prompt", "best_of_n_no_schema", "visa_rerank", "primitive_handoff"]
    rows: List[Dict[str, Any]] = []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        cf = counterfactual_for_task(task_id, task.language)
        if not cf:
            continue
        env, desc = get_libero_env(task, resolution=256)
        init_states = suite.get_task_init_states(task_id)
        for trial in range(min(args.num_trials_per_task, len(init_states))):
            init_state = init_states[trial]
            for method in methods:
                result = rollout(env, init_state, cf, method, model, processor, args)
                row = {
                    "task_id": task_id,
                    "trial_idx": trial,
                    "rollout_id": f"task{task_id}_trial{trial}",
                    "task_description": desc,
                    "original_instruction": cf["original_instruction"],
                    "counterfactual_instruction": cf["counterfactual_instruction"],
                    "original_target_object": cf["original_target_object"],
                    "counterfactual_target_object": cf["counterfactual_target_object"],
                    **result,
                }
                rows.append(row)
                print(json.dumps({k: row[k] for k in ["task_id", "trial_idx", "method", "target_preference_score", "integrated_target_preference_score", "first_close_target", "defer_rate", "error"]}, sort_keys=True), flush=True)
        env.close()

    summary = []
    for method in methods:
        g = [r for r in rows if r["method"] == method]
        summary.append({
            "method": method,
            "n": len(g),
            "mean_final_target_preference": mean([r["target_preference_score"] for r in g]),
            "mean_integrated_target_preference": mean([r["integrated_target_preference_score"] for r in g]),
            "mean_target_approach_auc": mean([r["target_approach_auc"] for r in g]),
            "mean_wrong_object_approach_auc": mean([r["wrong_object_approach_auc"] for r in g]),
            "counterfactual_approach_rate": rate([r["counterfactual_approach"] for r in g]),
            "wrong_object_approach_rate": rate([r["original_approach"] for r in g]),
            "first_close_correct_rate": rate([r["first_close_correct"] for r in g]),
            "wrong_object_first_close_rate": rate([r["wrong_object_first_close"] for r in g]),
            "mean_defer_rate": mean([r["defer_rate"] for r in g]),
            "error_rate": rate([bool(r["error"]) for r in g]),
        })
    visa_rows = [r for r in rows if r["method"] == "visa_rerank"]
    allowed_ids = {r["rollout_id"] for r in visa_rows if (r["defer_rate"] is not None and r["defer_rate"] < 1.0)}
    fully_allowed_ids = {r["rollout_id"] for r in visa_rows if (r["defer_rate"] is not None and r["defer_rate"] == 0.0)}
    matched_summary = []
    for label, ids in [("visa_any_allowed", allowed_ids), ("visa_fully_allowed", fully_allowed_ids)]:
        for method in ["native_counterfactual", "single_target_prompt", "best_of_n_no_schema", "visa_rerank", "primitive_handoff"]:
            g = [r for r in rows if r["method"] == method and r["rollout_id"] in ids]
            matched_summary.append({
                "subset": label,
                "method": method,
                "n": len(g),
                "coverage": None if not visa_rows else len(ids) / len(visa_rows),
                "mean_final_target_preference": mean([r["target_preference_score"] for r in g]),
                "mean_integrated_target_preference": mean([r["integrated_target_preference_score"] for r in g]),
                "mean_target_approach_auc": mean([r["target_approach_auc"] for r in g]),
                "mean_wrong_object_approach_auc": mean([r["wrong_object_approach_auc"] for r in g]),
                "counterfactual_approach_rate": rate([r["counterfactual_approach"] for r in g]),
                "wrong_object_approach_rate": rate([r["original_approach"] for r in g]),
                "first_close_correct_rate": rate([r["first_close_correct"] for r in g]),
                "wrong_object_first_close_rate": rate([r["wrong_object_first_close"] for r in g]),
                "mean_defer_rate": mean([r["defer_rate"] for r in g]),
                "error_rate": rate([bool(r["error"]) for r in g]),
            })
    write_jsonl(out_dir / "short_horizon_visa_rerank_rollouts.jsonl", rows)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "summary_allowed_matched.csv", matched_summary)
    write_json(out_dir / "summary.json", {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "horizon": args.horizon,
        "num_rows": len(rows),
        "summary": summary,
        "summary_allowed_matched": matched_summary,
        "note": "Closed-loop behavior metrics are distance/first-close proxies independent of ActCheck labels; VISA-Rerank still uses ActCheck internally for action-boundary selection.",
    })
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
