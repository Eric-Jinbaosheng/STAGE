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
from lb_run_exp6_schema_action_projection import (
    counterfactual_for_task,
    eef_pos,
    load_model_and_processor_with_diagnostics,
    site_pos,
)
from lb_run_exp6_sim_gate_sanity import inject_local_dataset_statistics
from lb_run_libero10_smoke import dummy_action, get_image, get_libero_env, normalize_for_env
from linguistic_blindness.utils.io import command_string, utc_timestamp, write_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="VISA-ClosedLoop: timestep-wise proposal/verification/fallback execution in LIBERO.")
    ap.add_argument("--task-suite-name", default="libero_spatial")
    ap.add_argument("--task-ids", default="", help="Comma-separated task ids; empty auto-selects target-swap tasks.")
    ap.add_argument("--evaluation-mode", choices=["counterfactual", "original_validation"], default="counterfactual")
    ap.add_argument("--methods", default="native_openvla,generic_selection,visa_closed_loop,oracle_handoff")
    ap.add_argument("--counterfactual-spec-file", default="", help="Optional JSON/JSONL list of counterfactual specs keyed by task_id.")
    ap.add_argument("--max-tasks", type=int, default=10)
    ap.add_argument("--num-trials-per-task", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--model-path", default="/scratch/bj2410/models/openvla-7b-finetuned-libero-90")
    ap.add_argument("--unnorm-key", default="libero_90_no_noops")
    ap.add_argument("--out-dir", default="outputs/linguistic_blindness/visa_closed_loop_execution")
    ap.add_argument("--alignment-margin", type=float, default=0.05)
    ap.add_argument("--contact-distance", type=float, default=0.06)
    ap.add_argument("--lift-success-dz", type=float, default=0.03)
    ap.add_argument("--primitive-step", type=float, default=0.02)
    ap.add_argument("--residual-alphas", default="0,0.25,0.5,0.75", help="Comma-separated residual steering strengths for residual methods.")
    ap.add_argument("--residual-release-distance", type=float, default=0.10, help="Use alpha=0 once the end effector is within this distance of the counterfactual target.")
    ap.add_argument("--selective-residual-alphas", default="0,0.1,0.2,0.3", help="Small steering strengths for phase-aware selective residual methods.")
    ap.add_argument("--retry-on-fail", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--attn-implementation", default="")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--center-crop", action="store_true")
    ap.add_argument("--save-step-traces", action="store_true")
    ap.add_argument("--log-every-step", type=int, default=0)
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


def schema_prompts(cf: Dict[str, Any]) -> List[Tuple[str, str]]:
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


def generic_prompts(cf: Dict[str, Any]) -> List[Tuple[str, str]]:
    raw = str(cf["counterfactual_instruction"])
    return [
        ("native_counterfactual", raw),
        ("generic_do_task", "do the task"),
        ("generic_follow_instruction", "follow the instruction"),
        ("generic_move_to_object", "move toward the object"),
        ("generic_pick_object", "pick up the object"),
        ("generic_complete_task", "complete the manipulation task"),
    ]


def infer_validation_target(description: str) -> str:
    text = description.lower()
    for name in ["top drawer", "middle drawer", "black bowl", "ramekin", "cookie box"]:
        if name in text:
            return name
    return "top drawer"


def original_validation_spec(task_id: int, description: str) -> Dict[str, Any]:
    target = infer_validation_target(description)
    return {
        "task_id": task_id,
        "original_target_object": target,
        "counterfactual_target_object": target,
        "original_instruction": description,
        "counterfactual_instruction": description,
        "schema": {
            "target_object": target,
            "target_exists": True,
            "phase": "original_task_validation",
            "allowed_actions": ["MOVE_TO", "APPROACH", "GRASP", "PICK", "PLACE", "OPEN", "CLOSE"],
            "blocked_actions": [],
            "next_action": "EXECUTE_ORIGINAL_TASK",
            "reason": "Original-task validation spec; used only to validate the closed-loop execution pipeline.",
            "confidence": 1.0,
        },
    }


def load_counterfactual_specs(path: str) -> Dict[int, Dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = data.get("counterfactual_specs", data) if isinstance(data, dict) else data
    specs: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        spec = dict(row)
        tid = int(spec["task_id"])
        spec.setdefault("schema", {
            "target_object": spec.get("counterfactual_target_object"),
            "target_exists": True,
            "phase": "approach",
            "allowed_actions": ["MOVE_TO", "APPROACH", "GRASP", "PICK", "PLACE"],
            "blocked_actions": [],
            "next_action": "MOVE_TO",
            "reason": "Manual base-policy-competent counterfactual spec.",
            "confidence": 1.0,
        })
        specs[tid] = spec
    return specs


def choose_candidate(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    aligned = [c for c in cands if c["actcheck"]["target_aligned"]]
    pool = aligned if aligned else cands
    best = max(pool, key=lambda c: -1e9 if c["actcheck"]["score"] is None else c["actcheck"]["score"])
    return {**best, "decision": "ALLOW" if aligned else "NO_ALIGNED", "num_aligned_candidates": len(aligned)}


def primitive_target_action(ee: Optional[np.ndarray], target: Optional[np.ndarray], args: argparse.Namespace) -> Optional[List[float]]:
    if ee is None or target is None:
        return None
    direction = unit(np.asarray(target, dtype=float) - np.asarray(ee, dtype=float))
    if direction is None:
        return None
    return [float(x) for x in (direction * args.primitive_step).tolist()] + [0.0, 0.0, 0.0, 0.0]


def residual_alphas(args: argparse.Namespace) -> List[float]:
    vals: List[float] = []
    for x in str(args.residual_alphas).split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    return vals or [0.0]


def selective_residual_alphas(args: argparse.Namespace) -> List[float]:
    vals: List[float] = []
    for x in str(args.selective_residual_alphas).split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    return vals or [0.0]


def residual_action(native: List[float], ee: Optional[np.ndarray], target: Optional[np.ndarray], alpha: float, args: argparse.Namespace) -> Optional[List[float]]:
    if ee is None or target is None:
        return None
    target_xy = np.asarray(target, dtype=float)[:2] - np.asarray(ee, dtype=float)[:2]
    direction_xy = unit(target_xy)
    if direction_xy is None:
        return None
    arr = np.asarray(native, dtype=float).reshape(-1).copy()
    if arr.shape[0] < 2:
        return None
    native_xy = arr[:2]
    native_mag = float(np.linalg.norm(native_xy))
    steer_mag = max(native_mag, float(args.primitive_step))
    steered_xy = direction_xy * steer_mag
    arr[:2] = (1.0 - alpha) * native_xy + alpha * steered_xy
    return [float(x) for x in arr.tolist()]


def choose_residual_action(native: List[float], ee: Optional[np.ndarray], po: Optional[np.ndarray], pc: Optional[np.ndarray], args: argparse.Namespace) -> Tuple[List[float], Dict[str, Any]]:
    dist_cf = None if ee is None or pc is None else float(np.linalg.norm(np.asarray(ee, dtype=float) - np.asarray(pc, dtype=float)))
    alphas = [0.0] if dist_cf is not None and dist_cf <= args.residual_release_distance else residual_alphas(args)
    cands: List[Dict[str, Any]] = []
    for alpha in alphas:
        cand = residual_action(native, ee, pc, alpha, args)
        if cand is None:
            continue
        cands.append({
            "candidate_name": f"residual_alpha_{alpha:g}",
            "action": cand,
            "alpha": alpha,
            "actcheck": actcheck(cand, ee, po, pc, args.alignment_margin),
        })
    if not cands:
        return native, {"decision": "ALLOW", "selected_candidate": "native_residual_fallback", "fallback": "native", "residual_alpha": 0.0, "actcheck": actcheck(native, ee, po, pc, args.alignment_margin)}
    chosen = choose_candidate(cands)
    return chosen["action"], {
        "decision": "ALLOW" if chosen["actcheck"].get("target_aligned") else "ALLOW_UNALIGNED_RESIDUAL",
        "selected_candidate": chosen["candidate_name"],
        "fallback": "none",
        "residual_alpha": chosen["alpha"],
        "residual_released": bool(dist_cf is not None and dist_cf <= args.residual_release_distance),
        "distance_to_counterfactual_target": dist_cf,
        "actcheck": chosen["actcheck"],
    }


def choose_selective_residual_action(native: List[float], ee: Optional[np.ndarray], po: Optional[np.ndarray], pc: Optional[np.ndarray], args: argparse.Namespace) -> Tuple[List[float], Dict[str, Any]]:
    native_ac = actcheck(native, ee, po, pc, args.alignment_margin)
    dist_cf = None if ee is None or pc is None else float(np.linalg.norm(np.asarray(ee, dtype=float) - np.asarray(pc, dtype=float)))
    gate_triggered = bool(native_ac.get("wrong_target"))
    in_release_phase = bool(dist_cf is not None and dist_cf <= args.residual_release_distance)
    if not gate_triggered or in_release_phase:
        return native, {
            "decision": "ALLOW_NATIVE_SELECTIVE",
            "selected_candidate": "native_selective_passthrough",
            "fallback": "none",
            "residual_alpha": 0.0,
            "selective_gate_triggered": gate_triggered,
            "selective_release_phase": in_release_phase,
            "distance_to_counterfactual_target": dist_cf,
            "actcheck": native_ac,
        }

    cands: List[Dict[str, Any]] = []
    for alpha in selective_residual_alphas(args):
        cand = residual_action(native, ee, pc, alpha, args)
        if cand is None:
            continue
        cands.append({
            "candidate_name": f"selective_residual_alpha_{alpha:g}",
            "action": cand,
            "alpha": alpha,
            "actcheck": actcheck(cand, ee, po, pc, args.alignment_margin),
        })
    if not cands:
        return native, {
            "decision": "ALLOW_NATIVE_SELECTIVE_FALLBACK",
            "selected_candidate": "native_selective_fallback",
            "fallback": "native",
            "residual_alpha": 0.0,
            "selective_gate_triggered": gate_triggered,
            "selective_release_phase": in_release_phase,
            "distance_to_counterfactual_target": dist_cf,
            "actcheck": native_ac,
        }
    chosen = choose_candidate(cands)
    return chosen["action"], {
        "decision": "ALLOW_SELECTIVE_RESIDUAL" if chosen["actcheck"].get("target_aligned") else "ALLOW_SELECTIVE_BEST_EFFORT",
        "selected_candidate": chosen["candidate_name"],
        "fallback": "none",
        "residual_alpha": chosen["alpha"],
        "selective_gate_triggered": gate_triggered,
        "selective_release_phase": in_release_phase,
        "distance_to_counterfactual_target": dist_cf,
        "actcheck": chosen["actcheck"],
    }


def propose_action(
    method: str,
    obs: Dict[str, Any],
    cf: Dict[str, Any],
    model: Any,
    processor: Any,
    env: Any,
    args: argparse.Namespace,
) -> Tuple[List[float], Dict[str, Any]]:
    img = get_image(obs, center_crop=args.center_crop)
    po, _, _ = site_pos(env, cf["original_target_object"])
    pc, _, _ = site_pos(env, cf["counterfactual_target_object"])
    ee = eef_pos(obs)

    if method == "native_openvla":
        raw = predict_action(model, processor, img, cf["counterfactual_instruction"], args)
        return raw, {"decision": "ALLOW", "selected_candidate": "native_openvla", "actcheck": actcheck(raw, ee, po, pc, args.alignment_margin), "fallback": "none"}

    if method == "gate_only":
        raw = predict_action(model, processor, img, cf["counterfactual_instruction"], args)
        native_ac = actcheck(raw, ee, po, pc, args.alignment_margin)
        return raw, {
            "decision": "GATE_ONLY_WRONG" if native_ac.get("wrong_target") else "GATE_ONLY_PASS",
            "selected_candidate": "gate_only_native_action",
            "actcheck": native_ac,
            "fallback": "none",
            "residual_alpha": 0.0,
            "selective_gate_triggered": bool(native_ac.get("wrong_target")),
        }

    if method in {"visa_residual", "oracle_residual", "generic_residual"}:
        raw = predict_action(model, processor, img, cf["counterfactual_instruction"], args)
        action, meta = choose_residual_action(raw, ee, po, pc, args)
        meta["native_actcheck"] = actcheck(raw, ee, po, pc, args.alignment_margin)
        meta["selected_candidate"] = f"{method}_{meta.get('selected_candidate')}"
        return action, meta

    if method in {"visa_selective_residual", "oracle_selective_residual", "generic_selective_residual"}:
        raw = predict_action(model, processor, img, cf["counterfactual_instruction"], args)
        action, meta = choose_selective_residual_action(raw, ee, po, pc, args)
        meta["native_actcheck"] = actcheck(raw, ee, po, pc, args.alignment_margin)
        meta["selected_candidate"] = f"{method}_{meta.get('selected_candidate')}"
        return action, meta

    if method in {"oracle_handoff", "visa_handoff"}:
        primitive = primitive_target_action(ee, pc, args)
        if primitive is not None:
            return primitive, {"decision": "ALLOW", "selected_candidate": f"{method}_primitive", "fallback": method, "actcheck": actcheck(primitive, ee, po, pc, args.alignment_margin)}
        raw = predict_action(model, processor, img, cf["counterfactual_instruction"], args)
        return raw, {"decision": "ALLOW", "selected_candidate": f"{method}_fallback_raw", "fallback": "raw", "actcheck": actcheck(raw, ee, po, pc, args.alignment_margin)}

    prompt_source = generic_prompts(cf) if method == "generic_selection" else schema_prompts(cf)
    all_cands: List[Dict[str, Any]] = []
    for retry in range(max(1, args.retry_on_fail + 1)):
        cands = []
        for name, prompt in prompt_source:
            raw = predict_action(model, processor, img, prompt, args)
            cands.append({"candidate_name": f"{name}_r{retry}", "instruction": prompt, "action": raw, "actcheck": actcheck(raw, ee, po, pc, args.alignment_margin)})
        all_cands.extend(cands)
        chosen = choose_candidate(all_cands)
        if chosen["decision"] == "ALLOW":
            return chosen["action"], {
                "decision": "ALLOW",
                "selected_candidate": chosen["candidate_name"],
                "selected_instruction": chosen["instruction"],
                "num_aligned_candidates": chosen["num_aligned_candidates"],
                "fallback": "none",
                "actcheck": chosen["actcheck"],
                "candidate_count": len(all_cands),
            }

    chosen = choose_candidate(all_cands)
    if method == "visa_closed_loop":
        primitive = primitive_target_action(ee, pc, args)
        if primitive is not None:
            return primitive, {
                "decision": "FALLBACK_PRIMITIVE",
                "selected_candidate": chosen["candidate_name"],
                "num_aligned_candidates": 0,
                "fallback": "schema_primitive",
                "actcheck": actcheck(primitive, ee, po, pc, args.alignment_margin),
                "candidate_count": len(all_cands),
            }
    return chosen["action"], {
        "decision": "FALLBACK_BEST",
        "selected_candidate": chosen["candidate_name"],
        "selected_instruction": chosen["instruction"],
        "num_aligned_candidates": 0,
        "fallback": "best_unaligned",
        "actcheck": chosen["actcheck"],
        "candidate_count": len(all_cands),
    }


def rollout(env: Any, init_state: Any, cf: Dict[str, Any], method: str, model: Any, processor: Any, args: argparse.Namespace) -> Dict[str, Any]:
    env.reset()
    obs = env.set_init_state(init_state)
    trace: List[Dict[str, Any]] = []
    done = False
    error = None
    fallback_steps = 0
    aligned_steps = 0
    unaligned_fallback_steps = 0
    residual_steps = 0
    residual_alpha_sum = 0.0
    first_contact: Optional[str] = None
    correct_contact_step: Optional[int] = None
    wrong_contact_step: Optional[int] = None
    cf_start_z = cf_final_z = cf_max_z = None
    orig_start_z = orig_final_z = orig_max_z = None
    distances: List[Dict[str, Any]] = []

    try:
        for t in range(args.max_steps + args.num_steps_wait):
            po, _, _ = site_pos(env, cf["original_target_object"])
            pc, _, _ = site_pos(env, cf["counterfactual_target_object"])
            ee = eef_pos(obs)
            if t >= args.num_steps_wait and ee is not None and po is not None and pc is not None:
                step = t - args.num_steps_wait
                do = float(np.linalg.norm(ee - po))
                dc = float(np.linalg.norm(ee - pc))
                distances.append({"step": step, "dist_original": do, "dist_counterfactual": dc})
                if dc <= args.contact_distance and correct_contact_step is None:
                    correct_contact_step = step
                    if first_contact is None:
                        first_contact = "counterfactual"
                if do <= args.contact_distance and wrong_contact_step is None:
                    wrong_contact_step = step
                    if first_contact is None:
                        first_contact = "original"
                if pc is not None:
                    z = float(pc[2])
                    cf_start_z = z if cf_start_z is None else cf_start_z
                    cf_final_z = z
                    cf_max_z = z if cf_max_z is None else max(cf_max_z, z)
                if po is not None:
                    z = float(po[2])
                    orig_start_z = z if orig_start_z is None else orig_start_z
                    orig_final_z = z
                    orig_max_z = z if orig_max_z is None else max(orig_max_z, z)

            if t < args.num_steps_wait:
                env_action = dummy_action()
                meta = {"decision": "WARMUP", "fallback": "warmup"}
            else:
                raw, meta = propose_action(method, obs, cf, model, processor, env, args)
                env_action = normalize_for_env(raw)
                fallback_steps += int(str(meta.get("decision", "")).startswith("FALLBACK"))
                unaligned_fallback_steps += int(meta.get("fallback") == "best_unaligned")
                aligned_steps += int((meta.get("actcheck") or {}).get("target_aligned") is True)
                alpha = meta.get("residual_alpha")
                if alpha is not None:
                    residual_alpha_sum += float(alpha)
                    residual_steps += int(float(alpha) > 0.0)
                if args.save_step_traces:
                    trace.append({"step": t - args.num_steps_wait, **meta})
                if args.log_every_step and (t - args.num_steps_wait) % args.log_every_step == 0:
                    print(json.dumps({"method": method, "step": t - args.num_steps_wait, "decision": meta.get("decision"), "fallback": meta.get("fallback"), "actcheck": (meta.get("actcheck") or {}).get("label")}), flush=True)
            obs, _, done, _ = env.step(env_action)
            if done:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    def mean(vals: Iterable[float]) -> Optional[float]:
        xs = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
        return None if not xs else float(sum(xs) / len(xs))

    if distances:
        start_cf = distances[0]["dist_counterfactual"]
        start_orig = distances[0]["dist_original"]
        final_cf = distances[-1]["dist_counterfactual"]
        final_orig = distances[-1]["dist_original"]
        delta_cf = start_cf - final_cf
        delta_orig = start_orig - final_orig
        pref = delta_cf - delta_orig
        integrated_pref = mean([d["dist_original"] - d["dist_counterfactual"] for d in distances])
        target_approach_auc = mean([start_cf - d["dist_counterfactual"] for d in distances])
        wrong_approach_auc = mean([start_orig - d["dist_original"] for d in distances])
        min_cf = min(d["dist_counterfactual"] for d in distances)
        min_orig = min(d["dist_original"] for d in distances)
    else:
        final_cf = final_orig = delta_cf = delta_orig = pref = integrated_pref = target_approach_auc = wrong_approach_auc = min_cf = min_orig = None

    cf_lift_delta = None if cf_start_z is None or cf_max_z is None else float(cf_max_z - cf_start_z)
    orig_lift_delta = None if orig_start_z is None or orig_max_z is None else float(orig_max_z - orig_start_z)
    action_steps = max(1, args.max_steps)
    out = {
        "method": method,
        "success": bool(done and error is None),
        "done": bool(done),
        "error": error,
        "fallback_steps": fallback_steps,
        "fallback_rate": fallback_steps / action_steps,
        "unaligned_fallback_steps": unaligned_fallback_steps,
        "aligned_step_rate": aligned_steps / action_steps,
        "residual_correction_rate": residual_steps / action_steps,
        "mean_residual_alpha": residual_alpha_sum / action_steps,
        "first_contact": first_contact or "none",
        "correct_contact": correct_contact_step is not None,
        "wrong_contact": wrong_contact_step is not None,
        "correct_first_contact": first_contact == "counterfactual",
        "wrong_first_contact": first_contact == "original",
        "correct_contact_step": correct_contact_step,
        "wrong_contact_step": wrong_contact_step,
        "final_dist_counterfactual": final_cf,
        "final_dist_original": final_orig,
        "min_dist_counterfactual": min_cf,
        "min_dist_original": min_orig,
        "delta_dist_counterfactual": delta_cf,
        "delta_dist_original": delta_orig,
        "target_preference_score": pref,
        "integrated_target_preference_score": integrated_pref,
        "target_approach_auc": target_approach_auc,
        "wrong_object_approach_auc": wrong_approach_auc,
        "counterfactual_approach": None if delta_cf is None else delta_cf > 0,
        "wrong_object_approach": None if delta_orig is None else delta_orig > 0,
        "correct_lift_delta": cf_lift_delta,
        "wrong_lift_delta": orig_lift_delta,
        "correct_lift": None if cf_lift_delta is None else cf_lift_delta >= args.lift_success_dz,
        "wrong_lift": None if orig_lift_delta is None else orig_lift_delta >= args.lift_success_dz,
    }
    if args.save_step_traces:
        out["trace"] = distances
        out["action_trace"] = trace
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


def summarize(rows: List[Dict[str, Any]], methods: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for method in methods:
        g = [r for r in rows if r["method"] == method]
        out.append({
            "method": method,
            "n": len(g),
            "env_success_rate": rate(r["success"] for r in g),
            "correct_contact_rate": rate(r["correct_contact"] for r in g),
            "correct_first_contact_rate": rate(r["correct_first_contact"] for r in g),
            "wrong_contact_rate": rate(r["wrong_contact"] for r in g),
            "wrong_first_contact_rate": rate(r["wrong_first_contact"] for r in g),
            "correct_lift_rate": rate(r["correct_lift"] for r in g),
            "wrong_lift_rate": rate(r["wrong_lift"] for r in g),
            "mean_correct_contact_step": mean(r["correct_contact_step"] for r in g),
            "mean_final_target_preference": mean(r["target_preference_score"] for r in g),
            "mean_integrated_target_preference": mean(r["integrated_target_preference_score"] for r in g),
            "mean_target_approach_auc": mean(r["target_approach_auc"] for r in g),
            "mean_wrong_object_approach_auc": mean(r["wrong_object_approach_auc"] for r in g),
            "counterfactual_approach_rate": rate(r["counterfactual_approach"] for r in g),
            "wrong_object_approach_rate": rate(r["wrong_object_approach"] for r in g),
            "mean_fallback_rate": mean(r["fallback_rate"] for r in g),
            "mean_aligned_step_rate": mean(r["aligned_step_rate"] for r in g),
            "mean_residual_correction_rate": mean(r["residual_correction_rate"] for r in g),
            "mean_residual_alpha": mean(r["mean_residual_alpha"] for r in g),
            "error_rate": rate(bool(r["error"]) for r in g),
        })
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from libero.libero import benchmark
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    model_free_methods = {"oracle_handoff", "visa_handoff"}
    if all(m in model_free_methods for m in methods):
        model, processor = None, None
        injected_stats = False
        print("[load] skipped OpenVLA model load for model-free handoff methods", flush=True)
    else:
        model, processor = load_model_and_processor_with_diagnostics(args)
        injected_stats = inject_local_dataset_statistics(model, args.model_path, args.unnorm_key)
        if injected_stats:
            print(f"[info] injected local dataset_statistics.json for unnorm_key={args.unnorm_key}", flush=True)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    manual_specs = load_counterfactual_specs(args.counterfactual_spec_file)
    if args.task_ids.strip():
        task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()]
    elif args.evaluation_mode == "original_validation":
        task_ids = list(range(min(args.max_tasks, suite.n_tasks)))
    else:
        task_ids = []
        for tid in range(min(args.max_tasks, suite.n_tasks)):
            if counterfactual_for_task(tid, suite.get_task(tid).language) is not None:
                task_ids.append(tid)
    rows: List[Dict[str, Any]] = []
    partial_path = out_dir / "visa_closed_loop_rollouts.partial.jsonl"
    partial_path.write_text("", encoding="utf-8")

    for task_id in task_ids:
        task = suite.get_task(task_id)
        if args.evaluation_mode == "original_validation":
            cf = original_validation_spec(task_id, task.language)
        elif task_id in manual_specs:
            cf = manual_specs[task_id]
        else:
            cf = counterfactual_for_task(task_id, task.language)
        if cf is None:
            print(f"[warn] skipping task_id={task_id}: no counterfactual mapping", flush=True)
            continue
        init_states = suite.get_task_init_states(task_id)
        env, desc = get_libero_env(task, resolution=256)
        for trial_idx in range(min(args.num_trials_per_task, len(init_states))):
            for method in methods:
                result = rollout(env, init_states[trial_idx], cf, method, model, processor, args)
                row = {
                    "task_suite": args.task_suite_name,
                    "evaluation_mode": args.evaluation_mode,
                    "task_id": task_id,
                    "trial_idx": trial_idx,
                    "rollout_id": f"task{task_id}_trial{trial_idx}",
                    "task_description": desc,
                    "original_instruction": cf["original_instruction"],
                    "counterfactual_instruction": cf["counterfactual_instruction"],
                    "original_target_object": cf["original_target_object"],
                    "counterfactual_target_object": cf["counterfactual_target_object"],
                    **result,
                }
                rows.append(row)
                with partial_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps({k: row.get(k) for k in ["task_id", "trial_idx", "method", "success", "correct_contact", "wrong_contact", "correct_lift", "first_contact", "mean_fallback_rate", "target_preference_score", "error"]}, sort_keys=True), flush=True)
        env.close()

    summary = summarize(rows, methods)
    write_jsonl(out_dir / "visa_closed_loop_rollouts.jsonl", rows)
    write_csv(out_dir / "summary.csv", summary)
    write_json(out_dir / "summary.json", {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "task_suite": args.task_suite_name,
        "evaluation_mode": args.evaluation_mode,
        "task_ids": task_ids,
        "model_path": args.model_path,
        "unnorm_key": args.unnorm_key,
        "counterfactual_spec_file": args.counterfactual_spec_file,
        "injected_local_dataset_statistics": injected_stats,
        "max_steps": args.max_steps,
        "num_steps_wait": args.num_steps_wait,
        "contact_distance": args.contact_distance,
        "lift_success_dz": args.lift_success_dz,
        "summary": summary,
        "note": "This is true closed-loop env.step execution. Contact/lift are simulator proxy metrics independent of ActCheck; VISA-ClosedLoop still uses ActCheck internally for action selection and schema-primitive fallback.",
    })
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
