#!/usr/bin/env python3
"""Exp6C: schema-conditioned action projection in LIBERO rollouts.

This is an intervention probe. It does not retrain OpenVLA and does not replace
the controller with a full planner. For target-swap instructions, it projects
OpenVLA's translational action component toward the schema target position and
measures whether the closed-loop end-effector trajectory is redirected.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in [REPO_ROOT / "third_party" / "LIBERO_pkg", REPO_ROOT / "third_party" / "openvla", REPO_ROOT / "scripts", REPO_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lb_run_exp6_sim_gate_sanity import inject_local_dataset_statistics  # noqa: E402
from lb_run_exp2_action_sensitivity import predict_action  # noqa: E402
from lb_run_openvla_action import dtype_from_args  # noqa: E402
from lb_run_libero10_smoke import (  # noqa: E402
    dummy_action,
    get_image,
    get_libero_env,
    normalize_for_env,
)
from linguistic_blindness.utils.io import command_string, utc_timestamp, write_csv, write_json, write_jsonl  # noqa: E402


TARGET_TO_SITE = {
    "black bowl": "akita_black_bowl_1_default_site",
    "butter": "butter_1_default_site",
    "butter at the front": "butter_1_default_site",
    "front butter": "butter_1_default_site",
    "butter_1": "butter_1_default_site",
    "butter at the back": "butter_2_default_site",
    "back butter": "butter_2_default_site",
    "butter_2": "butter_2_default_site",
    "chocolate pudding": "chocolate_pudding_1_default_site",
    "plate": "plate_1_default_site",
    "ramekin": "glazed_rim_porcelain_ramekin_1_default_site",
    "cookie box": "cookies_1_default_site",
    "top drawer": "wooden_cabinet_1_top_region",
    "middle drawer": "wooden_cabinet_1_middle_region",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp6C schema-conditioned action projection rollout probe.")
    p.add_argument("--task-suite-name", default="libero_spatial")
    p.add_argument("--max-tasks", type=int, default=10)
    p.add_argument("--task-ids", default="", help="Comma-separated LIBERO task ids; empty uses auto-selected target-swap tasks.")
    p.add_argument("--num-trials-per-task", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--model-path", default="/scratch/bj2410/models/openvla-7b-finetuned-libero-90")
    p.add_argument("--unnorm-key", default="libero_90_no_noops")
    p.add_argument("--out-dir", default="outputs/linguistic_blindness/exp6_schema_action_projection_smoke")
    p.add_argument("--alpha", type=float, default=0.7, help="Blend weight toward schema target direction.")
    p.add_argument("--alignment-threshold", type=float, default=0.3, help="Only project when raw action is poorly aligned with target.")
    p.add_argument("--project-step-scale", type=float, default=1.0, help="Multiplier on projected translational step norm.")
    p.add_argument("--projection-mode", choices=["alignment", "always", "until_close", "hybrid_until_close", "phase_adapter"], default="alignment")
    p.add_argument("--release-distance", type=float, default=0.05, help="For until_close, stop projection once eef is this close to target.")
    p.add_argument("--reacquire-distance", type=float, default=0.12, help="For hybrid_until_close, resume projection if raw VLA drifts beyond this distance.")
    p.add_argument("--min-step-scale", type=float, default=0.01)
    p.add_argument("--phase-descend-steps", type=int, default=12)
    p.add_argument("--phase-close-steps", type=int, default=18)
    p.add_argument("--phase-lift-steps", type=int, default=45)
    p.add_argument("--phase-descend-dz", type=float, default=-0.01)
    p.add_argument("--phase-lift-dz", type=float, default=0.02)
    p.add_argument("--lift-success-dz", type=float, default=0.03)
    p.add_argument("--always-project", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--attn-implementation", default="")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--center-crop", action="store_true")
    p.add_argument("--save-step-traces", action="store_true")
    return p.parse_args()


def load_model_and_processor_with_diagnostics(args: argparse.Namespace):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"[load] processor from {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code, local_files_only=True)
    print("[load] processor loaded", flush=True)
    kwargs = {
        "torch_dtype": dtype_from_args(args),
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": True,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    print(f"[load] model from_pretrained start dtype={kwargs['torch_dtype']} attn={args.attn_implementation or 'default'}", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(args.model_path, **kwargs)
    print("[load] model from_pretrained returned", flush=True)
    if args.device and not args.device.startswith("auto"):
        print(f"[load] moving model to {args.device}", flush=True)
        model = model.to(args.device)
        print(f"[load] moved model to {args.device}", flush=True)
    model.eval()
    print(f"[load] model ready on {next(model.parameters()).device}", flush=True)
    return model, processor


def eef_pos(obs: Dict[str, Any]) -> Optional[np.ndarray]:
    val = obs.get("robot0_eef_pos")
    if val is None:
        return None
    return np.asarray(val, dtype=float).reshape(-1)[:3]


def unit(v: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return None
    return v / norm


def cosine(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    ua, ub = unit(a), unit(b)
    if ua is None or ub is None:
        return None
    return float(np.clip(np.dot(ua, ub), -1.0, 1.0))


def site_pos(env: Any, target: str) -> Tuple[Optional[np.ndarray], Optional[str], str]:
    site_name = TARGET_TO_SITE.get(str(target or "").lower())
    if not site_name:
        return None, None, "missing_target_mapping"
    try:
        model = env.sim.model
        data = env.sim.data
        sid = model.site_name2id(site_name)
        return np.asarray(data.site_xpos[sid], dtype=float).copy(), site_name, "ok"
    except Exception as exc:
        return None, site_name, f"site_lookup_error:{type(exc).__name__}:{exc}"


def counterfactual_for_task(task_id: int, description: str) -> Optional[Dict[str, Any]]:
    text = description.lower()
    if "black bowl" not in text:
        return None
    if "ramekin" in text:
        target = "ramekin"
    elif "cookie box" in text:
        target = "cookie box"
    elif "top drawer" in text:
        target = "top drawer"
    else:
        return None
    return {
        "task_id": task_id,
        "original_target_object": "black bowl",
        "counterfactual_target_object": target,
        "original_instruction": description,
        "counterfactual_instruction": f"pick up the {target} and place it on the plate",
        "schema": {
            "target_object": target,
            "target_exists": True,
            "phase": "approach",
            "allowed_actions": ["MOVE_TO", "APPROACH", "GRASP", "PICK", "PLACE"],
            "blocked_actions": ["MOVE_TO_BLACK_BOWL", "GRASP_BLACK_BOWL", "PICK_BLACK_BOWL"],
            "next_action": "MOVE_TO",
            "reason": "Counterfactual target swap schema used for action-interface projection.",
            "confidence": 1.0,
        },
    }


def project_action(
    action_raw: List[float],
    ee: Optional[np.ndarray],
    target: Optional[np.ndarray],
    alpha: float,
    alignment_threshold: float,
    project_step_scale: float,
    min_step_scale: float,
    always_project: bool,
) -> Tuple[List[float], Dict[str, Any]]:
    arr = np.asarray(action_raw, dtype=float).reshape(-1)
    if arr.size < 7:
        arr = np.pad(arr, (0, 7 - arr.size))
    arr = arr[:7].copy()
    meta: Dict[str, Any] = {
        "projection_applied": False,
        "raw_alignment_to_target": None,
        "raw_pos_norm": float(np.linalg.norm(arr[:3])),
    }
    if ee is None or target is None:
        meta["projection_reason"] = "missing_position"
        return [float(x) for x in arr.tolist()], meta
    target_vec = np.asarray(target, dtype=float) - np.asarray(ee, dtype=float)
    target_dir = unit(target_vec)
    if target_dir is None:
        meta["projection_reason"] = "zero_target_direction"
        return [float(x) for x in arr.tolist()], meta
    raw_pos = arr[:3].copy()
    align = cosine(raw_pos, target_dir)
    meta["raw_alignment_to_target"] = align
    if not always_project and align is not None and align >= alignment_threshold:
        meta["projection_reason"] = "already_aligned"
        return [float(x) for x in arr.tolist()], meta
    step_scale = max(float(np.linalg.norm(raw_pos)), float(min_step_scale))
    desired = target_dir * step_scale * float(project_step_scale)
    arr[:3] = (1.0 - alpha) * raw_pos + alpha * desired
    meta["projection_applied"] = True
    meta["projection_reason"] = "projected_to_schema_target"
    meta["projected_alignment_to_target"] = cosine(arr[:3], target_dir)
    meta["projected_pos_norm"] = float(np.linalg.norm(arr[:3]))
    return [float(x) for x in arr.tolist()], meta


def primitive_env_action(phase: str, args: argparse.Namespace) -> List[float]:
    """Small proof-of-concept manipulation primitive in environment action space."""
    if phase == "descend":
        return [0.0, 0.0, float(args.phase_descend_dz), 0.0, 0.0, 0.0, -1.0]
    if phase == "close":
        # LIBERO / robosuite gripper convention: positive closes the gripper.
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    if phase == "lift":
        return [0.0, 0.0, float(args.phase_lift_dz), 0.0, 0.0, 0.0, 1.0]
    return dummy_action()


def rollout(
    env: Any,
    init_state: Any,
    instruction: str,
    schema_target: str,
    model: Any,
    processor: Any,
    args: argparse.Namespace,
    method: str,
) -> Dict[str, Any]:
    done = False
    error = None
    step_trace: List[Dict[str, Any]] = []
    distances: List[float] = []
    projection_count = 0
    reacquire_count = 0
    actions_taken = 0
    release_step = None
    released = False
    adapter_phase = "approach"
    adapter_phase_step = 0
    first_action = None
    last_action = None
    target_site = None
    target_status = None
    target_start_z = None
    target_final_z = None
    target_max_z = None
    try:
        env.reset()
        obs = env.set_init_state(init_state)
        for t in range(args.max_steps + args.num_steps_wait):
            ee_before = eef_pos(obs)
            target_pos, target_site, target_status = site_pos(env, schema_target)
            if target_pos is not None:
                z = float(target_pos[2])
                if target_start_z is None and t >= args.num_steps_wait:
                    target_start_z = z
                target_final_z = z
                target_max_z = z if target_max_z is None else max(target_max_z, z)
            dist_before = None if ee_before is None or target_pos is None else float(np.linalg.norm(ee_before - target_pos))
            if dist_before is not None and t >= args.num_steps_wait:
                distances.append(dist_before)
            if t < args.num_steps_wait:
                env_action = dummy_action()
                raw_action = None
                projected_action = None
                meta = {"projection_applied": False, "projection_reason": "warmup"}
            else:
                img = get_image(obs, center_crop=args.center_crop)
                raw_action = predict_action(model, processor, img, instruction, args)
                first_action = first_action or raw_action
                last_action = raw_action
                projected_action = raw_action
                meta = {"projection_applied": False, "projection_reason": "raw"}
                if method == "schema_projection":
                    mode = "always" if args.always_project and args.projection_mode == "alignment" else args.projection_mode
                    if mode == "phase_adapter":
                        if adapter_phase == "approach" and dist_before is not None and dist_before <= args.release_distance:
                            released = True
                            if release_step is None:
                                release_step = t - args.num_steps_wait
                            adapter_phase = "descend"
                            adapter_phase_step = 0
                        if adapter_phase == "approach":
                            projected_action, meta = project_action(
                                raw_action,
                                ee_before,
                                target_pos,
                                args.alpha,
                                args.alignment_threshold,
                                args.project_step_scale,
                                args.min_step_scale,
                                True,
                            )
                            env_action = normalize_for_env(projected_action)
                        else:
                            if adapter_phase == "descend" and adapter_phase_step >= args.phase_descend_steps:
                                adapter_phase = "close"
                                adapter_phase_step = 0
                            if adapter_phase == "close" and adapter_phase_step >= args.phase_close_steps:
                                adapter_phase = "lift"
                                adapter_phase_step = 0
                            if adapter_phase == "lift" and adapter_phase_step >= args.phase_lift_steps:
                                adapter_phase = "raw_after_lift"
                                adapter_phase_step = 0
                            if adapter_phase in {"descend", "close", "lift"}:
                                env_action = primitive_env_action(adapter_phase, args)
                            else:
                                env_action = normalize_for_env(raw_action)
                            projected_action = raw_action
                            meta = {
                                "projection_applied": False,
                                "projection_reason": f"phase_adapter_{adapter_phase}",
                                "released": True,
                                "adapter_phase": adapter_phase,
                                "adapter_phase_step": adapter_phase_step,
                                "release_distance": args.release_distance,
                            }
                            adapter_phase_step += 1
                        if adapter_phase == "approach":
                            meta["adapter_phase"] = adapter_phase
                            meta["adapter_phase_step"] = adapter_phase_step
                            meta["released"] = False
                            meta["release_distance"] = args.release_distance
                            projection_count += int(bool(meta.get("projection_applied")))
                            env_action = normalize_for_env(projected_action)
                        actions_taken += 1
                        obs, reward, done, info = env.step(env_action)
                        ee_after = eef_pos(obs)
                        dist_after = None if ee_after is None or target_pos is None else float(np.linalg.norm(ee_after - target_pos))
                        if args.save_step_traces and t >= args.num_steps_wait:
                            step_trace.append(
                                {
                                    "step": t - args.num_steps_wait,
                                    "ee_before": ee_before.tolist() if ee_before is not None else None,
                                    "ee_after": ee_after.tolist() if ee_after is not None else None,
                                    "target_pos": target_pos.tolist() if target_pos is not None else None,
                                    "distance_before": dist_before,
                                    "distance_after": dist_after,
                                    "raw_action": raw_action,
                                    "projected_action": projected_action,
                                    **meta,
                                }
                            )
                        if done:
                            break
                        continue
                    if mode in {"until_close", "hybrid_until_close"} and dist_before is not None and dist_before <= args.release_distance:
                        released = True
                        if release_step is None:
                            release_step = t - args.num_steps_wait
                        meta = {
                            "projection_applied": False,
                            "projection_reason": "released_until_close",
                            "released": True,
                            "release_distance": args.release_distance,
                        }
                    elif mode == "hybrid_until_close" and released and dist_before is not None and dist_before <= args.reacquire_distance:
                        meta = {
                            "projection_applied": False,
                            "projection_reason": "released_hysteresis",
                            "released": True,
                            "release_distance": args.release_distance,
                            "reacquire_distance": args.reacquire_distance,
                        }
                    elif mode == "hybrid_until_close" and released and dist_before is not None and dist_before > args.reacquire_distance:
                        projected_action, meta = project_action(
                            raw_action,
                            ee_before,
                            target_pos,
                            args.alpha,
                            args.alignment_threshold,
                            args.project_step_scale,
                            args.min_step_scale,
                            True,
                        )
                        meta["released"] = False
                        meta["release_distance"] = args.release_distance
                        meta["reacquired"] = True
                        meta["reacquire_distance"] = args.reacquire_distance
                        reacquire_count += 1
                    else:
                        projected_action, meta = project_action(
                            raw_action,
                            ee_before,
                            target_pos,
                            args.alpha,
                            args.alignment_threshold,
                            args.project_step_scale,
                            args.min_step_scale,
                            mode in {"always", "until_close"},
                        )
                        meta["released"] = False
                        meta["release_distance"] = args.release_distance if mode in {"until_close", "hybrid_until_close"} else None
                        meta["reacquire_distance"] = args.reacquire_distance if mode == "hybrid_until_close" else None
                    projection_count += int(bool(meta.get("projection_applied")))
                env_action = normalize_for_env(projected_action)
                actions_taken += 1
            obs, reward, done, info = env.step(env_action)
            ee_after = eef_pos(obs)
            dist_after = None if ee_after is None or target_pos is None else float(np.linalg.norm(ee_after - target_pos))
            if args.save_step_traces and t >= args.num_steps_wait:
                step_trace.append(
                    {
                        "step": t - args.num_steps_wait,
                        "ee_before": ee_before.tolist() if ee_before is not None else None,
                        "ee_after": ee_after.tolist() if ee_after is not None else None,
                        "target_pos": target_pos.tolist() if target_pos is not None else None,
                        "distance_before": dist_before,
                        "distance_after": dist_after,
                        "raw_action": raw_action,
                        "projected_action": projected_action,
                        **meta,
                    }
                )
            if done:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    finite = [d for d in distances if d is not None and not math.isnan(float(d))]
    final_dist = finite[-1] if finite else None
    min_dist = min(finite) if finite else None
    start_dist = finite[0] if finite else None
    decreases = 0
    total_pairs = 0
    for a, b in zip(finite, finite[1:]):
        decreases += int(b < a)
        total_pairs += 1
    return {
        "method": method,
        "instruction": instruction,
        "schema_target": schema_target,
        "target_site": target_site,
        "target_position_status": target_status,
        "success": bool(done and error is None),
        "done": bool(done),
        "error": error,
        "actions_taken": actions_taken,
        "projection_count": projection_count,
        "reacquire_count": reacquire_count,
        "projection_rate": projection_count / actions_taken if actions_taken else None,
        "projection_mode": ("always" if args.always_project and args.projection_mode == "alignment" else args.projection_mode) if method == "schema_projection" else "raw",
        "release_distance": args.release_distance if method == "schema_projection" else None,
        "reacquire_distance": args.reacquire_distance if method == "schema_projection" else None,
        "released": released,
        "release_step": release_step,
        "adapter_final_phase": adapter_phase if method == "schema_projection" else None,
        "start_distance_to_target": start_dist,
        "final_distance_to_target": final_dist,
        "min_distance_to_target": min_dist,
        "close_but_fail": bool(min_dist is not None and min_dist <= args.release_distance and not (done and error is None)) if method == "schema_projection" else None,
        "target_start_z": target_start_z,
        "target_final_z": target_final_z,
        "target_max_z": target_max_z,
        "target_lift_delta": None if target_start_z is None or target_max_z is None else float(target_max_z - target_start_z),
        "target_lifted": None if target_start_z is None or target_max_z is None else bool((target_max_z - target_start_z) >= args.lift_success_dz),
        "target_approach_rate": decreases / total_pairs if total_pairs else None,
        "first_action": first_action,
        "last_action": last_action,
        "step_trace": step_trace if args.save_step_traces else None,
    }


def mean_or_none(vals: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    return float(np.mean(clean)) if clean else None


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark

    model, processor = load_model_and_processor_with_diagnostics(args)
    injected_stats = inject_local_dataset_statistics(model, args.model_path, args.unnorm_key)
    if injected_stats:
        print(f"[info] injected local dataset_statistics.json for unnorm_key={args.unnorm_key}", flush=True)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if args.task_ids.strip():
        task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()]
    else:
        task_ids = []
        for tid in range(min(args.max_tasks, suite.n_tasks)):
            cf = counterfactual_for_task(tid, suite.get_task(tid).language)
            if cf is not None:
                task_ids.append(tid)

    rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        cf = counterfactual_for_task(task_id, task.language)
        if cf is None:
            print(f"[warn] skipping task_id={task_id}: no target-swap mapping for {task.language}", flush=True)
            continue
        initial_states = suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=256)
        for trial_idx in range(min(args.num_trials_per_task, len(initial_states))):
            init_state = initial_states[trial_idx]
            raw = rollout(env, init_state, cf["counterfactual_instruction"], cf["counterfactual_target_object"], model, processor, args, "openvla_raw_cf")
            proj = rollout(env, init_state, cf["counterfactual_instruction"], cf["counterfactual_target_object"], model, processor, args, "schema_projection")
            base = {
                "task_suite": args.task_suite_name,
                "task_id": task_id,
                "trial_idx": trial_idx,
                "task_description": task_description,
                "original_instruction": cf["original_instruction"],
                "counterfactual_instruction": cf["counterfactual_instruction"],
                "original_target_object": cf["original_target_object"],
                "counterfactual_target_object": cf["counterfactual_target_object"],
                "schema": cf["schema"],
            }
            raw_row = {**base, **raw}
            proj_row = {**base, **proj}
            rows.extend([raw_row, proj_row])
            redirection = {
                **base,
                "raw_final_distance_to_target": raw.get("final_distance_to_target"),
                "projection_final_distance_to_target": proj.get("final_distance_to_target"),
                "final_distance_redirection_score": None
                if raw.get("final_distance_to_target") is None or proj.get("final_distance_to_target") is None
                else float(raw["final_distance_to_target"] - proj["final_distance_to_target"]),
                "raw_min_distance_to_target": raw.get("min_distance_to_target"),
                "projection_min_distance_to_target": proj.get("min_distance_to_target"),
                "min_distance_redirection_score": None
                if raw.get("min_distance_to_target") is None or proj.get("min_distance_to_target") is None
                else float(raw["min_distance_to_target"] - proj["min_distance_to_target"]),
                "raw_target_approach_rate": raw.get("target_approach_rate"),
                "projection_target_approach_rate": proj.get("target_approach_rate"),
                "projection_rate": proj.get("projection_rate"),
                "projection_mode": proj.get("projection_mode"),
                "release_distance": proj.get("release_distance"),
                "reacquire_distance": proj.get("reacquire_distance"),
                "released": proj.get("released"),
                "release_step": proj.get("release_step"),
                "reacquire_count": proj.get("reacquire_count"),
                "close_but_fail": proj.get("close_but_fail"),
                "adapter_final_phase": proj.get("adapter_final_phase"),
                "target_lift_delta": proj.get("target_lift_delta"),
                "target_lifted": proj.get("target_lifted"),
                "raw_success": raw.get("success"),
                "projection_success": proj.get("success"),
                "target_position_status": proj.get("target_position_status"),
            }
            pair_rows.append(redirection)
            print(json.dumps(redirection), flush=True)
        env.close()

    summary_rows: List[Dict[str, Any]] = []
    for method in ["openvla_raw_cf", "schema_projection"]:
        mrows = [r for r in rows if r.get("method") == method]
        summary_rows.append(
            {
                "method": method,
                "n": len(mrows),
                "success_rate": sum(bool(r.get("success")) for r in mrows) / len(mrows) if mrows else None,
                "mean_start_distance_to_target": mean_or_none(r.get("start_distance_to_target") for r in mrows),
                "mean_final_distance_to_target": mean_or_none(r.get("final_distance_to_target") for r in mrows),
                "mean_min_distance_to_target": mean_or_none(r.get("min_distance_to_target") for r in mrows),
                "mean_target_approach_rate": mean_or_none(r.get("target_approach_rate") for r in mrows),
                "mean_projection_rate": mean_or_none(r.get("projection_rate") for r in mrows),
            }
        )
    summary = {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "task_suite": args.task_suite_name,
        "task_ids": task_ids,
        "model_path": args.model_path,
        "unnorm_key": args.unnorm_key,
        "injected_local_dataset_statistics": injected_stats,
        "alpha": args.alpha,
        "alignment_threshold": args.alignment_threshold,
        "project_step_scale": args.project_step_scale,
        "projection_mode": args.projection_mode,
        "always_project": args.always_project,
        "release_distance": args.release_distance,
        "reacquire_distance": args.reacquire_distance,
        "phase_descend_steps": args.phase_descend_steps,
        "phase_close_steps": args.phase_close_steps,
        "phase_lift_steps": args.phase_lift_steps,
        "phase_descend_dz": args.phase_descend_dz,
        "phase_lift_dz": args.phase_lift_dz,
        "lift_success_dz": args.lift_success_dz,
        "max_steps": args.max_steps,
        "interpretation": "Schema projection intervenes at the action interface; positive redirection scores mean the projected rollout ends closer to the schema target than raw counterfactual rollout.",
        "method_summary": summary_rows,
        "redirection_summary": {
            "n": len(pair_rows),
            "mean_final_distance_redirection_score": mean_or_none(r.get("final_distance_redirection_score") for r in pair_rows),
            "mean_min_distance_redirection_score": mean_or_none(r.get("min_distance_redirection_score") for r in pair_rows),
            "mean_projection_rate": mean_or_none(r.get("projection_rate") for r in pair_rows),
            "release_rate": mean_or_none(float(bool(r.get("released"))) for r in pair_rows),
            "mean_release_step": mean_or_none(r.get("release_step") for r in pair_rows),
            "mean_reacquire_count": mean_or_none(r.get("reacquire_count") for r in pair_rows),
            "close_but_fail_rate": mean_or_none(float(bool(r.get("close_but_fail"))) for r in pair_rows),
            "target_lift_rate": mean_or_none(float(bool(r.get("target_lifted"))) for r in pair_rows),
            "mean_target_lift_delta": mean_or_none(r.get("target_lift_delta") for r in pair_rows),
        },
    }
    write_jsonl(out_dir / "rollout_predictions.jsonl", rows)
    write_jsonl(out_dir / "redirection_pairs.jsonl", pair_rows)
    write_csv(out_dir / "method_summary.csv", summary_rows)
    write_csv(out_dir / "redirection_summary.csv", pair_rows)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
