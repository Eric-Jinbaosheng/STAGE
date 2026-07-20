#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in [REPO_ROOT / "third_party" / "LIBERO_pkg", REPO_ROOT / "scripts", REPO_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lb_run_exp6_schema_action_projection import counterfactual_for_task, eef_pos, site_pos  # noqa: E402
from lb_run_visa_closed_loop_execution import load_counterfactual_specs  # noqa: E402
from lb_run_libero10_smoke import get_libero_env  # noqa: E402
from linguistic_blindness.utils.io import command_string, utc_timestamp, write_csv, write_json, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate counterfactual contact/lift evaluator with direct simulator state interventions.")
    ap.add_argument("--task-suite-name", default="libero_spatial")
    ap.add_argument("--task-ids", default="1,3,4,5,6")
    ap.add_argument("--counterfactual-spec-file", default="")
    ap.add_argument("--trial-idx", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs/linguistic_blindness/counterfactual_evaluator_sanity")
    ap.add_argument("--contact-distance", type=float, default=0.06)
    ap.add_argument("--lift-success-dz", type=float, default=0.03)
    ap.add_argument("--lift-dz", type=float, default=0.06)
    return ap.parse_args()


def site_body_and_free_joint(env: Any, target: str) -> Tuple[Optional[str], Optional[int], Optional[int], str]:
    pos, site_name, status = site_pos(env, target)
    if pos is None or site_name is None:
        return site_name, None, None, status
    model = env.sim.model
    try:
        sid = model.site_name2id(site_name)
        body = int(model.site_bodyid[sid])
        while body != 0:
            for jid in range(model.njnt):
                if int(model.jnt_bodyid[jid]) == body and int(model.jnt_type[jid]) == 0:
                    return site_name, jid, int(model.jnt_qposadr[jid]), "ok"
            body = int(model.body_parentid[body])
    except Exception as exc:
        return site_name, None, None, f"joint_lookup_error:{type(exc).__name__}:{exc}"
    return site_name, None, None, "no_free_joint_found"


def set_free_joint_xyz(env: Any, qposadr: int, xyz: np.ndarray) -> None:
    env.sim.data.qpos[qposadr : qposadr + 3] = np.asarray(xyz, dtype=float).reshape(3)
    env.sim.forward()


def current_eef_pos(env: Any) -> Optional[np.ndarray]:
    ee = None
    if hasattr(env, "_get_observations"):
        try:
            ee = eef_pos(env._get_observations())
        except Exception:
            ee = None
    if ee is None:
        for site_name in ["gripper0_grip_site", "robot0_grip_site", "gripper_site"]:
            try:
                return np.asarray(env.sim.data.get_site_xpos(site_name), dtype=float).reshape(3)
            except Exception:
                pass
    return ee


def eval_state(env: Any, target: str, start_z: Optional[float], args: argparse.Namespace) -> Dict[str, Any]:
    pos, site_name, site_status = site_pos(env, target)
    ee = current_eef_pos(env)
    dist = None if pos is None or ee is None else float(np.linalg.norm(pos - ee))
    z = None if pos is None else float(pos[2])
    lift_delta = None if z is None or start_z is None else float(z - start_z)
    return {
        "site_name": site_name,
        "site_status": site_status,
        "target_z": z,
        "eef_to_target_dist": dist,
        "correct_contact": None if dist is None else dist <= args.contact_distance,
        "correct_lift_delta": lift_delta,
        "correct_lift": None if lift_delta is None else lift_delta >= args.lift_success_dz,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()]
    manual_specs = load_counterfactual_specs(args.counterfactual_spec_file)
    rows: List[Dict[str, Any]] = []

    for task_id in task_ids:
        task = suite.get_task(task_id)
        cf = manual_specs.get(task_id) or counterfactual_for_task(task_id, task.language)
        if cf is None:
            rows.append({"task_id": task_id, "error": "no_counterfactual_mapping"})
            continue
        init_states = suite.get_task_init_states(task_id)
        env, desc = get_libero_env(task, resolution=128)
        try:
            env.reset()
            env.set_init_state(init_states[args.trial_idx])
            target = cf["counterfactual_target_object"]
            pos0, _, _ = site_pos(env, target)
            ee0 = current_eef_pos(env)
            site_name, jid, qposadr, joint_status = site_body_and_free_joint(env, target)
            start_z = None if pos0 is None else float(pos0[2])
            base = eval_state(env, target, start_z, args)

            contact_eval = lift_eval = None
            if qposadr is not None and ee0 is not None:
                contact_xyz = np.asarray(ee0, dtype=float).reshape(3)
                set_free_joint_xyz(env, qposadr, contact_xyz)
                contact_eval = eval_state(env, target, start_z, args)

                lift_xyz = contact_xyz.copy()
                lift_xyz[2] = (start_z if start_z is not None else lift_xyz[2]) + args.lift_dz
                set_free_joint_xyz(env, qposadr, lift_xyz)
                lift_eval = eval_state(env, target, start_z, args)

            row = {
                "task_suite": args.task_suite_name,
                "task_id": task_id,
                "trial_idx": args.trial_idx,
                "task_description": desc,
                "counterfactual_target_object": target,
                "site_name": site_name,
                "free_joint_id": jid,
                "free_joint_qposadr": qposadr,
                "joint_status": joint_status,
                "baseline_correct_contact": None if base is None else base["correct_contact"],
                "baseline_correct_lift": None if base is None else base["correct_lift"],
                "oracle_contact_correct_contact": None if contact_eval is None else contact_eval["correct_contact"],
                "oracle_contact_distance": None if contact_eval is None else contact_eval["eef_to_target_dist"],
                "oracle_lift_correct_lift": None if lift_eval is None else lift_eval["correct_lift"],
                "oracle_lift_delta": None if lift_eval is None else lift_eval["correct_lift_delta"],
                "base_eval": base,
                "contact_eval": contact_eval,
                "lift_eval": lift_eval,
            }
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ["task_id", "counterfactual_target_object", "joint_status", "baseline_correct_contact", "oracle_contact_correct_contact", "oracle_lift_correct_lift", "oracle_lift_delta"]}, sort_keys=True), flush=True)
        finally:
            env.close()

    valid = [r for r in rows if not r.get("error")]
    summary = {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "task_suite": args.task_suite_name,
        "n": len(valid),
        "contact_oracle_pass_rate": None if not valid else sum(r.get("oracle_contact_correct_contact") is True for r in valid) / len(valid),
        "lift_oracle_pass_rate": None if not valid else sum(r.get("oracle_lift_correct_lift") is True for r in valid) / len(valid),
        "rows": rows,
    }
    write_jsonl(out_dir / "counterfactual_evaluator_sanity.jsonl", rows)
    write_csv(out_dir / "counterfactual_evaluator_sanity.csv", rows)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
