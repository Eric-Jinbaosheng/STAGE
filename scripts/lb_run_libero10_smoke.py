#!/usr/bin/env python3
"""Minimal LIBERO closed-loop smoke for OpenVLA / simulator wiring.

This is intentionally a smoke test, not a paper metric script. It can run either
with dummy actions (`--no-model`) to validate LIBERO reset/step/render, or with
OpenVLA actions for a tiny rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in [REPO_ROOT / "third_party" / "LIBERO_pkg", REPO_ROOT / "third_party" / "openvla", REPO_ROOT / "scripts", REPO_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lb_run_exp2_action_sensitivity import action_to_list, load_model_and_processor, predict_action  # noqa: E402
from linguistic_blindness.utils.io import command_string, utc_timestamp, write_json, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a tiny LIBERO-10 simulator smoke test.")
    p.add_argument("--task-suite-name", default="libero_10")
    p.add_argument("--max-tasks", type=int, default=1)
    p.add_argument("--num-trials-per-task", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--model-path", default="/scratch/bj2410/models/openvla-7b")
    p.add_argument("--unnorm-key", default="bridge_orig")
    p.add_argument("--out-dir", default="outputs/linguistic_blindness/exp6_libero10_smoke")
    p.add_argument("--no-model", action="store_true", help="Use dummy actions only to validate simulator.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--attn-implementation", default="")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--center-crop", action="store_true", help="Match OpenVLA LIBERO eval for image-aug fine-tuned checkpoints.")
    return p.parse_args()


def dummy_action() -> List[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def center_crop_image(img: np.ndarray, crop_scale: float = 0.9) -> np.ndarray:
    """PIL/NumPy implementation of the OpenVLA LIBERO center-crop path."""
    from PIL import Image

    h, w = img.shape[:2]
    crop_h = int(round(h * np.sqrt(crop_scale)))
    crop_w = int(round(w * np.sqrt(crop_scale)))
    y0 = max((h - crop_h) // 2, 0)
    x0 = max((w - crop_w) // 2, 0)
    cropped = img[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return np.asarray(Image.fromarray(cropped).resize((w, h), Image.Resampling.BICUBIC))


def get_image(obs: Dict[str, Any], center_crop: bool = False, resize_size: Tuple[int, int] = (224, 224)) -> Any:
    from PIL import Image

    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # Match OpenVLA LIBERO preprocessing convention.
    if center_crop:
        img = center_crop_image(img)
    return Image.fromarray(img).convert("RGB").resize(resize_size)


def default_max_steps(task_suite_name: str) -> int:
    # Match OpenVLA's official LIBERO eval step budgets.
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }.get(task_suite_name, 520)


def get_libero_env(task: Any, resolution: int = 256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def normalize_for_env(action: List[float]) -> List[float]:
    # Match OpenVLA LIBERO eval convention: gripper [0, 1] -> [-1, 1], then invert.
    arr = np.asarray(action, dtype=float).reshape(-1)[:7]
    if arr.size < 7:
        arr = np.pad(arr, (0, 7 - arr.size))
    arr[-1] = 2 * (arr[-1] - 0.0) - 1
    arr[-1] = np.sign(arr[-1]) if arr[-1] != 0 else -1
    arr[-1] *= -1.0
    return [float(x) for x in arr.tolist()]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Imports are delayed so PYTHONPATH/LIBERO_CONFIG_PATH can be configured by sbatch.
    from libero.libero import benchmark

    model = processor = None
    if not args.no_model:
        model, processor = load_model_and_processor(args)

    max_steps = args.max_steps if args.max_steps is not None else default_max_steps(args.task_suite_name)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    n_tasks = min(args.max_tasks, task_suite.n_tasks)
    rows: List[Dict[str, Any]] = []
    total_success = 0
    total = 0

    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=256)
        for trial_idx in range(min(args.num_trials_per_task, len(initial_states))):
            done = False
            error = None
            t = 0
            actions_taken = 0
            first_action = None
            try:
                env.reset()
                obs = env.set_init_state(initial_states[trial_idx])
                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait or args.no_model:
                        action = dummy_action()
                    else:
                        img = get_image(obs, center_crop=args.center_crop)
                        action_raw = predict_action(model, processor, img, task_description, args)
                        first_action = first_action or action_raw
                        action = normalize_for_env(action_raw)
                    obs, reward, done, info = env.step(action)
                    actions_taken += int(t >= args.num_steps_wait)
                    t += 1
                    if done:
                        break
            except Exception as e:  # Keep smoke output machine-readable even on simulator errors.
                error = f"{type(e).__name__}: {e}"
            row = {
                "task_suite": args.task_suite_name,
                "task_id": task_id,
                "trial_idx": trial_idx,
                "task_description": task_description,
                "method": "dummy" if args.no_model else "openvla_raw",
                "success": bool(done and error is None),
                "done": bool(done),
                "error": error,
                "steps": t,
                "actions_taken": actions_taken,
                "first_action": first_action,
            }
            rows.append(row)
            total += 1
            total_success += int(row["success"])
            print(json.dumps(row), flush=True)
        env.close()

    summary = {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "task_suite": args.task_suite_name,
        "num_rollouts": total,
        "successes": total_success,
        "success_rate": total_success / total if total else None,
        "no_model": args.no_model,
        "max_steps": max_steps,
        "num_steps_wait": args.num_steps_wait,
        "unnorm_key": args.unnorm_key,
        "center_crop": args.center_crop,
    }
    write_jsonl(out_dir / "rollout_predictions.jsonl", rows)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
