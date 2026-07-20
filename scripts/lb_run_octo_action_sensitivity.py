#!/usr/bin/env python3
"""Octo native action sensitivity probe for target-swap examples.

This is an architecture-level action-output diagnostic. Octo is not finetuned to
LIBERO here, so the result should be interpreted as native action sensitivity,
not LIBERO task success or calibrated control quality.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import re
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in [REPO_ROOT / "scripts", REPO_ROOT / "src", REPO_ROOT / "third_party" / "octo"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lb_run_exp2_action_sensitivity import (  # noqa: E402
    action_stats,
    by_example_id,
    load_index_maps,
    paraphrase_instruction,
    qwen_schema_sensitive,
    resolve_obs_ptr,
    threshold_from_controls,
)
from lb_run_openvla_action import load_hdf5_image_pointer  # noqa: E402
from linguistic_blindness.utils.io import command_string, read_jsonl, utc_timestamp, write_csv, write_json, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Octo native target-swap action sensitivity.")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--qwen-predictions", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--octo-checkpoint", default="hf://rail-berkeley/octo-small-1.5")
    p.add_argument("--index", default="data/processed/unified_index.parquet")
    p.add_argument("--max-examples", type=int, default=100)
    p.add_argument("--dataset-filter", default="libero")
    p.add_argument("--perturbation-types", default="target_swap")
    p.add_argument("--require-counterfactual-valid", action="store_true")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--sample-seed", type=int, default=7)
    p.add_argument("--control-mode", choices=["none", "same", "paraphrase", "both"], default="paraphrase")
    p.add_argument("--threshold-method", choices=["p95", "mean_plus_2std", "fixed"], default="p95")
    p.add_argument("--fixed-threshold", type=float, default=1e-6)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--octo-seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=50)
    return p.parse_args()


def load_image_for_example(example: Dict[str, Any], obs_map: Dict[str, str], image_size: int) -> tuple[np.ndarray, str]:
    ptr = resolve_obs_ptr(example, obs_map)
    if not ptr:
        raise FileNotFoundError(f"No obs_ptr for {example.get('example_id')}")
    if "#" in ptr:
        img = load_hdf5_image_pointer(ptr)
        if img is None:
            raise FileNotFoundError(f"Could not load HDF5 image {ptr}")
    else:
        img = Image.open(ptr).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8), ptr


def make_observation(image: np.ndarray) -> Dict[str, Any]:
    # Octo expects batch and time axes. We provide a single-frame history.
    return {
        "image_primary": image[None, None],
        "timestep_pad_mask": np.array([[True]], dtype=bool),
    }


def flatten_action(action: Any) -> List[float]:
    arr = np.asarray(action, dtype=float)
    # Octo usually returns [batch, horizon, action_dim]. Use first action in the chunk.
    if arr.ndim >= 3:
        arr = arr[0, 0]
    elif arr.ndim == 2:
        arr = arr[0]
    arr = arr.reshape(-1)
    return [float(x) for x in arr.tolist()]


def load_octo_model(checkpoint: str):
    from octo.model.octo_model import OctoModel

    print(f"loading Octo checkpoint {checkpoint}", flush=True)
    model = OctoModel.load_pretrained(checkpoint)
    print(model.get_pretty_spec(), flush=True)
    return model


def predict_octo_action(model: Any, image: np.ndarray, instruction: str, seed: int) -> List[float]:
    import jax

    observation = make_observation(image)
    task = model.create_tasks(texts=[instruction])
    rng = jax.random.PRNGKey(seed)
    action = model.sample_actions(observation, task, rng=rng)
    return flatten_action(action)


def action_std_for_rows(rows: List[Dict[str, Any]]) -> np.ndarray:
    actions = []
    for r in rows:
        for k in ["octo_action_original", "octo_action_counterfactual", "control_action"]:
            if r.get(k) is not None:
                arr = np.asarray(r[k], dtype=float).reshape(-1)
                if arr.size:
                    actions.append(arr)
    if not actions:
        return np.ones(7, dtype=float)
    max_dim = max(a.size for a in actions)
    stack = []
    for a in actions:
        if a.size < max_dim:
            a = np.pad(a, (0, max_dim - a.size), mode="constant")
        stack.append(a[:max_dim])
    std = np.std(np.stack(stack, axis=0), axis=0)
    return np.where(std < 1e-6, 1.0, std).astype(float)


def action_stats_dynamic(a: List[float], b: List[float], std: np.ndarray) -> Dict[str, float]:
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    dim = max(aa.size, bb.size, std.size)
    aa = np.pad(aa, (0, max(0, dim - aa.size)), mode="constant")[:dim]
    bb = np.pad(bb, (0, max(0, dim - bb.size)), mode="constant")[:dim]
    ss = np.pad(std, (0, max(0, dim - std.size)), constant_values=1.0)[:dim]
    diff = aa - bb
    first3 = min(3, dim)
    next3 = min(6, dim)
    pos = float(np.linalg.norm(diff[:first3]))
    rot = float(np.linalg.norm(diff[3:next3])) if dim > 3 else 0.0
    grip = float(abs(diff[6])) if dim > 6 else 0.0
    norm = float(np.linalg.norm(diff / ss))
    denom = float(np.linalg.norm(aa[:next3]) * np.linalg.norm(bb[:next3]))
    cosine = None if denom < 1e-12 else float(np.dot(aa[:next3], bb[:next3]) / denom)
    return {"delta_pos": pos, "delta_rot": rot, "delta_gripper": grip, "normalized_action_delta": norm, "action_cosine": cosine}


def aggregate_bool(vals: Iterable[Any]) -> Dict[str, Any]:
    clean = [v for v in vals if v is not None]
    if not clean:
        return {"mean": None, "count": 0}
    return {"mean": sum(bool(v) for v in clean) / len(clean), "count": len(clean)}


def aggregate_float(vals: Iterable[Any]) -> Dict[str, Any]:
    clean = []
    for v in vals:
        if v is None:
            continue
        fv = float(v)
        if not math.isnan(fv):
            clean.append(fv)
    if not clean:
        return {"mean": None, "count": 0}
    return {"mean": sum(clean) / len(clean), "count": len(clean)}


def summarize(rows: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    for r in rows:
        sens = r["normalized_action_delta"] >= threshold
        r["octo_action_sensitive"] = bool(sens)
        r["low_action_sensitivity"] = not bool(sens)
        r["semantic_action_gap_case"] = bool(r.get("qwen_schema_sensitive") is True and not sens)
        r["action_sensitivity_threshold"] = threshold
    schema = aggregate_bool(r.get("qwen_schema_sensitive") for r in rows)
    action = aggregate_bool(r.get("octo_action_sensitive") for r in rows)
    gap = None if schema["mean"] is None or action["mean"] is None else schema["mean"] - action["mean"]
    return {
        "num_examples": len(rows),
        "action_sensitivity_threshold": threshold,
        "vlm_schema_sensitivity": schema,
        "octo_action_sensitivity": action,
        "semantic_action_gap": {"mean": gap, "count": min(schema["count"], action["count"])},
        "low_sensitivity_rate": aggregate_bool(r.get("low_action_sensitivity") for r in rows),
        "semantic_action_gap_case_rate": aggregate_bool(r.get("semantic_action_gap_case") for r in rows),
        "mean_normalized_action_delta": aggregate_float(r.get("normalized_action_delta") for r in rows),
        "mean_control_normalized_action_delta": aggregate_float(r.get("control_normalized_action_delta") for r in rows),
        "mean_action_cosine": aggregate_float(r.get("action_cosine") for r in rows),
        "action_dim": len(rows[0]["octo_action_original"]) if rows else None,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ptypes = {x.strip() for x in args.perturbation_types.split(",") if x.strip()}
    datasets = {x.strip() for x in args.dataset_filter.split(",") if x.strip()}
    qwen_cf = by_example_id(read_jsonl(args.qwen_predictions))
    examples = [
        x for x in read_jsonl(args.benchmark)
        if (not ptypes or x.get("perturbation_type") in ptypes)
        and (not datasets or str(x.get("dataset", "")) in datasets)
        and (not args.require_counterfactual_valid or bool(x.get("counterfactual_valid")))
    ]
    if args.shuffle:
        random.Random(args.sample_seed).shuffle(examples)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    obs_map, _ = load_index_maps(args.index)
    print(f"loaded examples={len(examples)} qwen_cf={len(qwen_cf)}", flush=True)
    model = load_octo_model(args.octo_checkpoint)

    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples, 1):
        image, image_source = load_image_for_example(ex, obs_map, args.image_size)
        orig = str(ex.get("original_instruction") or "")
        cf = str(ex.get("counterfactual_instruction") or ex.get("instruction") or "")
        action_orig = predict_octo_action(model, image, orig, args.octo_seed)
        action_cf = predict_octo_action(model, image, cf, args.octo_seed)
        control_instruction = None
        control_action = None
        if args.control_mode in {"same", "both"}:
            control_instruction = orig
            control_action = predict_octo_action(model, image, control_instruction, args.octo_seed)
        if args.control_mode in {"paraphrase", "both"}:
            control_instruction = paraphrase_instruction(orig)
            control_action = predict_octo_action(model, image, control_instruction, args.octo_seed)
        rows.append({
            "example_id": ex.get("example_id"),
            "observation_id": ex.get("observation_id"),
            "perturbation_type": ex.get("perturbation_type"),
            "original_instruction": orig,
            "counterfactual_instruction": cf,
            "original_target_object": ex.get("original_target_object"),
            "counterfactual_target_object": ex.get("counterfactual_target_object"),
            "qwen_schema_counterfactual": (qwen_cf.get(str(ex.get("example_id"))) or {}).get("parsed_schema"),
            "qwen_schema_sensitive": qwen_schema_sensitive(ex, qwen_cf.get(str(ex.get("example_id"))), None),
            "octo_action_original": action_orig,
            "octo_action_counterfactual": action_cf,
            "control_instruction": control_instruction,
            "control_action": control_action,
            "image_source": image_source,
            "smoke_test": bool(ex.get("smoke_test")),
        })
        if args.log_every > 0 and (i % args.log_every == 0 or i == len(examples)):
            print(f"progress {i}/{len(examples)}", flush=True)
        if args.save_every > 0 and i % args.save_every == 0:
            write_jsonl(out_dir / "predictions.partial.jsonl", rows)

    action_std = action_std_for_rows(rows)
    write_json(out_dir / "action_std.json", {"action_std": action_std.tolist(), "source": "octo_predicted_actions"})
    for r in rows:
        r.update(action_stats_dynamic(r["octo_action_original"], r["octo_action_counterfactual"], action_std))
        if r.get("control_action") is not None:
            cs = action_stats_dynamic(r["octo_action_original"], r["control_action"], action_std)
            r["control_normalized_action_delta"] = cs["normalized_action_delta"]
    threshold = threshold_from_controls(rows, args.threshold_method, args.fixed_threshold)
    summary = summarize(rows, threshold)
    write_jsonl(out_dir / "predictions.jsonl", rows)
    write_json(out_dir / "metrics.json", summary)
    write_csv(out_dir / "summary.csv", [{"method": "octo", **{k: (v.get("mean") if isinstance(v, dict) and "mean" in v else v) for k, v in summary.items() if k != "num_examples"}, "num_examples": summary["num_examples"]}])
    write_json(out_dir / "run_metadata.json", {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "benchmark": args.benchmark,
        "qwen_predictions": args.qwen_predictions,
        "octo_checkpoint": args.octo_checkpoint,
        "max_examples": args.max_examples,
        "require_counterfactual_valid": args.require_counterfactual_valid,
        "shuffle": args.shuffle,
        "sample_seed": args.sample_seed,
        "control_mode": args.control_mode,
        "threshold_method": args.threshold_method,
        "threshold": threshold,
        "python": sys.version,
        "platform": platform.platform(),
        "note": "Octo is evaluated as native action-output sensitivity, not LIBERO task success.",
    })
    print(json.dumps({"out_dir": str(out_dir), "num_examples": len(rows), "threshold": threshold, "semantic_action_gap": summary["semantic_action_gap"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
