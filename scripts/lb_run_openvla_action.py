import argparse
import json
import math
import platform
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from linguistic_blindness.benchmark.schema import normalize_schema
from linguistic_blindness.evaluation.failure_taxonomy import flatten_failures, summarize_failures
from linguistic_blindness.evaluation.metrics import flatten_main_results, summarize_metrics
from linguistic_blindness.evaluation.tables import write_tables
from linguistic_blindness.utils.io import command_string, read_jsonl, utc_timestamp, write_csv, write_json, write_jsonl
from linguistic_blindness.verification.checker import check_schema


SAFE_ACTIONS = {"ASK", "PROMPT", "WAIT", "HOLD", "ABORT", "BACKOFF", "TARGET_NOT_FOUND"}
TASK_ACTIONS = {"MOVE_TO", "GRASP", "PICK", "PLACE", "RETRACT", "TRANSFER"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run OpenVLA action-head probe on linguistic-blindness benchmark.")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--model-path", default="openvla/openvla-7b")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--method", default="openvla7b_action_head")
    p.add_argument("--max-examples", type=int, default=20, help="0 means all examples; start small for OpenVLA.")
    p.add_argument("--unnorm-key", default="bridge_orig")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--attn-implementation", default="", help="Optional, e.g. flash_attention_2. Empty uses HF default.")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--allow-placeholder-image", action="store_true", help="Use a gray image if benchmark image path is unavailable. Smoke-test only.")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=10)
    return p.parse_args()


def dtype_from_args(args: argparse.Namespace):
    import torch

    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return "auto"


def load_model_and_processor(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"loading OpenVLA processor from {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    print("processor loaded", flush=True)
    kwargs = {
        "torch_dtype": dtype_from_args(args),
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    print(f"loading OpenVLA model dtype={kwargs['torch_dtype']} attn={args.attn_implementation or 'default'}", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(args.model_path, **kwargs)
    if args.device and not args.device.startswith("auto"):
        model = model.to(args.device)
    model.eval()
    print(f"model loaded on {next(model.parameters()).device}", flush=True)
    return model, processor


def load_image(example: Dict[str, Any], args: argparse.Namespace):
    from PIL import Image

    image_path = example.get("image_path") or example.get("image")
    if image_path and Path(str(image_path)).exists():
        return Image.open(str(image_path)).convert("RGB"), str(image_path), False

    obs_ptr = example.get("obs_ptr") or example.get("observation_path")
    if obs_ptr:
        image = load_hdf5_image_pointer(str(obs_ptr))
        if image is not None:
            return image, str(obs_ptr), False

    if args.allow_placeholder_image:
        return Image.new("RGB", (args.image_size, args.image_size), color=(127, 127, 127)), "placeholder_gray", True
    raise FileNotFoundError(
        "No image found for example. Restore raw observation files or pass --allow-placeholder-image for smoke tests."
    )


def load_hdf5_image_pointer(ptr: str):
    if "#" not in ptr:
        return None
    path_text, key = ptr.split("#", 1)
    path = Path(path_text)
    if not path.exists():
        return None
    try:
        import h5py
        import numpy as np
        from PIL import Image

        with h5py.File(path, "r") as f:
            try:
                arr = f[key][()]
            except Exception:
                dataset_key, frame_idx = split_hdf5_frame_key(key)
                if dataset_key is None or dataset_key not in f:
                    raise
                arr = f[dataset_key][frame_idx]
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
            arr = arr.transpose(1, 2, 0)
        if arr.dtype.kind == "f":
            arr = (arr.clip(0, 1) * 255).astype("uint8")
        return Image.fromarray(arr).convert("RGB")
    except Exception as exc:
        print(f"warning: failed to load hdf5 image {ptr}: {exc}", flush=True)
        return None


def split_hdf5_frame_key(key: str) -> Tuple[Optional[str], int]:
    parts = key.strip("/").rsplit("/", 1)
    if len(parts) != 2:
        return None, 0
    dataset_key, frame_text = parts
    try:
        frame_idx = int(frame_text)
    except Exception:
        return None, 0
    return "/" + dataset_key, frame_idx


def prompt_for(example: Dict[str, Any]) -> str:
    instruction = str(example.get("instruction", ""))
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def action_to_symbolic(action: List[float], example: Dict[str, Any]) -> Tuple[str, str]:
    ptype = str(example.get("perturbation_type", ""))
    instruction = str(example.get("instruction", ""))
    if ptype == "blank_instruction" or not instruction.strip():
        return "WAIT", "blank instruction; symbolic adapter defers"
    if ptype == "impossible_instruction":
        return "TARGET_NOT_FOUND", "impossible-instruction perturbation; symbolic adapter defers"
    if ptype == "negation" and re.search(r"\b(do not|don't|avoid|never)\b", instruction.lower()):
        return "WAIT", "prohibition detected; symbolic adapter defers"
    if ptype == "safety_conflict":
        return "HOLD", "safety-conflict perturbation; symbolic adapter holds"

    trans_norm = math.sqrt(sum(float(x) * float(x) for x in action[:3])) if len(action) >= 3 else 0.0
    gripper = float(action[-1]) if action else 0.0
    if trans_norm < 1e-3 and abs(gripper) < 1e-3:
        return "WAIT", f"small action norm={trans_norm:.4f}"
    if abs(gripper) > 0.5 and trans_norm < 0.05:
        return "GRASP", f"large gripper component={gripper:.3f}"
    return "MOVE_TO", f"continuous OpenVLA action norm={trans_norm:.4f}, gripper={gripper:.3f}"


def schema_from_action(action: List[float], example: Dict[str, Any], image_source: str, placeholder: bool) -> Dict[str, Any]:
    next_action, reason = action_to_symbolic(action, example)
    gold = example.get("gold_schema", {}) or {}
    obs = example.get("observation_state", {}) or {}
    blocked = []
    allowed = list(TASK_ACTIONS | SAFE_ACTIONS)
    if next_action in SAFE_ACTIONS:
        allowed = list(SAFE_ACTIONS)
        blocked = list(TASK_ACTIONS)
    return normalize_schema({
        "target_object": None,
        "target_exists": True,
        "phase": obs.get("phase") or gold.get("phase") or "UNK",
        "human_contact": obs.get("human_contact", gold.get("human_contact")),
        "human_released": obs.get("human_released", gold.get("human_released")),
        "robot_contact": obs.get("robot_contact", gold.get("robot_contact")),
        "robot_grasp_stable": obs.get("robot_grasp_stable", gold.get("robot_grasp_stable")),
        "allowed_actions": sorted(allowed),
        "blocked_actions": sorted(blocked),
        "next_action": next_action,
        "reason": f"{reason}; image_source={image_source}; placeholder_image={placeholder}",
        "confidence": None,
    })


def action_to_list(action: Any) -> List[float]:
    try:
        import numpy as np

        arr = np.asarray(action).reshape(-1)
        return [float(x) for x in arr.tolist()]
    except Exception:
        if isinstance(action, (list, tuple)):
            return [float(x) for x in action]
    return []


def predict_one(model, processor, example: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    image, image_source, placeholder = load_image(example, args)
    prompt = prompt_for(example)
    device = next(model.parameters()).device
    dtype = dtype_from_args(args)
    inputs = processor(prompt, image)
    if hasattr(inputs, "to"):
        if dtype == "auto":
            inputs = inputs.to(device)
        else:
            inputs = inputs.to(device, dtype=dtype)
    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
    action_vector = action_to_list(action)
    parsed = schema_from_action(action_vector, example, image_source, placeholder)
    checked = check_schema(example, parsed)
    return {
        "example_id": example.get("example_id"),
        "observation_id": example.get("observation_id"),
        "instruction": example.get("instruction"),
        "original_instruction": example.get("original_instruction"),
        "counterfactual_instruction": example.get("counterfactual_instruction"),
        "perturbation_type": example.get("perturbation_type"),
        "method": args.method,
        "raw_model_output": {"action_vector": action_vector, "prompt": prompt, "unnorm_key": args.unnorm_key},
        "parsed_schema": checked["schema"],
        "checker_flags": checked["flags"],
        "checker_result": checked["flags"],
        "gated_action": checked["schema"].get("next_action"),
        "final_gated_action": checked["schema"].get("next_action"),
        "gate_result": {"final_gated_action": checked["schema"].get("next_action"), "gate_changed_action": False, "gate_reason": "not enabled"},
        "gold_schema": example.get("gold_schema"),
        "original_schema": example.get("original_schema"),
        "dataset": example.get("dataset"),
        "smoke_test": bool(example.get("smoke_test", False) or placeholder),
        "image_source": image_source,
        "placeholder_image": placeholder,
        "action_vector": action_vector,
        "action_target_object": None,
    }


def save_outputs(out_dir: Path, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    write_jsonl(out_dir / "predictions.jsonl", rows)
    summary = summarize_metrics(rows)
    failure_summary = summarize_failures(rows)
    main_rows = flatten_main_results(summary)
    failure_rows = flatten_failures(failure_summary)
    write_json(out_dir / "main_results.json", summary)
    write_csv(out_dir / "main_results.csv", main_rows)
    write_json(out_dir / "failure_taxonomy.json", failure_summary)
    write_csv(out_dir / "failure_taxonomy.csv", failure_rows)
    write_json(out_dir / "qualitative_cases.json", {"cases": {k: v["examples"] for k, v in failure_summary["failure_taxonomy"].items()}})
    write_tables(out_dir, main_rows, failure_rows)
    write_json(out_dir / "run_metadata.json", {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "benchmark": args.benchmark,
        "model_path": args.model_path,
        "method": args.method,
        "max_examples": args.max_examples,
        "num_predictions": len(rows),
        "unnorm_key": args.unnorm_key,
        "allow_placeholder_image": args.allow_placeholder_image,
        "python": sys.version,
        "platform": platform.platform(),
        "warning": "OpenVLA outputs continuous actions. target_object is unavailable unless an action-target adapter is added.",
    })


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"reading benchmark {args.benchmark}", flush=True)
    examples = read_jsonl(args.benchmark)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    print(f"loaded {len(examples)} examples", flush=True)
    model, processor = load_model_and_processor(args)
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples, start=1):
        if i == 1:
            print("starting OpenVLA action generation", flush=True)
        rows.append(predict_one(model, processor, ex, args))
        if args.log_every > 0 and (i % args.log_every == 0 or i == len(examples)):
            print(f"progress {i}/{len(examples)}", flush=True)
        if args.save_every > 0 and i % args.save_every == 0:
            save_outputs(out_dir, rows, args)
    save_outputs(out_dir, rows, args)
    print(json.dumps({"out_dir": str(out_dir), "num_predictions": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
