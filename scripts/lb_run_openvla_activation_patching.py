#!/usr/bin/env python
"""Causal activation-patching sanity test for OpenVLA target-swap examples.

For each fixed observation, cache the counterfactual prompt hidden state at a
chosen Llama layer, patch the original-instruction prefill hidden state at the
same layer/last token, then decode OpenVLA's native action tokens. This tests
whether target information readable in the representation causally changes the
action output.
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lb_run_exp2_action_sensitivity import load_index_maps, prompt as action_prompt
from lb_run_openvla_action import dtype_from_args, load_hdf5_image_pointer, load_model_and_processor
from linguistic_blindness.utils.io import command_string, read_jsonl, utc_timestamp, write_csv, write_json, write_jsonl


def parse_args():
    p = argparse.ArgumentParser(description="OpenVLA activation patching / representation-to-action perturbation test.")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--exp2-predictions", required=True)
    p.add_argument("--index", default="data/processed/unified_index.parquet")
    p.add_argument("--model-path", default="/scratch/bj2410/models/openvla-7b")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-examples", type=int, default=100)
    p.add_argument("--sample-seed", type=int, default=19)
    p.add_argument("--layer-index", type=int, default=16)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--unnorm-key", default="bridge_orig")
    p.add_argument("--threshold", type=float, default=2.2851185083448717)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--attn-implementation", default="")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def by_id(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("example_id")): r for r in rows if r.get("example_id")}


def norm(x: Any) -> str:
    return str(x or "").strip().lower()


def resolve_obs_ptr(example: Dict[str, Any], obs_map: Dict[str, str]) -> Optional[str]:
    for key in ["obs_ptr", "observation_path", "image_path", "image"]:
        if example.get(key):
            return str(example[key])
    return obs_map.get(str(example.get("observation_id") or ""))


def load_image(example: Dict[str, Any], obs_map: Dict[str, str]):
    from PIL import Image
    ptr = resolve_obs_ptr(example, obs_map)
    if ptr and "#" in ptr:
        img = load_hdf5_image_pointer(ptr)
        if img is not None:
            return img, ptr
    if ptr and Path(ptr).exists():
        return Image.open(ptr).convert("RGB"), ptr
    raise FileNotFoundError(f"no image for {example.get('example_id')} obs={example.get('observation_id')}")


def move_inputs(inputs, model, args):
    dtype = dtype_from_args(args)
    device = next(model.parameters()).device
    if dtype == "auto":
        return inputs.to(device)
    return inputs.to(device, dtype=dtype)


def ensure_action_prompt_token(inputs):
    import torch
    if not torch.all(inputs["input_ids"][:, -1] == 29871):
        tok = torch.full((inputs["input_ids"].shape[0], 1), 29871, dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], tok], dim=1)
        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            am = torch.ones((inputs["attention_mask"].shape[0], 1), dtype=inputs["attention_mask"].dtype, device=inputs["attention_mask"].device)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], am], dim=1)
    return inputs


def action_to_list(action: Any) -> List[float]:
    arr = np.asarray(action).reshape(-1)
    return [float(x) for x in arr.tolist()]


def norm_delta(a, b, std):
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    ss = np.asarray(std, dtype=float).reshape(-1)
    ss = np.where(ss[: len(aa)] < 1e-6, 1.0, ss[: len(aa)])
    return float(np.linalg.norm((aa - bb) / ss))


def predict_action(model, processor, image, instruction, args, hook=None):
    import torch
    inputs = processor(action_prompt(instruction), image)
    inputs = move_inputs(inputs, model, args)
    handle = None
    if hook is not None:
        handle = hook()
    try:
        with torch.no_grad():
            action = model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
    finally:
        if handle is not None:
            handle.remove()
    return action_to_list(action)


def cache_hidden(model, processor, image, instruction, args):
    import torch
    inputs = processor(action_prompt(instruction), image)
    inputs = move_inputs(inputs, model, args)
    inputs = ensure_action_prompt_token(inputs)
    pixel_values = inputs.get("pixel_values")
    with torch.no_grad():
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=pixel_values,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    layer = args.layer_index
    hs = out.hidden_states
    if layer < 0:
        layer = len(hs) + layer
    return hs[layer][0, -1, :].detach()


def make_patch_hook(model, layer_index: int, patch_vec, alpha: float):
    layers = model.language_model.model.layers
    layer = layer_index if layer_index >= 0 else len(layers) + layer_index
    target_layer = layers[layer]
    state = {"used": False}

    def hook_fn(module, inputs, output):
        # Patch only the prefill pass, not single-token cached generation steps.
        hidden = output[0] if isinstance(output, tuple) else output
        if state["used"] or hidden is None or hidden.ndim != 3 or hidden.shape[1] <= 1:
            return output
        pv = patch_vec.to(device=hidden.device, dtype=hidden.dtype)
        new_hidden = hidden.clone()
        new_hidden[:, -1, :] = (1.0 - alpha) * new_hidden[:, -1, :] + alpha * pv.view(1, -1)
        state["used"] = True
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    return target_layer.register_forward_hook(hook_fn)


def bootstrap_ci(vals: List[float], seed=0, n=1000):
    if not vals:
        return [None, None]
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    boots = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp2 = by_id(read_jsonl(args.exp2_predictions))
    examples = [
        x for x in read_jsonl(args.benchmark)
        if x.get("perturbation_type") == "target_swap"
        and x.get("dataset") == "libero"
        and bool(x.get("counterfactual_valid"))
        and str(x.get("example_id")) in exp2
    ]
    random.Random(args.sample_seed).shuffle(examples)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    obs_map, action_std = load_index_maps(args.index)
    model, processor = load_model_and_processor(args)
    rows = []
    for i, ex in enumerate(examples, 1):
        eid = str(ex.get("example_id"))
        e2 = exp2[eid]
        image, image_source = load_image(ex, obs_map)
        orig_i = str(ex.get("original_instruction") or "")
        cf_i = str(ex.get("counterfactual_instruction") or "")
        a_orig = action_to_list(e2.get("openvla_action_original")) if e2.get("openvla_action_original") else predict_action(model, processor, image, orig_i, args)
        a_cf = action_to_list(e2.get("openvla_action_counterfactual")) if e2.get("openvla_action_counterfactual") else predict_action(model, processor, image, cf_i, args)
        cf_vec = cache_hidden(model, processor, image, cf_i, args)
        a_patch = predict_action(
            model,
            processor,
            image,
            orig_i,
            args,
            hook=lambda cf_vec=cf_vec: make_patch_hook(model, args.layer_index, cf_vec, args.alpha),
        )
        d_orig_cf = norm_delta(a_orig, a_cf, action_std)
        d_orig_patch = norm_delta(a_orig, a_patch, action_std)
        d_patch_cf = norm_delta(a_patch, a_cf, action_std)
        rows.append({
            "example_id": eid,
            "observation_id": ex.get("observation_id"),
            "original_target": norm(ex.get("original_target_object")),
            "counterfactual_target": norm(ex.get("counterfactual_target_object")),
            "layer_index": args.layer_index,
            "alpha": args.alpha,
            "image_source": image_source,
            "native_delta_orig_cf": d_orig_cf,
            "patched_delta_from_orig": d_orig_patch,
            "patched_distance_to_cf_action": d_patch_cf,
            "native_action_sensitive": bool(d_orig_cf >= args.threshold),
            "patched_action_sensitive": bool(d_orig_patch >= args.threshold),
            "patch_moves_toward_cf_action": bool(d_patch_cf < d_orig_cf),
            "openvla_action_original": a_orig,
            "openvla_action_counterfactual": a_cf,
            "patched_action_original_to_cf_hidden": a_patch,
        })
        if args.log_every and (i % args.log_every == 0 or i == len(examples)):
            print(f"patched {i}/{len(examples)}", flush=True)
    summary = {
        "n": len(rows),
        "layer_index": args.layer_index,
        "alpha": args.alpha,
        "threshold": args.threshold,
        "native_action_sensitivity": float(np.mean([r["native_action_sensitive"] for r in rows])) if rows else None,
        "patched_action_sensitivity": float(np.mean([r["patched_action_sensitive"] for r in rows])) if rows else None,
        "patch_moves_toward_cf_action_rate": float(np.mean([r["patch_moves_toward_cf_action"] for r in rows])) if rows else None,
        "mean_native_delta_orig_cf": float(np.mean([r["native_delta_orig_cf"] for r in rows])) if rows else None,
        "mean_patched_delta_from_orig": float(np.mean([r["patched_delta_from_orig"] for r in rows])) if rows else None,
        "mean_patched_distance_to_cf_action": float(np.mean([r["patched_distance_to_cf_action"] for r in rows])) if rows else None,
        "patched_action_sensitivity_ci95": bootstrap_ci([float(r["patched_action_sensitive"]) for r in rows], args.sample_seed),
        "patch_moves_toward_cf_ci95": bootstrap_ci([float(r["patch_moves_toward_cf_action"]) for r in rows], args.sample_seed + 1),
    }
    write_jsonl(out_dir / "activation_patching_predictions.jsonl", rows)
    write_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "summary.csv", [summary])
    write_json(out_dir / "run_metadata.json", {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "note": "Counterfactual hidden state at selected OpenVLA language-model layer patches original-instruction prefill last token before native action decoding.",
    })
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
