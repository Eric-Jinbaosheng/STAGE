#!/usr/bin/env python
import argparse
import json
import math
import platform
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lb_run_exp2_action_sensitivity import load_index_maps, load_model_and_processor, prompt as action_prompt
from lb_run_openvla_action import dtype_from_args, load_hdf5_image_pointer
from linguistic_blindness.utils.io import command_string, read_jsonl, utc_timestamp, write_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen OpenVLA hidden-state linear target probe on target-swap examples.")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--exp2-predictions", required=True)
    p.add_argument("--index", default="data/processed/unified_index.parquet")
    p.add_argument("--model-path", default="/scratch/bj2410/models/openvla-7b")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--perturbation-types", default="target_swap")
    p.add_argument("--dataset-filter", default="libero")
    p.add_argument("--require-counterfactual-valid", action="store_true")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--sample-seed", type=int, default=7)
    p.add_argument("--split-seed", type=int, default=13)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--max-examples", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--attn-implementation", default="")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--ridge-lambda", type=float, default=1.0)
    p.add_argument("--random-labels", action="store_true", help="Shuffle train labels before fitting as a negative-control probe.")
    p.add_argument("--feature-pool", choices=["last_token", "mean_text"], default="last_token")
    p.add_argument("--layer-index", type=int, default=-1, help="Hidden-state layer index to probe; supports negative indexes. Default -1 is final layer.")
    p.add_argument(
        "--feature-condition",
        choices=[
            "full",
            "language_only",
            "image_only",
            "mask_target",
            "shuffled_image",
            "anchor_masked",
            "relation_masked",
        ],
        default="full",
        help="Ablation of what information reaches the frozen OpenVLA representation.",
    )
    p.add_argument("--allow-placeholder-image", action="store_true")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-features", action="store_true")
    return p.parse_args()


def by_example_id(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("example_id")): r for r in rows if r.get("example_id")}


def norm(x: Any) -> str:
    return str(x or "").strip().lower()


def resolve_obs_ptr(example: Dict[str, Any], obs_map: Dict[str, str]) -> Optional[str]:
    for key in ["obs_ptr", "observation_path", "image_path", "image"]:
        if example.get(key):
            return str(example[key])
    return obs_map.get(str(example.get("observation_id") or ""))


def load_image_for_example(example: Dict[str, Any], obs_map: Dict[str, str], allow_placeholder: bool):
    from PIL import Image
    ptr = resolve_obs_ptr(example, obs_map)
    if ptr:
        if "#" in ptr:
            img = load_hdf5_image_pointer(ptr)
            if img is not None:
                return img, ptr, False
        elif Path(ptr).exists():
            return Image.open(ptr).convert("RGB"), ptr, False
    if allow_placeholder:
        return Image.new("RGB", (224, 224), color=(127, 127, 127)), "placeholder_gray", True
    raise FileNotFoundError(f"No image found for {example.get('example_id')} observation_id={example.get('observation_id')}")


def move_to_device(inputs: Any, model: Any, args: argparse.Namespace):
    device = next(model.parameters()).device
    dtype = dtype_from_args(args)
    if hasattr(inputs, "to"):
        if dtype == "auto":
            return inputs.to(device)
        return inputs.to(device, dtype=dtype)
    return inputs


def mask_target_words(instruction: str, targets: Iterable[str]) -> str:
    out = str(instruction or "")
    for target in sorted({norm(t) for t in targets if norm(t)}, key=len, reverse=True):
        out = out.replace(target, "the object")
    return out


def mask_anchor_words(instruction: str, scene_objects: Iterable[str], keep_target: str) -> str:
    out = str(instruction or "")
    keep = norm(keep_target)
    for obj in sorted({norm(o) for o in scene_objects if norm(o) and norm(o) != keep}, key=len, reverse=True):
        out = out.replace(obj, "the anchor object")
    return out


def mask_relation_words(instruction: str) -> str:
    out = str(instruction or "")
    replacements = [
        "to the right of",
        "to the left of",
        "right of",
        "left of",
        "in front of",
        "behind",
        "between",
        "next to",
        "on top of",
        "under",
        "above",
        "below",
    ]
    for rel in sorted(replacements, key=len, reverse=True):
        out = out.replace(rel, "[RELATION]")
    return out


def extract_feature(
    model: Any,
    processor: Any,
    image: Any,
    instruction: str,
    args: argparse.Namespace,
    targets_to_mask: Optional[Iterable[str]] = None,
    scene_objects: Optional[Iterable[str]] = None,
    keep_target: str = "",
) -> np.ndarray:
    import torch

    if args.feature_condition == "image_only":
        effective_instruction = "do the task"
    elif args.feature_condition == "mask_target":
        effective_instruction = mask_target_words(instruction, targets_to_mask or [])
    elif args.feature_condition == "anchor_masked":
        effective_instruction = mask_anchor_words(instruction, scene_objects or [], keep_target)
    elif args.feature_condition == "relation_masked":
        effective_instruction = mask_relation_words(instruction)
    else:
        effective_instruction = instruction

    text = action_prompt(effective_instruction)
    if args.feature_condition == "language_only":
        tok = processor.tokenizer(text, return_tensors="pt")
        inputs = tok.to(next(model.parameters()).device)
        pixel_values = None
    else:
        inputs = processor(text, image)
        inputs = move_to_device(inputs, model, args)
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
    layer_index = args.layer_index
    num_layers = len(out.hidden_states)
    if layer_index < 0:
        layer_index = num_layers + layer_index
    if layer_index < 0 or layer_index >= num_layers:
        raise IndexError(f"layer_index {args.layer_index} resolved to {layer_index}, but model returned {num_layers} hidden-state tensors")
    h = out.hidden_states[layer_index][0].detach().float().cpu().numpy()
    if args.feature_pool == "mean_text":
        # The multimodal sequence inserts visual tokens after BOS. Mean pooling is a coarse robustness check.
        feat = h.mean(axis=0)
    else:
        feat = h[-1]
    return feat.astype(np.float32)


def make_instances(examples: List[Dict[str, Any]], exp2: Dict[str, Dict[str, Any]], model: Any, processor: Any, obs_map: Dict[str, str], args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = []
    shuffled_images: Dict[str, Tuple[Any, str, bool]] = {}
    if args.feature_condition == "shuffled_image":
        image_keys = [str(x.get("example_id")) for x in examples]
        shuffled_keys = image_keys[1:] + image_keys[:1]
        image_by_id = {}
        for ex in examples:
            image_by_id[str(ex.get("example_id"))] = load_image_for_example(ex, obs_map, args.allow_placeholder_image)
        for src, dst in zip(image_keys, shuffled_keys):
            shuffled_images[src] = image_by_id[dst]
    for i, ex in enumerate(examples, start=1):
        eid = str(ex.get("example_id"))
        e2 = exp2.get(eid) or {}
        if args.feature_condition == "shuffled_image":
            image, image_source, placeholder = shuffled_images[eid]
            image_source = f"shuffled_from_other_example::{image_source}"
        else:
            image, image_source, placeholder = load_image_for_example(ex, obs_map, args.allow_placeholder_image)
        for side, instr_key, target_key in [
            ("original", "original_instruction", "original_target_object"),
            ("counterfactual", "counterfactual_instruction", "counterfactual_target_object"),
        ]:
            instr = str(ex.get(instr_key) or ex.get("instruction") or "")
            target = norm(ex.get(target_key))
            feat = extract_feature(
                model,
                processor,
                image,
                instr,
                args,
                targets_to_mask=[ex.get("original_target_object"), ex.get("counterfactual_target_object")],
                scene_objects=ex.get("scene_objects") or [],
                keep_target=target,
            )
            rows.append({
                "instance_id": f"{eid}::{side}",
                "example_id": eid,
                "side": side,
                "observation_id": ex.get("observation_id"),
                "instruction": instr,
                "feature_condition": args.feature_condition,
            "layer_index": args.layer_index,
                "target_object": target,
                "target_pair": f"{norm(ex.get('original_target_object'))} -> {norm(ex.get('counterfactual_target_object'))}",
                "image_source": image_source,
                "placeholder_image": placeholder,
                "openvla_action_sensitive": e2.get("openvla_action_sensitive"),
                "normalized_action_delta": e2.get("normalized_action_delta"),
                "feature": feat.tolist() if args.save_features else None,
                "_feature_np": feat,
                "feature_index": len(rows),
            })
        if args.log_every > 0 and (i % args.log_every == 0 or i == len(examples)):
            print(f"features {i}/{len(examples)} examples ({len(rows)} instances)", flush=True)
    return rows


def split_by_example(examples: List[Dict[str, Any]], train_frac: float, seed: int) -> Tuple[set, set]:
    ids = [str(x.get("example_id")) for x in examples]
    rng = random.Random(seed)
    for _ in range(100):
        shuffled = ids[:]
        rng.shuffle(shuffled)
        n_train = max(1, int(round(len(shuffled) * train_frac)))
        train = set(shuffled[:n_train])
        test = set(shuffled[n_train:])
        if test:
            return train, test
    return set(ids), set()


def ridge_fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, lam: float) -> np.ndarray:
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    Xtr = (X_train - mu) / sigma
    Xte = (X_test - mu) / sigma
    Xtr = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1), dtype=Xtr.dtype)], axis=1)
    Xte = np.concatenate([Xte, np.ones((Xte.shape[0], 1), dtype=Xte.dtype)], axis=1)
    classes = sorted(set(y_train.tolist()))
    cls_to_i = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(y_train), len(classes)), dtype=np.float32)
    for i, y in enumerate(y_train):
        Y[i, cls_to_i[y]] = 1.0
    reg = lam * np.eye(Xtr.shape[1], dtype=np.float32)
    reg[-1, -1] = 0.0
    W = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ Y)
    scores = Xte @ W
    pred_idx = scores.argmax(axis=1)
    return np.asarray([classes[i] for i in pred_idx])


def bootstrap_ci(vals: List[float], seed: int = 0, n_boot: int = 1000) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    boots = []
    for _ in range(n_boot):
        sample = arr[rng.integers(0, len(arr), len(arr))]
        boots.append(float(sample.mean()))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def evaluate(rows: List[Dict[str, Any]], examples: List[Dict[str, Any]], train_ids: set, test_ids: set, args: argparse.Namespace) -> Dict[str, Any]:
    X = []
    y = []
    idx_rows = []
    # Rehydrate features from in-memory hidden vectors stored separately in _feature_np.
    for r in rows:
        X.append(r["_feature_np"])
        y.append(r["target_object"])
        idx_rows.append(r)
    X = np.stack(X, axis=0)
    y = np.asarray(y)
    train_mask = np.asarray([r["example_id"] in train_ids for r in idx_rows])
    test_mask = np.asarray([r["example_id"] in test_ids for r in idx_rows])
    y_train_for_fit = y[train_mask].copy()
    if args.random_labels:
        rng = np.random.default_rng(args.split_seed + 1701)
        y_train_for_fit = rng.permutation(y_train_for_fit)
    y_pred = np.empty_like(y)
    y_pred[:] = ""
    y_pred[test_mask] = ridge_fit_predict(X[train_mask], y_train_for_fit, X[test_mask], args.ridge_lambda)
    majority = Counter(y[train_mask].tolist()).most_common(1)[0][0]
    for r, pred in zip(idx_rows, y_pred):
        if r["example_id"] in test_ids:
            r["probe_pred_target"] = str(pred)
            r["probe_target_correct"] = bool(str(pred) == r["target_object"])
        else:
            r["probe_pred_target"] = None
            r["probe_target_correct"] = None
    test_rows = [r for r in idx_rows if r["example_id"] in test_ids]
    acc_vals = [1.0 if r["probe_target_correct"] else 0.0 for r in test_rows]
    majority_vals = [1.0 if majority == r["target_object"] else 0.0 for r in test_rows]
    pair_rows = []
    pair_sens_vals = []
    action_vals = []
    by_id = defaultdict(dict)
    for r in test_rows:
        by_id[r["example_id"]][r["side"]] = r
    for ex in examples:
        eid = str(ex.get("example_id"))
        if eid not in test_ids:
            continue
        orig = by_id[eid].get("original")
        cf = by_id[eid].get("counterfactual")
        if not orig or not cf:
            continue
        sens = bool(orig.get("probe_pred_target") == orig.get("target_object") and cf.get("probe_pred_target") == cf.get("target_object"))
        action_sens = orig.get("openvla_action_sensitive")
        pair_sens_vals.append(1.0 if sens else 0.0)
        if action_sens is not None:
            action_vals.append(1.0 if action_sens else 0.0)
        pair_rows.append({
            "example_id": eid,
            "target_pair": orig.get("target_pair"),
            "original_target": orig.get("target_object"),
            "counterfactual_target": cf.get("target_object"),
            "probe_pred_original": orig.get("probe_pred_target"),
            "probe_pred_counterfactual": cf.get("probe_pred_target"),
            "hidden_probe_schema_sensitive": sens,
            "openvla_action_sensitive": action_sens,
            "semantic_action_gap_case": bool(sens and action_sens is False),
            "normalized_action_delta": orig.get("normalized_action_delta"),
        })
    probe_sens = float(np.mean(pair_sens_vals)) if pair_sens_vals else None
    action_sens = float(np.mean(action_vals)) if action_vals else None
    gap = None if probe_sens is None or action_sens is None else probe_sens - action_sens
    by_pair = []
    for pair in sorted({r["target_pair"] for r in pair_rows}):
        prs = [r for r in pair_rows if r["target_pair"] == pair]
        ps = np.mean([r["hidden_probe_schema_sensitive"] for r in prs]) if prs else None
        av = [r["openvla_action_sensitive"] for r in prs if r["openvla_action_sensitive"] is not None]
        aa = np.mean(av) if av else None
        by_pair.append({
            "target_pair": pair,
            "n": len(prs),
            "hidden_probe_sensitivity": float(ps) if ps is not None else None,
            "openvla_action_sensitivity": float(aa) if aa is not None else None,
            "hidden_probe_action_gap": None if ps is None or aa is None else float(ps - aa),
            "mean_normalized_action_delta": float(np.mean([r["normalized_action_delta"] for r in prs if r.get("normalized_action_delta") is not None])) if any(r.get("normalized_action_delta") is not None for r in prs) else None,
        })
    ci_acc = bootstrap_ci(acc_vals, args.split_seed)
    ci_sens = bootstrap_ci(pair_sens_vals, args.split_seed + 1)
    return {
        "instance_rows": idx_rows,
        "pair_rows": pair_rows,
        "target_pair_breakdown": by_pair,
        "summary": {
            "num_examples": len(examples),
            "num_instances": len(rows),
            "num_train_examples": len(train_ids),
            "num_test_examples": len(test_ids),
            "num_test_instances": len(test_rows),
            "classes": sorted(set(y.tolist())),
            "train_label_counts": dict(Counter(y[train_mask].tolist())),
            "test_label_counts": dict(Counter(y[test_mask].tolist())),
            "hidden_probe_target_accuracy": {"mean": float(np.mean(acc_vals)) if acc_vals else None, "count": len(acc_vals), "ci95": ci_acc},
            "majority_baseline_accuracy": {"mean": float(np.mean(majority_vals)) if majority_vals else None, "count": len(majority_vals)},
            "hidden_probe_schema_sensitivity": {"mean": probe_sens, "count": len(pair_sens_vals), "ci95": ci_sens},
            "openvla_action_sensitivity": {"mean": action_sens, "count": len(action_vals)},
            "hidden_probe_action_gap": {"mean": gap, "count": min(len(pair_sens_vals), len(action_vals))},
            "feature_pool": args.feature_pool,
            "feature_condition": args.feature_condition,
            "layer_index": args.layer_index,
            "ridge_lambda": args.ridge_lambda,
            "random_labels": bool(args.random_labels),
        },
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ptypes = {x.strip() for x in args.perturbation_types.split(",") if x.strip()}
    datasets = {x.strip() for x in args.dataset_filter.split(",") if x.strip()}
    exp2 = by_example_id(read_jsonl(args.exp2_predictions))
    examples = [
        x for x in read_jsonl(args.benchmark)
        if (not ptypes or x.get("perturbation_type") in ptypes)
        and (not datasets or str(x.get("dataset", "")) in datasets)
        and (not args.require_counterfactual_valid or bool(x.get("counterfactual_valid")))
        and str(x.get("example_id")) in exp2
    ]
    if args.shuffle:
        random.Random(args.sample_seed).shuffle(examples)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    obs_map, _ = load_index_maps(args.index)
    print(f"loaded examples={len(examples)} exp2={len(exp2)}", flush=True)
    model, processor = load_model_and_processor(args)
    rows = make_instances(examples, exp2, model, processor, obs_map, args)
    train_ids, test_ids = split_by_example(examples, args.train_frac, args.split_seed)
    result = evaluate(rows, examples, train_ids, test_ids, args)
    json_instance_rows = []
    for r in result["instance_rows"]:
        rr = {k: v for k, v in r.items() if k != "_feature_np"}
        if not args.save_features:
            rr.pop("feature", None)
        json_instance_rows.append(rr)
    write_jsonl(out_dir / "hidden_state_instances.jsonl", json_instance_rows)
    write_jsonl(out_dir / "hidden_state_pair_predictions.jsonl", result["pair_rows"])
    write_json(out_dir / "metrics.json", result["summary"])
    write_csv(out_dir / "summary.csv", [{
        "num_examples": result["summary"]["num_examples"],
        "num_train_examples": result["summary"]["num_train_examples"],
        "num_test_examples": result["summary"]["num_test_examples"],
        "hidden_probe_target_accuracy": result["summary"]["hidden_probe_target_accuracy"]["mean"],
        "majority_baseline_accuracy": result["summary"]["majority_baseline_accuracy"]["mean"],
        "hidden_probe_schema_sensitivity": result["summary"]["hidden_probe_schema_sensitivity"]["mean"],
        "openvla_action_sensitivity": result["summary"]["openvla_action_sensitivity"]["mean"],
        "hidden_probe_action_gap": result["summary"]["hidden_probe_action_gap"]["mean"],
        "feature_condition": result["summary"]["feature_condition"],
        "layer_index": result["summary"]["layer_index"],
        "random_labels": result["summary"]["random_labels"],
    }])
    write_csv(out_dir / "target_pair_breakdown.csv", result["target_pair_breakdown"])
    write_json(out_dir / "qualitative_cases.json", {
        "hidden_readable_action_insensitive": [r for r in result["pair_rows"] if r.get("hidden_probe_schema_sensitive") and r.get("openvla_action_sensitive") is False][:30]
    })
    write_json(out_dir / "run_metadata.json", {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "benchmark": args.benchmark,
        "exp2_predictions": args.exp2_predictions,
        "model_path": args.model_path,
        "max_examples": args.max_examples,
        "train_frac": args.train_frac,
        "sample_seed": args.sample_seed,
        "split_seed": args.split_seed,
        "feature_pool": args.feature_pool,
        "feature_condition": args.feature_condition,
        "ridge_lambda": args.ridge_lambda,
        "random_labels": bool(args.random_labels),
        "python": sys.version,
        "platform": platform.platform(),
        "note": "Frozen OpenVLA action-prompt hidden states with a lightweight ridge linear target probe. This tests readout, not text generation.",
    })
    print(json.dumps({
        "out_dir": str(out_dir),
        "summary": result["summary"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
