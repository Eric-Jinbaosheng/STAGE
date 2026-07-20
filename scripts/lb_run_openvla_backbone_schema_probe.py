#!/usr/bin/env python
import argparse
import json
import math
import platform
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lb_run_exp2_action_sensitivity import load_index_maps, load_model_and_processor, qwen_schema_sensitive
from lb_run_openvla_action import dtype_from_args, load_hdf5_image_pointer
from linguistic_blindness.utils.io import command_string, read_jsonl, utc_timestamp, write_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe OpenVLA VLM/backbone target semantics on counterfactual target swaps.")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--exp2-predictions", default="", help="Existing Exp2 action predictions, used to import action sensitivity.")
    p.add_argument("--qwen-predictions", default="", help="Optional Qwen schema predictions for comparison.")
    p.add_argument("--index", default="data/processed/unified_index.parquet")
    p.add_argument("--model-path", default="/scratch/bj2410/models/openvla-7b")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--probe-mode", choices=["forced_choice", "direct_generate", "both"], default="forced_choice")
    p.add_argument("--choice-scoring", choices=["label", "object"], default="label",
                   help="label scores A/B/C answers to reduce object-token and candidate-length priors; object scores raw object names.")
    p.add_argument("--perturbation-types", default="target_swap")
    p.add_argument("--dataset-filter", default="libero")
    p.add_argument("--require-counterfactual-valid", action="store_true")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--sample-seed", type=int, default=7)
    p.add_argument("--max-examples", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--attn-implementation", default="")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--allow-placeholder-image", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=25)
    return p.parse_args()


def by_example_id(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("example_id")): r for r in rows if r.get("example_id")}


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


def normalize_text(x: Any) -> str:
    return str(x or "").strip().lower()


def unique_keep_order(vals: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in vals:
        s = normalize_text(v)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def target_candidates(example: Dict[str, Any]) -> List[str]:
    scene = example.get("scene_objects") or []
    if isinstance(scene, str):
        scene = [x.strip() for x in re.split(r"[,;|]", scene) if x.strip()]
    return unique_keep_order([
        example.get("original_target_object"),
        example.get("counterfactual_target_object"),
        example.get("target_object"),
        *scene,
    ])


def target_prompt(instruction: str, candidates: List[str]) -> str:
    choices = ", ".join(candidates)
    return (
        "In: You are analyzing a robot manipulation scene. "
        f"Instruction: {instruction.strip()}\n"
        f"Scene objects: {choices}\n"
        "Question: Which single scene object is the target object named by the instruction? "
        "Answer with exactly one object name from the list.\nOut:"
    )


def label_target_prompt(instruction: str, candidates: List[str]) -> Tuple[str, Dict[str, str]]:
    labels = [chr(ord("A") + i) for i in range(len(candidates))]
    mapping = dict(zip(labels, candidates))
    choices = "\n".join(f"{label}. {target}" for label, target in mapping.items())
    prompt = (
        "In: You are analyzing a robot manipulation instruction.\n"
        f"Instruction: {instruction.strip()}\n"
        "Choose the option that names the object the robot is instructed to manipulate.\n"
        f"Options:\n{choices}\n"
        "Answer with only the option letter.\nOut:"
    )
    return prompt, mapping


def schema_prompt(instruction: str, candidates: List[str]) -> str:
    choices = ", ".join(candidates)
    return (
        "In: You are analyzing a robot manipulation scene. "
        f"Instruction: {instruction.strip()}\n"
        f"Scene objects: {choices}\n"
        "Return compact JSON with keys target_object, target_exists, next_action. "
        "target_object must be one object from Scene objects if the target exists.\nOut:"
    )


def move_to_device(inputs: Any, model: Any, args: argparse.Namespace):
    import torch

    device = next(model.parameters()).device
    dtype = dtype_from_args(args)
    if hasattr(inputs, "to"):
        if dtype == "auto":
            return inputs.to(device)
        return inputs.to(device, dtype=dtype)
    return inputs


def score_completion(model: Any, processor: Any, image: Any, base_prompt: str, completion: str, args: argparse.Namespace) -> float:
    import torch

    tok = processor.tokenizer
    full_text = f"{base_prompt} {completion}"
    inputs = processor(full_text, image)
    inputs = move_to_device(inputs, model, args)
    labels = inputs["input_ids"].clone()
    prompt_len = tok(base_prompt, return_tensors="pt").input_ids.shape[1]
    labels[:, :prompt_len] = -100
    with torch.no_grad():
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
    # HF loss is mean NLL over non-ignored target tokens; higher is better after negation.
    loss = float(out.loss.detach().float().cpu().item())
    token_count = int((labels != -100).sum().detach().cpu().item())
    return -loss if token_count > 0 and math.isfinite(loss) else float("-inf")


def forced_choice_scores(model: Any, processor: Any, image: Any, instruction: str, candidates: List[str], args: argparse.Namespace) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if args.choice_scoring == "label":
        base_prompt, label_to_target = label_target_prompt(instruction, candidates)
        for label, target in label_to_target.items():
            scores[target] = score_completion(model, processor, image, base_prompt, label, args)
    else:
        base_prompt = target_prompt(instruction, candidates)
        for cand in candidates:
            scores[cand] = score_completion(model, processor, image, base_prompt, cand, args)
    return scores


def direct_generate(model: Any, processor: Any, image: Any, instruction: str, candidates: List[str], args: argparse.Namespace) -> str:
    import torch

    prompt = schema_prompt(instruction, candidates)
    inputs = processor(prompt, image)
    inputs = move_to_device(inputs, model, args)
    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    gen_ids = ids[0, inputs["input_ids"].shape[1] :]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()


def parse_generated_target(text: str, candidates: List[str]) -> Optional[str]:
    raw = normalize_text(text)
    if not raw:
        return None
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip().rstrip("`").strip()
        m = re.search(r"\{.*\}", cleaned, flags=re.S)
        obj = json.loads(m.group(0) if m else cleaned)
        tgt = normalize_text(obj.get("target_object"))
        if tgt:
            return tgt
    except Exception:
        pass
    for cand in sorted(candidates, key=len, reverse=True):
        if normalize_text(cand) in raw:
            return normalize_text(cand)
    return None


def pred_from_scores(scores: Dict[str, float]) -> Optional[str]:
    if not scores:
        return None
    return max(scores.items(), key=lambda kv: kv[1])[0]


def sensitivity(orig_pred: Optional[str], cf_pred: Optional[str], orig_gold: str, cf_gold: str) -> bool:
    return normalize_text(orig_pred) == normalize_text(orig_gold) and normalize_text(cf_pred) == normalize_text(cf_gold) and normalize_text(orig_gold) != normalize_text(cf_gold)


def aggregate_bool(vals: List[Optional[bool]]) -> Dict[str, Any]:
    clean = [v for v in vals if v is not None]
    return {"count": len(clean), "mean": (sum(bool(v) for v in clean) / len(clean) if clean else None)}


def aggregate_float(vals: List[Any]) -> Dict[str, Any]:
    clean = []
    for v in vals:
        try:
            fv = float(v)
            if math.isfinite(fv):
                clean.append(fv)
        except Exception:
            pass
    return {"count": len(clean), "mean": (sum(clean) / len(clean) if clean else None)}


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    schema_key = "openvla_backbone_schema_sensitive"
    action_vals = [r.get("openvla_action_sensitive") for r in rows if r.get("openvla_action_sensitive") is not None]
    schema = aggregate_bool([r.get(schema_key) for r in rows])
    action = aggregate_bool(action_vals)
    gap = None if schema["mean"] is None or action["mean"] is None else schema["mean"] - action["mean"]
    out = {
        "num_examples": len(rows),
        "probe_mode": rows[0].get("probe_mode") if rows else None,
        "openvla_backbone_schema_sensitivity": schema,
        "openvla_action_sensitivity": action,
        "openvla_backbone_action_gap": {"mean": gap, "count": min(schema["count"], action["count"])},
        "qwen_schema_sensitivity": aggregate_bool([r.get("qwen_schema_sensitive") for r in rows]),
        "target_correct_original": aggregate_bool([r.get("openvla_backbone_original_target_correct") for r in rows]),
        "target_correct_counterfactual": aggregate_bool([r.get("openvla_backbone_counterfactual_target_correct") for r in rows]),
        "mean_normalized_action_delta": aggregate_float([r.get("normalized_action_delta") for r in rows]),
    }
    by_pair = []
    pairs = sorted({r.get("target_pair") for r in rows})
    for pair in pairs:
        prows = [r for r in rows if r.get("target_pair") == pair]
        ps = aggregate_bool([r.get(schema_key) for r in prows])
        pa = aggregate_bool([r.get("openvla_action_sensitive") for r in prows])
        by_pair.append({
            "target_pair": pair,
            "n": len(prows),
            "openvla_backbone_schema_sensitivity": ps["mean"],
            "openvla_action_sensitivity": pa["mean"],
            "openvla_backbone_action_gap": None if ps["mean"] is None or pa["mean"] is None else ps["mean"] - pa["mean"],
            "target_correct_original": aggregate_bool([r.get("openvla_backbone_original_target_correct") for r in prows])["mean"],
            "target_correct_counterfactual": aggregate_bool([r.get("openvla_backbone_counterfactual_target_correct") for r in prows])["mean"],
            "mean_normalized_action_delta": aggregate_float([r.get("normalized_action_delta") for r in prows])["mean"],
        })
    out["target_pair_breakdown"] = by_pair
    return out


def latex_table(summary: Dict[str, Any]) -> str:
    rows = [
        ("Qwen2.5-VL-7B schema", summary.get("qwen_schema_sensitivity", {}).get("mean"), None, None),
        ("OpenVLA backbone probe", summary.get("openvla_backbone_schema_sensitivity", {}).get("mean"), None, None),
        ("OpenVLA action head", None, summary.get("openvla_action_sensitivity", {}).get("mean"), None),
        ("Backbone vs action", summary.get("openvla_backbone_schema_sensitivity", {}).get("mean"), summary.get("openvla_action_sensitivity", {}).get("mean"), summary.get("openvla_backbone_action_gap", {}).get("mean")),
    ]
    def fmt(x):
        return "--" if x is None else f"{float(x):.3f}"
    lines = ["\\begin{tabular}{lrrr}", "Probe / Output & Schema Sens. & Action Sens. & Gap \\\\", "\\hline"]
    for name, s, a, g in rows:
        lines.append(f"{name} & {fmt(s)} & {fmt(a)} & {fmt(g)} \\\\")
    lines.append("\\end{tabular}\n")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ptypes = {x.strip() for x in args.perturbation_types.split(",") if x.strip()}
    datasets = {x.strip() for x in args.dataset_filter.split(",") if x.strip()}
    exp2 = by_example_id(read_jsonl(args.exp2_predictions)) if args.exp2_predictions else {}
    qwen = by_example_id(read_jsonl(args.qwen_predictions)) if args.qwen_predictions else {}
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
    print(f"loaded examples={len(examples)} exp2={len(exp2)} qwen={len(qwen)}", flush=True)
    model, processor = load_model_and_processor(args)
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples, start=1):
        eid = str(ex.get("example_id"))
        image, image_source, placeholder = load_image_for_example(ex, obs_map, args.allow_placeholder_image)
        candidates = target_candidates(ex)
        if len(candidates) < 2:
            print(f"warning: fewer than two candidates for {eid}: {candidates}", flush=True)
        orig_instr = str(ex.get("original_instruction") or "")
        cf_instr = str(ex.get("counterfactual_instruction") or ex.get("instruction") or "")
        mode = args.probe_mode
        orig_scores = cf_scores = None
        orig_raw = cf_raw = None
        if mode in {"forced_choice", "both"}:
            orig_scores = forced_choice_scores(model, processor, image, orig_instr, candidates, args)
            cf_scores = forced_choice_scores(model, processor, image, cf_instr, candidates, args)
            orig_pred = pred_from_scores(orig_scores)
            cf_pred = pred_from_scores(cf_scores)
        else:
            orig_pred = cf_pred = None
        if mode in {"direct_generate", "both"}:
            orig_raw = direct_generate(model, processor, image, orig_instr, candidates, args)
            cf_raw = direct_generate(model, processor, image, cf_instr, candidates, args)
            gen_orig_pred = parse_generated_target(orig_raw, candidates)
            gen_cf_pred = parse_generated_target(cf_raw, candidates)
            if mode == "direct_generate":
                orig_pred, cf_pred = gen_orig_pred, gen_cf_pred
        orig_gold = normalize_text(ex.get("original_target_object"))
        cf_gold = normalize_text(ex.get("counterfactual_target_object"))
        e2 = exp2.get(eid) or {}
        q = qwen.get(eid) or {}
        row = {
            "example_id": eid,
            "observation_id": ex.get("observation_id"),
            "perturbation_type": ex.get("perturbation_type"),
            "probe_mode": mode,
            "image_source": image_source,
            "placeholder_image": placeholder,
            "original_instruction": orig_instr,
            "counterfactual_instruction": cf_instr,
            "original_target_object": orig_gold,
            "counterfactual_target_object": cf_gold,
            "target_pair": f"{orig_gold} -> {cf_gold}",
            "scene_objects": candidates,
            "openvla_backbone_target_original": orig_pred,
            "openvla_backbone_target_counterfactual": cf_pred,
            "openvla_backbone_original_target_correct": normalize_text(orig_pred) == orig_gold,
            "openvla_backbone_counterfactual_target_correct": normalize_text(cf_pred) == cf_gold,
            "openvla_backbone_schema_sensitive": sensitivity(orig_pred, cf_pred, orig_gold, cf_gold),
            "openvla_forced_choice_scores_original": orig_scores,
            "openvla_forced_choice_scores_counterfactual": cf_scores,
            "choice_scoring": args.choice_scoring,
            "openvla_direct_raw_original": orig_raw,
            "openvla_direct_raw_counterfactual": cf_raw,
            "openvla_action_sensitive": e2.get("openvla_action_sensitive"),
            "normalized_action_delta": e2.get("normalized_action_delta"),
            "action_sensitivity_threshold": e2.get("action_sensitivity_threshold"),
            "qwen_schema_sensitive": qwen_schema_sensitive(ex, q, None) if q else e2.get("qwen_schema_sensitive"),
            "qwen_target_counterfactual": ((q.get("parsed_schema") or {}) if q else (e2.get("qwen_schema_counterfactual") or {})).get("target_object"),
            "smoke_test": bool(placeholder or ex.get("smoke_test")),
        }
        rows.append(row)
        if args.log_every > 0 and (i % args.log_every == 0 or i == len(examples)):
            print(f"progress {i}/{len(examples)}", flush=True)
        if args.save_every > 0 and i % args.save_every == 0:
            write_jsonl(out_dir / "predictions.partial.jsonl", rows)
    summary = summarize(rows)
    write_jsonl(out_dir / "predictions.jsonl", rows)
    write_json(out_dir / "metrics.json", summary)
    write_csv(out_dir / "summary.csv", [{
        "num_examples": summary["num_examples"],
        "probe_mode": summary["probe_mode"],
        "qwen_schema_sensitivity": summary["qwen_schema_sensitivity"]["mean"],
        "openvla_backbone_schema_sensitivity": summary["openvla_backbone_schema_sensitivity"]["mean"],
        "openvla_action_sensitivity": summary["openvla_action_sensitivity"]["mean"],
        "openvla_backbone_action_gap": summary["openvla_backbone_action_gap"]["mean"],
        "target_correct_original": summary["target_correct_original"]["mean"],
        "target_correct_counterfactual": summary["target_correct_counterfactual"]["mean"],
    }])
    write_csv(out_dir / "target_pair_breakdown.csv", summary["target_pair_breakdown"])
    (out_dir / "latex_table_openvla_backbone_probe.tex").write_text(latex_table(summary), encoding="utf-8")
    cases = [r for r in rows if r.get("openvla_backbone_schema_sensitive") and r.get("openvla_action_sensitive") is False][:20]
    write_json(out_dir / "qualitative_cases.json", {"backbone_correct_action_insensitive": cases})
    write_json(out_dir / "run_metadata.json", {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "benchmark": args.benchmark,
        "exp2_predictions": args.exp2_predictions,
        "qwen_predictions": args.qwen_predictions,
        "model_path": args.model_path,
        "probe_mode": args.probe_mode,
        "choice_scoring": args.choice_scoring,
        "max_examples": args.max_examples,
        "shuffle": args.shuffle,
        "sample_seed": args.sample_seed,
        "python": sys.version,
        "platform": platform.platform(),
        "note": "OpenVLA VLM/backbone semantic probe. Forced-choice uses conditional target likelihood, not trained probes or action decoding.",
    })
    print(json.dumps({
        "out_dir": str(out_dir),
        "num_examples": len(rows),
        "openvla_backbone_schema_sensitivity": summary["openvla_backbone_schema_sensitivity"],
        "openvla_action_sensitivity": summary["openvla_action_sensitivity"],
        "openvla_backbone_action_gap": summary["openvla_backbone_action_gap"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
