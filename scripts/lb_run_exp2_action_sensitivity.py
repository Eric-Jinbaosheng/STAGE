import argparse
import json
import math
import platform
import random
import re
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lb_run_openvla_action import dtype_from_args, load_hdf5_image_pointer
from linguistic_blindness.utils.io import command_string, read_jsonl, utc_timestamp, write_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 2A: VLM schema sensitivity vs OpenVLA 7-DoF action sensitivity.")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--qwen-predictions", default="", help="Qwen schema predictions on counterfactual benchmark examples.")
    p.add_argument("--qwen-original-predictions", default="", help="Optional Qwen schema predictions on original-instruction examples.")
    p.add_argument("--index", default="data/processed/unified_index.parquet", help="Used to recover obs_ptr/action std when benchmark omits them.")
    p.add_argument("--model-path", default="/scratch/bj2410/models/openvla-7b")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--method", default="qwen_schema_vs_openvla7b_action")
    p.add_argument("--perturbation-types", default="target_swap", help="Comma-separated; target_swap is the cleanest Exp2A setting.")
    p.add_argument("--dataset-filter", default="libero", help="Comma-separated datasets to keep; empty keeps all. OpenVLA needs image observations.")
    p.add_argument("--require-counterfactual-valid", action="store_true", help="Keep only examples marked counterfactual_valid=true before max_examples.")
    p.add_argument("--shuffle", action="store_true", help="Deterministically shuffle after filtering and before max_examples.")
    p.add_argument("--sample-seed", type=int, default=7)
    p.add_argument("--max-examples", type=int, default=100)
    p.add_argument("--unnorm-key", default="bridge_orig")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--attn-implementation", default="")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--control-mode", choices=["none", "same", "paraphrase", "both"], default="paraphrase")
    p.add_argument("--threshold-method", choices=["p95", "mean_plus_2std", "fixed"], default="p95")
    p.add_argument("--fixed-threshold", type=float, default=1e-6)
    p.add_argument("--allow-placeholder-image", action="store_true", help="Smoke-test only.")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=50)
    return p.parse_args()


def load_model_and_processor(args: argparse.Namespace):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"loading OpenVLA processor from {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    print("processor loaded", flush=True)
    kwargs = {"torch_dtype": dtype_from_args(args), "low_cpu_mem_usage": True, "trust_remote_code": args.trust_remote_code}
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    print(f"loading OpenVLA model dtype={kwargs['torch_dtype']} attn={args.attn_implementation or 'default'}", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(args.model_path, **kwargs)
    if args.device and not args.device.startswith("auto"):
        model = model.to(args.device)
    model.eval()
    print(f"model loaded on {next(model.parameters()).device}", flush=True)
    return model, processor


def inject_local_dataset_statistics(model: Any, model_path: str, unnorm_key: str) -> bool:
    """Merge local OpenVLA dataset statistics when config norm_stats omits them.

    Some finetuned OpenVLA checkpoints keep LIBERO normalization statistics in a
    sidecar dataset_statistics.json instead of config.json. predict_action()
    validates against model.norm_stats, so the stats must be available there.
    """
    stats_path = Path(model_path) / "dataset_statistics.json"
    if not stats_path.exists() or not hasattr(model, "norm_stats"):
        return False
    norm_stats = getattr(model, "norm_stats", {})
    if unnorm_key in norm_stats:
        return False
    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"warning: failed to read {stats_path}: {exc}", flush=True)
        return False
    if unnorm_key not in data:
        print(f"warning: {unnorm_key} not found in {stats_path}", flush=True)
        return False
    norm_stats[unnorm_key] = data[unnorm_key]
    model.norm_stats = norm_stats
    return True


def by_example_id(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for row in rows:
        eid = row.get("example_id")
        if eid:
            out[str(eid)] = row
    return out


def load_index_maps(index_path: str) -> Tuple[Dict[str, str], np.ndarray]:
    import ast
    import pandas as pd

    path = Path(index_path)
    if not path.exists():
        return {}, np.ones(7, dtype=float)
    df = pd.read_parquet(path, columns=["episode_id", "frame_id", "obs_ptr", "action"])
    obs = {}
    actions = []
    for row in df.itertuples(index=False):
        sample_id = f"{row.episode_id}:{int(row.frame_id)}"
        if row.obs_ptr:
            obs[sample_id] = str(row.obs_ptr)
        try:
            val = row.action
            arr = ast.literal_eval(val) if isinstance(val, str) else val
            arr = np.asarray(arr, dtype=float).reshape(-1)
            if arr.size == 7:
                actions.append(arr)
        except Exception:
            pass
    if actions:
        std = np.std(np.stack(actions, axis=0), axis=0)
        std = np.where(std < 1e-6, 1.0, std)
    else:
        std = np.ones(7, dtype=float)
    return obs, std.astype(float)


def resolve_obs_ptr(example: Dict[str, Any], obs_map: Dict[str, str]) -> Optional[str]:
    for key in ["obs_ptr", "observation_path", "image_path", "image"]:
        if example.get(key):
            return str(example[key])
    obs_id = str(example.get("observation_id") or "")
    return obs_map.get(obs_id)


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


def prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def paraphrase_instruction(instruction: str) -> str:
    text = instruction.strip()
    if not text:
        return text
    replacements = [
        (r"\bpick up\b", "grab"),
        (r"\bput\b", "place"),
        (r"\bopen\b", "pull open"),
        (r"\bturn on\b", "switch on"),
        (r"\bmove\b", "shift"),
    ]
    out = text
    for pat, repl in replacements:
        new = re.sub(pat, repl, out, flags=re.I)
        if new != out:
            return new
    return "Please " + text[0].lower() + text[1:]


def action_to_list(action: Any) -> List[float]:
    arr = np.asarray(action, dtype=float).reshape(-1)
    return [float(x) for x in arr.tolist()]


def predict_action(model, processor, image, instruction: str, args: argparse.Namespace) -> List[float]:
    import torch

    device = next(model.parameters()).device
    dtype = dtype_from_args(args)
    inputs = processor(prompt(instruction), image)
    if hasattr(inputs, "to"):
        inputs = inputs.to(device) if dtype == "auto" else inputs.to(device, dtype=dtype)
    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
    return action_to_list(action)


def vec(action: List[float], size: int = 7) -> np.ndarray:
    arr = np.asarray(action, dtype=float).reshape(-1)
    if arr.size < size:
        arr = np.pad(arr, (0, size - arr.size), mode="constant")
    return arr[:size]


def action_stats(a: List[float], b: List[float], action_std: np.ndarray) -> Dict[str, float]:
    va, vb = vec(a), vec(b)
    diff = va - vb
    pos = float(np.linalg.norm(diff[:3]))
    rot = float(np.linalg.norm(diff[3:6]))
    grip = float(abs(diff[6]))
    norm = float(np.linalg.norm(diff / action_std))
    na, nb = va[:6], vb[:6]
    denom = float(np.linalg.norm(na) * np.linalg.norm(nb))
    cosine = None if denom < 1e-12 else float(np.dot(na, nb) / denom)
    return {"delta_pos": pos, "delta_rot": rot, "delta_gripper": grip, "normalized_action_delta": norm, "action_cosine": cosine}


def qwen_schema_sensitive(example: Dict[str, Any], qwen_cf: Optional[Dict[str, Any]], qwen_orig: Optional[Dict[str, Any]]) -> Optional[bool]:
    ptype = example.get("perturbation_type")
    if ptype != "target_swap":
        return None
    cf_target = example.get("counterfactual_target_object")
    orig_target = example.get("original_target_object")
    if not qwen_cf or not cf_target:
        return None
    cf_schema = qwen_cf.get("parsed_schema", {}) or {}
    cf_ok = cf_schema.get("target_object") == cf_target
    if qwen_orig:
        orig_schema = qwen_orig.get("parsed_schema", {}) or {}
        return bool(cf_ok and orig_schema.get("target_object") == orig_target)
    return bool(cf_ok and orig_target != cf_target)


def aggregate_bool(vals: List[Optional[bool]]) -> Dict[str, Any]:
    clean = [v for v in vals if v is not None]
    if not clean:
        return {"mean": None, "count": 0}
    return {"mean": sum(bool(v) for v in clean) / len(clean), "count": len(clean)}


def aggregate_float(vals: List[Optional[float]]) -> Dict[str, Any]:
    clean = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not clean:
        return {"mean": None, "count": 0}
    return {"mean": sum(clean) / len(clean), "count": len(clean)}


def threshold_from_controls(rows: List[Dict[str, Any]], method: str, fixed: float) -> float:
    vals = [r.get("control_normalized_action_delta") for r in rows if r.get("control_normalized_action_delta") is not None]
    vals = [float(v) for v in vals if not math.isnan(float(v))]
    if method == "fixed" or not vals:
        return fixed
    if method == "mean_plus_2std":
        return float(mean(vals) + 2.0 * pstdev(vals))
    return float(np.percentile(np.asarray(vals, dtype=float), 95))


def summarize(rows: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    for r in rows:
        sens = r["normalized_action_delta"] >= threshold
        r["openvla_action_sensitive"] = bool(sens)
        r["low_action_sensitivity"] = not bool(sens)
        r["semantic_action_gap_case"] = bool(r.get("qwen_schema_sensitive") is True and not sens)
        r["action_sensitivity_threshold"] = threshold
    schema = aggregate_bool([r.get("qwen_schema_sensitive") for r in rows])
    action = aggregate_bool([r.get("openvla_action_sensitive") for r in rows])
    gap = None if schema["mean"] is None or action["mean"] is None else schema["mean"] - action["mean"]
    out = {
        "num_examples": len(rows),
        "action_sensitivity_threshold": threshold,
        "vlm_schema_sensitivity": schema,
        "openvla_action_sensitivity": action,
        "semantic_action_gap": {"mean": gap, "count": min(schema["count"], action["count"])},
        "low_sensitivity_rate": aggregate_bool([r.get("low_action_sensitivity") for r in rows]),
        "semantic_action_gap_case_rate": aggregate_bool([r.get("semantic_action_gap_case") for r in rows]),
        "mean_delta_pos": aggregate_float([r.get("delta_pos") for r in rows]),
        "mean_delta_rot": aggregate_float([r.get("delta_rot") for r in rows]),
        "mean_delta_gripper": aggregate_float([r.get("delta_gripper") for r in rows]),
        "mean_normalized_action_delta": aggregate_float([r.get("normalized_action_delta") for r in rows]),
        "mean_action_cosine": aggregate_float([r.get("action_cosine") for r in rows]),
        "mean_control_normalized_action_delta": aggregate_float([r.get("control_normalized_action_delta") for r in rows]),
    }
    by_ptype = {}
    for ptype in sorted({str(r.get("perturbation_type")) for r in rows}):
        prows = [r for r in rows if str(r.get("perturbation_type")) == ptype]
        ps = aggregate_bool([r.get("qwen_schema_sensitive") for r in prows])
        pa = aggregate_bool([r.get("openvla_action_sensitive") for r in prows])
        by_ptype[ptype] = {
            "count": len(prows),
            "vlm_schema_sensitivity": ps,
            "openvla_action_sensitivity": pa,
            "semantic_action_gap": None if ps["mean"] is None or pa["mean"] is None else ps["mean"] - pa["mean"],
            "low_sensitivity_rate": aggregate_bool([r.get("low_action_sensitivity") for r in prows]),
            "mean_normalized_action_delta": aggregate_float([r.get("normalized_action_delta") for r in prows]),
            "mean_action_cosine": aggregate_float([r.get("action_cosine") for r in prows]),
        }
    out["per_perturbation"] = by_ptype
    return out


def write_tables(out_dir: Path, summary: Dict[str, Any]) -> None:
    rows2a = [{
        "condition": "all",
        "vlm_schema_sensitivity": summary["vlm_schema_sensitivity"]["mean"],
        "openvla_action_sensitivity": summary["openvla_action_sensitivity"]["mean"],
        "semantic_action_gap": summary["semantic_action_gap"]["mean"],
        "low_sensitivity_rate": summary["low_sensitivity_rate"]["mean"],
        "count": summary["num_examples"],
    }]
    for ptype, vals in summary["per_perturbation"].items():
        rows2a.append({
            "condition": ptype,
            "vlm_schema_sensitivity": vals["vlm_schema_sensitivity"]["mean"],
            "openvla_action_sensitivity": vals["openvla_action_sensitivity"]["mean"],
            "semantic_action_gap": vals["semantic_action_gap"],
            "low_sensitivity_rate": vals["low_sensitivity_rate"]["mean"],
            "count": vals["count"],
        })
    write_csv(out_dir / "table2a_semantic_action_gap.csv", rows2a)
    detail = [{
        "condition": "all",
        "delta_pos": summary["mean_delta_pos"]["mean"],
        "delta_rot": summary["mean_delta_rot"]["mean"],
        "delta_gripper": summary["mean_delta_gripper"]["mean"],
        "normalized_action_delta": summary["mean_normalized_action_delta"]["mean"],
        "action_cosine": summary["mean_action_cosine"]["mean"],
        "low_sensitivity_rate": summary["low_sensitivity_rate"]["mean"],
    }]
    write_csv(out_dir / "table2b_action_sensitivity_details.csv", detail)
    (out_dir / "table2a_semantic_action_gap.tex").write_text(latex_table2a(rows2a), encoding="utf-8")
    (out_dir / "table2b_action_sensitivity_details.tex").write_text(latex_table2b(detail), encoding="utf-8")


def fmt(x: Any) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def latex_table2a(rows: List[Dict[str, Any]]) -> str:
    lines = ["\\begin{tabular}{lrrrrr}", "Condition & VLM Schema & OpenVLA Action & SAG & Low Sens. & N \\\\", "\\hline"]
    for r in rows:
        lines.append(f"{r['condition']} & {fmt(r['vlm_schema_sensitivity'])} & {fmt(r['openvla_action_sensitivity'])} & {fmt(r['semantic_action_gap'])} & {fmt(r['low_sensitivity_rate'])} & {r['count']} \\\\")
    lines.append("\\end{tabular}\n")
    return "\n".join(lines)


def latex_table2b(rows: List[Dict[str, Any]]) -> str:
    lines = ["\\begin{tabular}{lrrrrr}", "Condition & $\\Delta$pos & $\\Delta$rot & $\\Delta$grip & Norm. $\\Delta$ & Cosine \\\\", "\\hline"]
    for r in rows:
        lines.append(f"{r['condition']} & {fmt(r['delta_pos'])} & {fmt(r['delta_rot'])} & {fmt(r['delta_gripper'])} & {fmt(r['normalized_action_delta'])} & {fmt(r['action_cosine'])} \\\\")
    lines.append("\\end{tabular}\n")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ptypes = {x.strip() for x in args.perturbation_types.split(",") if x.strip()}
    datasets = {x.strip() for x in args.dataset_filter.split(",") if x.strip()}
    qwen_cf = by_example_id(read_jsonl(args.qwen_predictions)) if args.qwen_predictions else {}
    qwen_orig = by_example_id(read_jsonl(args.qwen_original_predictions)) if args.qwen_original_predictions else {}
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
    obs_map, action_std = load_index_maps(args.index)
    write_json(out_dir / "action_std.json", {"action_std": action_std.tolist(), "source": args.index})
    print(f"loaded examples={len(examples)} qwen_cf={len(qwen_cf)} action_std={action_std.tolist()}", flush=True)
    model, processor = load_model_and_processor(args)
    injected_stats = inject_local_dataset_statistics(model, args.model_path, args.unnorm_key)
    if injected_stats:
        print(f"injected local dataset_statistics.json for unnorm_key={args.unnorm_key}", flush=True)
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples, start=1):
        image, image_source, placeholder = load_image_for_example(ex, obs_map, args.allow_placeholder_image)
        orig_instr = str(ex.get("original_instruction") or "")
        cf_instr = str(ex.get("counterfactual_instruction") or ex.get("instruction") or "")
        action_orig = predict_action(model, processor, image, orig_instr, args)
        action_cf = predict_action(model, processor, image, cf_instr, args)
        stats = action_stats(action_orig, action_cf, action_std)
        control_action = None
        control_stats = {}
        control_instruction = None
        if args.control_mode in {"same", "both"}:
            control_instruction = orig_instr
            control_action = predict_action(model, processor, image, control_instruction, args)
            control_stats = action_stats(action_orig, control_action, action_std)
        if args.control_mode in {"paraphrase", "both"}:
            control_instruction = paraphrase_instruction(orig_instr)
            control_action = predict_action(model, processor, image, control_instruction, args)
            para_stats = action_stats(action_orig, control_action, action_std)
            if not control_stats or para_stats["normalized_action_delta"] > control_stats["normalized_action_delta"]:
                control_stats = para_stats
        row = {
            "example_id": ex.get("example_id"),
            "observation_id": ex.get("observation_id"),
            "perturbation_type": ex.get("perturbation_type"),
            "original_instruction": orig_instr,
            "counterfactual_instruction": cf_instr,
            "original_target_object": ex.get("original_target_object"),
            "counterfactual_target_object": ex.get("counterfactual_target_object"),
            "qwen_schema_original": (qwen_orig.get(str(ex.get("example_id"))) or {}).get("parsed_schema"),
            "qwen_schema_counterfactual": (qwen_cf.get(str(ex.get("example_id"))) or {}).get("parsed_schema"),
            "qwen_target_original": ((qwen_orig.get(str(ex.get("example_id"))) or {}).get("parsed_schema") or {}).get("target_object"),
            "qwen_target_counterfactual": ((qwen_cf.get(str(ex.get("example_id"))) or {}).get("parsed_schema") or {}).get("target_object"),
            "qwen_schema_sensitive": qwen_schema_sensitive(ex, qwen_cf.get(str(ex.get("example_id"))), qwen_orig.get(str(ex.get("example_id")))),
            "openvla_action_original": action_orig,
            "openvla_action_counterfactual": action_cf,
            "control_instruction": control_instruction,
            "control_action": control_action,
            "control_normalized_action_delta": control_stats.get("normalized_action_delta"),
            "image_source": image_source,
            "placeholder_image": placeholder,
            "smoke_test": bool(placeholder or ex.get("smoke_test")),
            **stats,
        }
        rows.append(row)
        if args.log_every > 0 and (i % args.log_every == 0 or i == len(examples)):
            print(f"progress {i}/{len(examples)}", flush=True)
        if args.save_every > 0 and i % args.save_every == 0:
            write_jsonl(out_dir / "predictions.partial.jsonl", rows)
    threshold = threshold_from_controls(rows, args.threshold_method, args.fixed_threshold)
    summary = summarize(rows, threshold)
    write_jsonl(out_dir / "predictions.jsonl", rows)
    write_json(out_dir / "metrics.json", summary)
    write_json(out_dir / "adapter_analysis.json", {"status": "not_implemented", "reason": "Primary Exp2 uses native 7-DoF action sensitivity. Target attribution requires object positions and is secondary."})
    write_tables(out_dir, summary)
    write_json(out_dir / "run_metadata.json", {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "benchmark": args.benchmark,
        "qwen_predictions": args.qwen_predictions,
        "model_path": args.model_path,
        "method": args.method,
        "max_examples": args.max_examples,
        "require_counterfactual_valid": args.require_counterfactual_valid,
        "shuffle": args.shuffle,
        "sample_seed": args.sample_seed,
        "control_mode": args.control_mode,
        "threshold_method": args.threshold_method,
        "threshold": threshold,
        "unnorm_key": args.unnorm_key,
        "injected_local_dataset_statistics": injected_stats,
        "python": sys.version,
        "platform": platform.platform(),
        "note": "Primary evidence is native OpenVLA 7-DoF counterfactual action sensitivity; adapter target attribution is secondary.",
    })
    print(json.dumps({"out_dir": str(out_dir), "num_examples": len(rows), "threshold": threshold, "semantic_action_gap": summary["semantic_action_gap"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
