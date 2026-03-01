import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate schema-text generation model.")
    p.add_argument("--model-name", required=True)
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--schema-text", required=True)
    p.add_argument("--instr-wrong", default="")
    p.add_argument("--split-path", default="")
    p.add_argument("--split", default="")
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--out", required=True)
    return p.parse_args()


def pick_dtype(args: argparse.Namespace):
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return None


def normalize_tri(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"TRUE", "1"}:
        return "T"
    if text in {"FALSE", "0"}:
        return "F"
    if text in {"T", "F", "UNK"}:
        return text
    return "UNK"


def generate_schema_text(model, processor, sample: dict, max_length: int, max_new_tokens: int, device: torch.device) -> str:
    encoded = processor(
        text=[sample["prompt_text"]],
        images=[sample["image"].numpy()],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    for key, value in list(encoded.items()):
        if torch.is_tensor(value):
            encoded[key] = value.to(device)
    out = model.generate(**encoded, max_new_tokens=max_new_tokens)
    decoded = processor.batch_decode(out, skip_special_tokens=True)
    text = decoded[0] if decoded else ""
    marker = "OUTPUT_SCHEMA:"
    if marker in text:
        text = text.split(marker, 1)[1].strip()
    return text.strip()


def score_aff(gt: Dict[str, int], pred: Dict[str, int]) -> Dict[str, float]:
    keys = sorted(set(gt.keys()) | set(pred.keys()))
    tp = fp = fn = 0
    for key in keys:
        g = int(gt.get(key, 0))
        p = int(pred.get(key, 0))
        if g == 1 and p == 1:
            tp += 1
        elif g == 0 and p == 1:
            fp += 1
        elif g == 1 and p == 0:
            fn += 1
    precision = tp / float(max(1, tp + fp))
    recall = tp / float(max(1, tp + fn))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    args = parse_args()

    from peft import PeftModel

    from data.dataset_sft import SchemaSFTDataset
    from data.schema_text import parse_schema_text
    from model.load_vlm import VLMConfig, load_processor, load_vlm_model

    vlm_cfg = VLMConfig(
        model_name=args.model_name,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=pick_dtype(args),
    )
    processor = load_processor(vlm_cfg)
    base_model = load_vlm_model(vlm_cfg)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ds = SchemaSFTDataset(
        index_path=args.index,
        schema_text_path=args.schema_text,
        instr_wrong_path=args.instr_wrong or "",
        split_path=args.split_path or "",
        split_name=args.split or "",
        image_size=args.image_size,
    )

    target_correct = 0
    phase_correct = 0
    state_total = 0
    state_correct = 0
    aff_scores: List[float] = []

    blank_total = 0
    blank_changed = 0
    swap_total = 0
    swap_flip_correct = 0

    cases: List[Dict[str, Any]] = []
    rows = min(len(ds), args.max_samples)
    for i in range(rows):
        sample = ds[i]
        pred_text = generate_schema_text(
            model=model,
            processor=processor,
            sample=sample,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        pred = parse_schema_text(pred_text)
        gt = parse_schema_text(sample["target_schema_text"])

        if pred.get("TARGET", "UNK") == gt.get("TARGET", "UNK"):
            target_correct += 1
        if pred.get("PHASE", "UNK") == gt.get("PHASE", "UNK"):
            phase_correct += 1

        for key in ("SECURED", "CONTACT", "OCCLUDED"):
            state_total += 1
            if normalize_tri(pred.get(key, "UNK")) == normalize_tri(gt.get(key, "UNK")):
                state_correct += 1

        aff = score_aff(gt.get("AFF", {}), pred.get("AFF", {}))
        aff_scores.append(aff["f1"])

        pred_blank_target = ""
        if sample.get("instr_blank", "") != "":
            blank_total += 1
            blank_sample = dict(sample)
            blank_sample["prompt_text"] = sample["prompt_text"].replace(sample["instruction"], sample["instr_blank"])
            blank_text = generate_schema_text(
                model=model,
                processor=processor,
                sample=blank_sample,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            pred_blank = parse_schema_text(blank_text)
            pred_blank_target = pred_blank.get("TARGET", "UNK")
            if pred_blank_target != pred.get("TARGET", "UNK"):
                blank_changed += 1

        pred_swap_target = ""
        if sample.get("swap_valid", False) and sample.get("instr_swap", ""):
            swap_total += 1
            swap_sample = dict(sample)
            swap_sample["prompt_text"] = sample["prompt_text"].replace(sample["instruction"], sample["instr_swap"])
            swap_text = generate_schema_text(
                model=model,
                processor=processor,
                sample=swap_sample,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            pred_swap = parse_schema_text(swap_text)
            pred_swap_target = pred_swap.get("TARGET", "UNK")
            src = gt.get("TARGET", "UNK")
            tgt = pred_swap_target
            try:
                swap_meta = json.loads(sample.get("swap_meta", "{}") or "{}")
            except json.JSONDecodeError:
                swap_meta = {}
            want_tgt = str(swap_meta.get("swap_target", "UNK"))
            if pred.get("TARGET", "UNK") == src and pred_swap_target == want_tgt:
                swap_flip_correct += 1

        if len(cases) < 30:
            cases.append(
                {
                    "sample_id": sample["sample_id"],
                    "instruction": sample["instruction"],
                    "instr_blank": sample.get("instr_blank", ""),
                    "instr_swap": sample.get("instr_swap", ""),
                    "swap_valid": bool(sample.get("swap_valid", False)),
                    "gt_schema_text": sample["target_schema_text"],
                    "pred_schema_text": pred_text,
                    "pred_target_blank": pred_blank_target,
                    "pred_target_swap": pred_swap_target,
                }
            )

    ds.close()

    result = {
        "metrics": {
            "target_acc": target_correct / float(max(1, rows)),
            "phase_acc": phase_correct / float(max(1, rows)),
            "state_acc": state_correct / float(max(1, state_total)),
            "aff_mask_f1": sum(aff_scores) / float(max(1, len(aff_scores))),
            "schema_sensitivity_score_target": blank_changed / float(max(1, blank_total)),
            "target_flip_rate_swap": swap_flip_correct / float(max(1, swap_total)),
            "evaluated_samples": rows,
        },
        "language_sensitivity": {
            "blank_total": blank_total,
            "blank_changed": blank_changed,
            "swap_total": swap_total,
            "swap_flip_correct": swap_flip_correct,
        },
        "qual_cases": cases,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print("target_acc", result["metrics"]["target_acc"])
    print("phase_acc", result["metrics"]["phase_acc"])
    print("state_acc", result["metrics"]["state_acc"])
    print("aff_mask_f1", result["metrics"]["aff_mask_f1"])
    print("schema_sensitivity_score_target", result["metrics"]["schema_sensitivity_score_target"])
    print("target_flip_rate_swap", result["metrics"]["target_flip_rate_swap"])
