from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one-sample preview through a VLM processor/model.")
    p.add_argument("--index", default="data/processed/libero_index.parquet")
    p.add_argument("--schema-text", default="data/processed/schema_text.parquet")
    p.add_argument("--instr-wrong", default="data/processed/instr_wrong.parquet")
    p.add_argument("--split-path", default="")
    p.add_argument("--split", default="", choices=["", "train", "val", "test"])
    p.add_argument("--model-name", required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--row-idx", type=int, default=0)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--use-generate", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=64)
    return p.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.data.dataset_sft import SchemaSFTDataset
    from src.model.load_vlm import load_vlm_and_processor, print_trainable_parameter_summary

    args = parse_args()
    ds = SchemaSFTDataset(
        index_path=args.index,
        schema_text_path=args.schema_text,
        instr_wrong_path=(args.instr_wrong or None),
        image_size=args.image_size,
        split_path=(args.split_path or None),
        split_name=(args.split or None),
    )
    if len(ds) == 0:
        raise RuntimeError("SchemaSFTDataset is empty")
    idx = max(0, min(int(args.row_idx), len(ds) - 1))
    sample = ds[idx]

    model, processor = load_vlm_and_processor(
        model_name_or_path=args.model_name,
        trust_remote_code=bool(args.trust_remote_code),
        use_lora=False,
    )
    summary = print_trainable_parameter_summary(model)

    inputs = processor(
        text=[sample["prompt_text"]],
        images=[sample["image"]],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    shape_summary = {k: list(v.shape) for k, v in inputs.items() if hasattr(v, "shape")}

    payload = {
        "sample_id": sample["sample_id"],
        "instruction": sample["instruction"],
        "prompt_text": sample["prompt_text"],
        "target_schema_text": sample["target_schema_text"],
        "processor_shapes": shape_summary,
        "model_param_summary": summary,
    }

    if args.use_generate:
        import torch

        model.eval()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(args.max_new_tokens),
            )
        tokenizer = getattr(processor, "tokenizer", processor)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        payload["generated_text"] = decoded[0] if decoded else ""

    print(json.dumps(payload, indent=2, ensure_ascii=True))
    ds.close()


if __name__ == "__main__":
    main()
