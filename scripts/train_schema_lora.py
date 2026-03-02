import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA SFT training for schema-text VLM.")
    p.add_argument("--model-name", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--schema-text", required=True)
    p.add_argument("--instr-wrong", default="")
    p.add_argument("--split-path", default="")
    p.add_argument("--split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--max-val-batches", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated LoRA target module names.",
    )
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--device-map", default="")
    p.add_argument("--flash-attn-2", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--load-in-8bit", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_dtype(args: argparse.Namespace) -> Optional[torch.dtype]:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return None


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def build_model_kwargs(batch: dict) -> dict:
    kwargs = {"labels": batch["labels"]}
    for key, value in batch.items():
        if key == "labels":
            continue
        if torch.is_tensor(value):
            kwargs[key] = value
    return kwargs


def linear_warmup_decay(step: int, max_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    if max_steps <= warmup_steps:
        return 1.0
    remain = max_steps - step - 1
    decay_steps = max_steps - warmup_steps
    return max(0.0, float(remain) / float(max(1, decay_steps)))


@torch.no_grad()
def evaluate(model, loader, device: torch.device, max_batches: int) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = to_device(batch, device)
        out = model(**build_model_kwargs(batch))
        losses.append(float(out.loss.detach().cpu()))
    model.train()
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def save_checkpoint(model, processor, out_dir: Path, step: int, best_val_loss: Optional[float], config: dict) -> None:
    ckpt_dir = out_dir / "checkpoints" / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(ckpt_dir))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(ckpt_dir / "processor"))
    payload = {
        "step": step,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    (ckpt_dir / "train_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    from data.collator import SchemaTextSFTCollator
    from data.dataset_sft import SchemaSFTDataset
    from model.load_vlm import load_vlm_and_processor, print_trainable_parameter_summary

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lora_target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    model, processor = load_vlm_and_processor(
        model_name_or_path=args.model_name,
        trust_remote_code=bool(args.trust_remote_code),
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
        torch_dtype=(
            "bfloat16" if args.bf16 else
            "float16" if args.fp16 else
            "auto"
        ),
        device_map=(args.device_map or "auto"),
        use_flash_attention_2=bool(args.flash_attn_2),
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
    )

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device_map:
        # model sharded by transformers/bitsandbytes; leave placement to backend
        pass
    else:
        model.to(device)

    train_ds = SchemaSFTDataset(
        index_path=args.index,
        schema_text_path=args.schema_text,
        instr_wrong_path=args.instr_wrong or "",
        split_path=args.split_path or "",
        split_name=args.split or "",
        image_size=args.image_size,
    )
    train_collator = SchemaTextSFTCollator(processor=processor, max_length=args.max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=train_collator,
    )

    val_loader = None
    val_ds = None
    if args.split_path and args.val_split:
        val_ds = SchemaSFTDataset(
            index_path=args.index,
            schema_text_path=args.schema_text,
            instr_wrong_path=args.instr_wrong or "",
            split_path=args.split_path,
            split_name=args.val_split,
            image_size=args.image_size,
        )
        val_collator = SchemaTextSFTCollator(processor=processor, max_length=args.max_length)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=val_collator,
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    warmup_steps = int(args.max_steps * args.warmup_ratio)
    scaler = None
    use_amp = device.type == "cuda" and (args.bf16 or args.fp16)

    config = {
        "model_name": args.model_name,
        "index": args.index,
        "schema_text": args.schema_text,
        "instr_wrong": args.instr_wrong,
        "split_path": args.split_path,
        "split": args.split,
        "val_split": args.val_split,
        "image_size": args.image_size,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": lora_target_modules,
        },
        "vlm": {
            "trust_remote_code": args.trust_remote_code,
            "device_map": args.device_map,
            "flash_attn_2": args.flash_attn_2,
            "load_in_4bit": args.load_in_4bit,
            "load_in_8bit": args.load_in_8bit,
            "dtype": str(pick_dtype(args)),
        },
        "trainable_parameter_summary": print_trainable_parameter_summary(model),
    }
    (out_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("Running config:")
    print(json.dumps(config, indent=2))

    global_step = 0
    best_val_loss = None
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    while global_step < args.max_steps:
        for batch in train_loader:
            batch = to_device(batch, device)
            model_kwargs = build_model_kwargs(batch)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=pick_dtype(args) or torch.float16):
                    out = model(**model_kwargs)
                    loss = out.loss / args.grad_accum
            else:
                out = model(**model_kwargs)
                loss = out.loss / args.grad_accum

            loss.backward()
            running_loss += float(loss.detach().cpu())

            if (global_step + 1) % args.grad_accum == 0:
                optimizer.step()
                scale = linear_warmup_decay(global_step, args.max_steps, warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * scale
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if global_step % args.log_every == 0:
                payload = {
                    "step": global_step,
                    "loss": running_loss / float(args.log_every),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                print(json.dumps(payload))
                running_loss = 0.0

            if val_loader is not None and global_step % args.eval_every == 0:
                val_loss = evaluate(model, val_loader, device, args.max_val_batches)
                payload = {"step": global_step, "val_loss": val_loss}
                print(json.dumps(payload))
                if best_val_loss is None or (val_loss == val_loss and val_loss < best_val_loss):
                    best_val_loss = val_loss
                    best_dir = out_dir / "best"
                    best_dir.mkdir(parents=True, exist_ok=True)
                    if hasattr(model, "save_pretrained"):
                        model.save_pretrained(str(best_dir))
                    if hasattr(processor, "save_pretrained"):
                        processor.save_pretrained(str(best_dir / "processor"))
                    (best_dir / "best_metrics.json").write_text(
                        json.dumps({"step": global_step, "val_loss": val_loss}, indent=2),
                        encoding="utf-8",
                    )

            if global_step % args.save_every == 0:
                save_checkpoint(model, processor, out_dir, global_step, best_val_loss, config)

            if global_step >= args.max_steps:
                break

    last_dir = out_dir / "last"
    last_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(last_dir))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(last_dir / "processor"))
    (last_dir / "final_metrics.json").write_text(
        json.dumps({"step": global_step, "best_val_loss": best_val_loss}, indent=2),
        encoding="utf-8",
    )

    train_ds.close()
    if val_ds is not None:
        val_ds.close()

    print(f"Training finished. Outputs in {out_dir}")
