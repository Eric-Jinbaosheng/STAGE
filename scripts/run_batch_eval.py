import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run batch evaluation from a trained baseline checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    p.add_argument("--text-vocab", required=True, help="Path to text_vocab.json")
    p.add_argument("--index", default="", help="Optional override for index parquet")
    p.add_argument("--labels", default="", help="Optional override for labels parquet")
    p.add_argument("--dataset", default="", help="Optional dataset filter, e.g. libero or handover_sim")
    p.add_argument("--split-path", default="", help="Optional episode split JSON.")
    p.add_argument("--split", default="", choices=["", "train", "val", "test"], help="Optional split name.")
    p.add_argument("--max-samples", type=int, default=128, help="Maximum number of samples to evaluate")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    p.add_argument("--out", default="", help="Optional JSON output path")
    return p.parse_args()


def update_confusion(matrix, y_true, y_pred) -> None:
    for t, p in zip(y_true, y_pred):
        matrix[int(t)][int(p)] += 1


def per_class_stats(matrix, labels):
    stats = []
    n = len(labels)
    for i in range(n):
        tp = int(matrix[i][i])
        support = int(sum(matrix[i]))
        pred_total = int(sum(matrix[r][i] for r in range(n)))
        precision = float(tp / pred_total) if pred_total > 0 else 0.0
        recall = float(tp / support) if support > 0 else 0.0
        stats.append(
            {
                "label": str(labels[i]),
                "precision": precision,
                "recall": recall,
                "support": support,
            }
        )
    return stats


def encode_instruction_text(ds, text: str):
    import torch

    input_ids, attention_mask = ds.encode_instruction(text)
    return torch.tensor(input_ids).unsqueeze(0).long(), torch.tensor(attention_mask).unsqueeze(0).float()


def main() -> None:
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
    except Exception as exc:
        raise RuntimeError("torch is required for run_batch_eval.py") from exc

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.data.dataloader import UnifiedSchemaDataset

    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt["args"]

    index_path = args.index or train_args["index"]
    labels_path = args.labels or train_args["labels"]
    view_names = [x.strip() for x in str(train_args.get("views", "agentview_rgb")).split(",") if x.strip()]

    ds = UnifiedSchemaDataset(
        index_path,
        labels_path,
        use_image=bool(train_args.get("use_image", False)),
        image_size=int(train_args.get("image_size", 64)),
        frame_window=int(train_args.get("frame_window", 1)),
        view_names=view_names,
        text_vocab_size=int(train_args.get("text_vocab_size", 2048)),
        text_max_len=int(train_args.get("text_max_len", 16)),
        text_vocab_path=args.text_vocab,
        split_path=(args.split_path or None),
        split_name=(args.split or None),
    )

    feature_dim = 5
    hidden_dim = 128
    object_classes = max(2, len(ds.object_vocab))
    phase_classes = max(2, len(ds.phase_vocab))
    text_embed_dim = int(train_args.get("text_embed_dim", 64))
    text_hidden_dim = int(train_args.get("text_hidden_dim", 64))
    use_image = bool(train_args.get("use_image", False))

    class MultiHeadModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.use_image = use_image
            if self.use_image:
                self.image_encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(ds.image_channels, 32, kernel_size=5, stride=2, padding=2),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((1, 1)),
                    torch.nn.Flatten(),
                )
                img_dim = 96
            else:
                self.image_encoder = None
                img_dim = 0
            self.text_embedding = torch.nn.Embedding(ds.text_vocab_size, text_embed_dim, padding_idx=ds.pad_id)
            self.text_rnn = torch.nn.GRU(
                input_size=text_embed_dim,
                hidden_size=text_hidden_dim,
                batch_first=True,
            )
            fusion_in = feature_dim + text_hidden_dim + img_dim
            self.fusion = torch.nn.Sequential(
                torch.nn.Linear(fusion_in, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )
            self.object_head = torch.nn.Linear(hidden_dim, object_classes)
            self.phase_head = torch.nn.Linear(hidden_dim, phase_classes)

        def forward(self, x_num, x_img, input_ids, attention_mask):
            tok = self.text_embedding(input_ids)
            lengths = torch.clamp(attention_mask.sum(dim=1).long(), min=1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                tok,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, h_n = self.text_rnn(packed)
            text_feat = h_n[-1]
            parts = [x_num, text_feat]
            if self.use_image:
                parts.append(self.image_encoder(x_img))
            x = torch.cat(parts, dim=1)
            z = self.fusion(x)
            return self.object_head(z), self.phase_head(z)

    selected_indices = list(range(len(ds)))
    if args.dataset:
        selected_indices = [i for i in selected_indices if str(ds.df.iloc[i].get("dataset", "")) == args.dataset]
    if args.max_samples > 0:
        selected_indices = selected_indices[: args.max_samples]
    subset = Subset(ds, selected_indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

    model = MultiHeadModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    obj_correct = 0
    phase_correct = 0
    total = 0
    blank_changed = 0
    blank_total = 0
    swap_flip_correct = 0
    swap_total = 0
    obj_conf = [[0 for _ in range(object_classes)] for _ in range(object_classes)]
    phase_conf = [[0 for _ in range(phase_classes)] for _ in range(phase_classes)]
    preview = []
    qual_cases: List[Dict] = []

    with torch.no_grad():
        base_offset = 0
        for batch in loader:
            x_num = batch["x"].float()
            x_img = batch["image"].float()
            input_ids = batch["input_ids"].long()
            attention_mask = batch["attention_mask"].float()
            y_obj = batch["label_target_object"].long()
            y_phase = batch["label_phase"].long()

            logits_obj, logits_phase = model(x_num, x_img, input_ids, attention_mask)
            pred_obj = torch.argmax(logits_obj, dim=1)
            pred_phase = torch.argmax(logits_phase, dim=1)

            obj_correct += int((pred_obj == y_obj).sum().item())
            phase_correct += int((pred_phase == y_phase).sum().item())
            batch_n = int(y_obj.shape[0])
            total += batch_n
            update_confusion(obj_conf, y_obj.tolist(), pred_obj.tolist())
            update_confusion(phase_conf, y_phase.tolist(), pred_phase.tolist())

            if len(preview) < 5 or len(qual_cases) < 30:
                probs_obj = torch.softmax(logits_obj, dim=1)
                probs_phase = torch.softmax(logits_phase, dim=1)
                for j in range(batch_n):
                    global_idx = selected_indices[base_offset + j]
                    row = ds.df.iloc[global_idx]
                    orig_pred_obj = ds.object_vocab[int(pred_obj[j].item())]
                    instr_blank = str(row.get("instr_blank", ""))
                    instr_swap = str(row.get("instr_swap", ""))
                    swap_valid = bool(row.get("swap_valid", False))

                    x_num_one = batch["x"][j].unsqueeze(0).float()
                    x_img_one = batch["image"][j].unsqueeze(0).float()

                    blank_ids, blank_mask = encode_instruction_text(ds, instr_blank)
                    blank_logits_obj, _ = model(x_num_one, x_img_one, blank_ids, blank_mask)
                    blank_pred_obj = ds.object_vocab[int(torch.argmax(blank_logits_obj, dim=1)[0].item())]
                    blank_total += 1
                    if blank_pred_obj != orig_pred_obj:
                        blank_changed += 1

                    swap_pred_obj = ""
                    if swap_valid:
                        swap_ids, swap_mask = encode_instruction_text(ds, instr_swap)
                        swap_logits_obj, _ = model(x_num_one, x_img_one, swap_ids, swap_mask)
                        swap_pred_obj = ds.object_vocab[int(torch.argmax(swap_logits_obj, dim=1)[0].item())]
                        swap_total += 1
                        swap_meta_raw = str(row.get("swap_meta", "{}"))
                        try:
                            swap_meta = json.loads(swap_meta_raw)
                        except Exception:
                            swap_meta = {}
                        src_obj = str(swap_meta.get("src", ""))
                        tgt_obj = str(swap_meta.get("tgt", ""))
                        if orig_pred_obj == src_obj and swap_pred_obj == tgt_obj:
                            swap_flip_correct += 1

                    if len(preview) < 5:
                        preview.append(
                            {
                                "episode_id": str(row.get("episode_id", "")),
                                "frame_id": int(row.get("frame_id", 0)),
                                "pred_target_object": orig_pred_obj,
                                "pred_phase": ds.phase_vocab[int(pred_phase[j].item())],
                                "true_target_object": ds.object_vocab[int(y_obj[j].item())],
                                "true_phase": ds.phase_vocab[int(y_phase[j].item())],
                                "target_object_score": float(probs_obj[j, int(pred_obj[j].item())].item()),
                                "phase_score": float(probs_phase[j, int(pred_phase[j].item())].item()),
                            }
                        )

                    if len(qual_cases) < 30:
                        try:
                            swap_meta = json.loads(str(row.get("swap_meta", "{}")))
                        except Exception:
                            swap_meta = {}
                        qual_cases.append(
                            {
                                "episode_id": str(row.get("episode_id", "")),
                                "frame_id": int(row.get("frame_id", 0)),
                                "instruction": str(row.get("instruction", "")),
                                "instr_blank": instr_blank,
                                "instr_swap": instr_swap,
                                "swap_valid": swap_valid,
                                "gt_target_object": str(row.get("target_object", "")),
                                "gt_phase": str(row.get("phase", "")),
                                "pred_target_object": orig_pred_obj,
                                "pred_phase": ds.phase_vocab[int(pred_phase[j].item())],
                                "pred_target_object_blank": blank_pred_obj,
                                "pred_target_object_swap": swap_pred_obj,
                                "swap_meta": swap_meta,
                            }
                        )
            base_offset += batch_n

    result = {
        "config": {
            "checkpoint": args.checkpoint,
            "text_vocab": args.text_vocab,
            "index": index_path,
            "labels": labels_path,
            "dataset_filter": args.dataset,
            "split_path": args.split_path,
            "split": args.split,
            "evaluated_samples": total,
            "max_samples": int(args.max_samples),
            "batch_size": int(args.batch_size),
        },
        "metrics": {
            "target_object_accuracy": float(obj_correct / max(total, 1)),
            "phase_accuracy": float(phase_correct / max(total, 1)),
            "schema_sensitivity_score_target": float(blank_changed / max(blank_total, 1)),
            "target_flip_rate_swap": float(swap_flip_correct / max(swap_total, 1)),
        },
        "label_spaces": {
            "target_object": ds.object_vocab,
            "phase": ds.phase_vocab,
        },
        "confusion": {
            "target_object": obj_conf,
            "phase": phase_conf,
        },
        "per_class": {
            "target_object": per_class_stats(obj_conf, ds.object_vocab),
            "phase": per_class_stats(phase_conf, ds.phase_vocab),
        },
        "preview": preview,
        "language_sensitivity": {
            "blank_total": int(blank_total),
            "blank_changed": int(blank_changed),
            "swap_total": int(swap_total),
            "swap_flip_correct": int(swap_flip_correct),
        },
        "qual_cases": qual_cases,
    }

    payload = json.dumps(result, indent=2, ensure_ascii=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(payload)

    ds.close()


if __name__ == "__main__":
    main()
