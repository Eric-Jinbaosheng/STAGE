import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline fine-tune skeleton on frozen processed files.")
    p.add_argument("--index", default="data/processed/libero_index.parquet")
    p.add_argument("--labels", default="data/processed/schema_labels.parquet")
    p.add_argument("--out-dir", default="artifacts/finetune_baseline")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--frame-window", type=int, default=1)
    p.add_argument("--views", default="agentview_rgb")
    p.add_argument("--text-vocab-size", type=int, default=2048)
    p.add_argument("--text-max-len", type=int, default=16)
    p.add_argument("--text-vocab-path", default="", help="Optional existing text_vocab.json to reuse.")
    p.add_argument("--text-embed-dim", type=int, default=64)
    p.add_argument("--text-hidden-dim", type=int, default=64)
    p.add_argument("--use-image", action="store_true", help="Load image tensors from obs_ptr when available.")
    p.add_argument(
        "--task",
        choices=["target_object", "phase", "multi"],
        default="multi",
        help="Which label head to optimize.",
    )
    return p.parse_args()


def main() -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError("torch is required for train_finetune_baseline.py") from exc

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.data.dataloader import UnifiedSchemaDataset

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    view_names = [x.strip() for x in args.views.split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = UnifiedSchemaDataset(
        args.index,
        args.labels,
        use_image=args.use_image,
        image_size=args.image_size,
        frame_window=args.frame_window,
        view_names=view_names,
        text_vocab_size=args.text_vocab_size,
        text_max_len=args.text_max_len,
        text_vocab_path=(args.text_vocab_path or None),
    )
    if not args.text_vocab_path:
        ds.export_text_vocab(str(out_dir / "text_vocab.json"))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    feature_dim = 5
    hidden_dim = 128
    object_classes = max(2, len(ds.object_vocab))
    phase_classes = max(2, len(ds.phase_vocab))

    class MultiHeadModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.use_image = bool(args.use_image)
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
            self.text_embedding = torch.nn.Embedding(ds.text_vocab_size, args.text_embed_dim, padding_idx=ds.pad_id)
            self.text_rnn = torch.nn.GRU(
                input_size=args.text_embed_dim,
                hidden_size=args.text_hidden_dim,
                batch_first=True,
            )
            fusion_in = feature_dim + args.text_hidden_dim + img_dim
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

    model = MultiHeadModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = torch.nn.CrossEntropyLoss()

    history = []
    it = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        x_num = batch["x"].float()
        x_img = batch["image"].float()
        input_ids = batch["input_ids"].long()
        attention_mask = batch["attention_mask"].float()
        y_obj = batch["label_target_object"].long()
        y_phase = batch["label_phase"].long()

        logits_obj, logits_phase = model(x_num, x_img, input_ids, attention_mask)
        loss_obj = ce(logits_obj, y_obj)
        loss_phase = ce(logits_phase, y_phase)

        if args.task == "target_object":
            loss = loss_obj
        elif args.task == "phase":
            loss = loss_phase
        else:
            loss = loss_obj + loss_phase

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 50 == 0 or step == args.steps:
            with torch.no_grad():
                obj_acc = float((logits_obj.argmax(dim=1) == y_obj).float().mean().item())
                phase_acc = float((logits_phase.argmax(dim=1) == y_phase).float().mean().item())
            record = {
                "step": step,
                "loss": float(loss.item()),
                "loss_target_object": float(loss_obj.item()),
                "loss_phase": float(loss_phase.item()),
                "acc_target_object": obj_acc,
                "acc_phase": phase_acc,
                "use_image": bool(args.use_image),
                "frame_window": int(args.frame_window),
                "views": view_names,
                "text_vocab_size": int(ds.text_vocab_size),
                "text_max_len": int(ds.text_max_len),
                "text_embed_dim": int(args.text_embed_dim),
                "text_hidden_dim": int(args.text_hidden_dim),
            }
            history.append(record)
            print(record)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "object_vocab": ds.object_vocab,
            "phase_vocab": ds.phase_vocab,
            "args": vars(args),
        },
        out_dir / "checkpoint.pt",
    )
    (out_dir / "metrics.json").write_text(json.dumps(history, indent=2, ensure_ascii=True), encoding="utf-8")
    ds.close()

    print(f"Wrote checkpoint: {out_dir / 'checkpoint.pt'}")
    print(f"Wrote metrics: {out_dir / 'metrics.json'}")
    if not args.text_vocab_path:
        print(f"Wrote text vocab: {out_dir / 'text_vocab.json'}")
    else:
        print(f"Reused text vocab: {args.text_vocab_path}")


if __name__ == "__main__":
    main()
