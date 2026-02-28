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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = UnifiedSchemaDataset(args.index, args.labels)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    feature_dim = 5
    hidden_dim = 64
    object_classes = max(2, len(ds.object_vocab))
    phase_classes = max(2, len(ds.phase_vocab))

    class MultiHeadModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )
            self.object_head = torch.nn.Linear(hidden_dim, object_classes)
            self.phase_head = torch.nn.Linear(hidden_dim, phase_classes)

        def forward(self, x):
            z = self.encoder(x)
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

        x = batch["x"].float()
        y_obj = batch["label_target_object"].long()
        y_phase = batch["label_phase"].long()

        logits_obj, logits_phase = model(x)
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

    print(f"Wrote checkpoint: {out_dir / 'checkpoint.pt'}")
    print(f"Wrote metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
