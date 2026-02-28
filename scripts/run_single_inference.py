import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run single-sample inference from a trained baseline checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    p.add_argument("--text-vocab", required=True, help="Path to text_vocab.json")
    p.add_argument("--index", default="", help="Optional override for index parquet")
    p.add_argument("--labels", default="", help="Optional override for labels parquet")
    p.add_argument("--episode-id", default="", help="Episode id to select")
    p.add_argument("--frame-id", type=int, default=-1, help="Frame id to select with --episode-id")
    p.add_argument("--row-idx", type=int, default=-1, help="Fallback: direct row index after merge")
    p.add_argument("--out", default="", help="Optional JSON output path")
    return p.parse_args()


def main() -> None:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for run_single_inference.py") from exc

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

    model = MultiHeadModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if args.episode_id and args.frame_id >= 0:
        sub = ds.df[(ds.df["episode_id"] == args.episode_id) & (ds.df["frame_id"] == args.frame_id)]
        if len(sub) == 0:
            raise ValueError("Requested episode_id/frame_id not found in merged dataset")
        row_idx = int(sub.index[0])
        iloc_idx = int(ds.df.index.get_loc(row_idx))
    elif args.row_idx >= 0:
        iloc_idx = int(args.row_idx)
    else:
        iloc_idx = 0

    item = ds[iloc_idx]
    row = ds.df.iloc[iloc_idx]

    x_num = item["x"].unsqueeze(0).float()
    x_img = item["image"].unsqueeze(0).float()
    input_ids = item["input_ids"].unsqueeze(0).long()
    attention_mask = item["attention_mask"].unsqueeze(0).float()

    with torch.no_grad():
        logits_obj, logits_phase = model(x_num, x_img, input_ids, attention_mask)
        prob_obj = torch.softmax(logits_obj, dim=1)[0]
        prob_phase = torch.softmax(logits_phase, dim=1)[0]

    pred_obj_id = int(torch.argmax(prob_obj).item())
    pred_phase_id = int(torch.argmax(prob_phase).item())

    result = {
        "sample": {
            "dataset": str(row.get("dataset", "")),
            "episode_id": str(row.get("episode_id", "")),
            "frame_id": int(row.get("frame_id", 0)),
            "instruction": str(row.get("instruction", "")),
            "obs_ptr": str(row.get("obs_ptr", "")),
        },
        "prediction": {
            "pred_target_object": str(ds.object_vocab[pred_obj_id]),
            "pred_phase": str(ds.phase_vocab[pred_phase_id]),
            "target_object_topk": [
                {"label": ds.object_vocab[int(i)], "score": float(prob_obj[int(i)].item())}
                for i in torch.argsort(prob_obj, descending=True)[:3]
            ],
            "phase_topk": [
                {"label": ds.phase_vocab[int(i)], "score": float(prob_phase[int(i)].item())}
                for i in torch.argsort(prob_phase, descending=True)[:3]
            ],
        },
        "ground_truth": {
            "target_object": str(row.get("target_object", "")),
            "phase": str(row.get("phase", "")),
        },
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=True))

    ds.close()


if __name__ == "__main__":
    main()
