import argparse
import json
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build episode-level train/val/test splits.")
    p.add_argument("--index", default="data/processed/libero_index.parquet")
    p.add_argument("--out", default="data/processed/episode_splits.json")
    p.add_argument("--dataset", default="", help="Optional dataset filter, e.g. libero")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if total_ratio <= 0:
        raise ValueError("train/val/test ratios must sum to a positive value")

    df = pd.read_parquet(args.index)
    if args.dataset:
        df = df[df["dataset"].astype(str) == args.dataset].reset_index(drop=True)

    if "episode_id" not in df.columns:
        raise KeyError("episode_id column not found in index parquet")

    episodes = sorted(df["episode_id"].astype(str).unique().tolist())
    rng = random.Random(args.seed)
    rng.shuffle(episodes)

    n = len(episodes)
    n_train = int(round(n * (args.train_ratio / total_ratio)))
    n_val = int(round(n * (args.val_ratio / total_ratio)))
    n_train = min(max(n_train, 0), n)
    n_val = min(max(n_val, 0), max(0, n - n_train))
    n_test = max(0, n - n_train - n_val)

    train_eps = episodes[:n_train]
    val_eps = episodes[n_train : n_train + n_val]
    test_eps = episodes[n_train + n_val : n_train + n_val + n_test]

    payload = {
        "config": {
            "index": args.index,
            "dataset_filter": args.dataset,
            "seed": int(args.seed),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(args.test_ratio),
            "num_episodes": n,
        },
        "splits": {
            "train": train_eps,
            "val": val_eps,
            "test": test_eps,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"train={len(train_eps)} val={len(val_eps)} test={len(test_eps)}")


if __name__ == "__main__":
    main()
