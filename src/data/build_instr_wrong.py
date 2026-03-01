import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd

from common_io import write_json, write_parquet
from parse_instruction import extract_object_mentions, load_object_synonyms, parse_target_object, swap_target_in_instruction


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build blank/shuffle/swap counterfactual instructions.")
    p.add_argument("--index", default="data/processed/libero_index.parquet")
    p.add_argument("--out", default="data/processed/instr_wrong.parquet")
    p.add_argument("--meta-out", default="data/processed/meta.json")
    p.add_argument("--object-vocab", default="data/processed/object_vocab.json")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def build_scene_objects(df: pd.DataFrame, object_synonyms: Dict[str, List[str]]) -> Dict[str, List[str]]:
    per_episode: Dict[str, List[str]] = {}
    for ep, sub in df.groupby("episode_id", sort=False):
        mentions: List[str] = []
        for text in sub["instruction"].astype(str).tolist():
            for item in extract_object_mentions(text, object_synonyms):
                if item not in mentions:
                    mentions.append(item)
        per_episode[str(ep)] = mentions
    return per_episode


def build_shuffle(df: pd.DataFrame, seed: int) -> List[str]:
    shuffled = [""] * len(df)
    rng = random.Random(seed)
    for dataset, sub in df.groupby("dataset", sort=False):
        indices = list(sub.index)
        insts = sub["instruction"].astype(str).tolist()
        if len(insts) <= 1:
            for idx, inst in zip(indices, insts):
                shuffled[idx] = inst
            continue
        perm = insts[:]
        rng.shuffle(perm)
        if perm == insts:
            perm = perm[1:] + perm[:1]
        for idx, inst in zip(indices, perm):
            shuffled[idx] = inst
    return shuffled


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.index).copy()
    if df.empty:
        write_parquet([], Path(args.out))
        print(f"Wrote empty {args.out}")
        return

    df["sample_id"] = df["episode_id"].astype(str) + ":" + df["frame_id"].astype(int).astype(str)
    object_synonyms = load_object_synonyms(args.object_vocab)
    per_episode_scene = build_scene_objects(df, object_synonyms)
    shuffled_instr = build_shuffle(df, args.seed)

    rng = random.Random(args.seed)
    rows: List[Dict] = []
    for pos, (_, row) in enumerate(df.iterrows()):
        instruction = str(row.get("instruction", ""))
        dataset = str(row.get("dataset", ""))
        episode_id = str(row.get("episode_id", ""))
        frame_id = int(row.get("frame_id", 0))
        sample_id = str(row["sample_id"])
        src = parse_target_object(instruction, object_synonyms)
        scene_objects = [x for x in per_episode_scene.get(episode_id, []) if x != src]
        swap_valid = len(scene_objects) > 0 and src != "UNK"
        tgt = rng.choice(scene_objects) if swap_valid else ""
        instr_swap = swap_target_in_instruction(instruction, src, tgt, object_synonyms) if swap_valid else instruction
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": dataset,
                "episode_id": episode_id,
                "frame_id": frame_id,
                "instr": instruction,
                "instr_blank": "",
                "instr_shuffle": shuffled_instr[pos],
                "instr_swap": instr_swap,
                "swap_valid": bool(swap_valid),
                "swap_meta": json.dumps(
                    {
                        "src": src,
                        "tgt": tgt,
                        "scene_objects": per_episode_scene.get(episode_id, []),
                    },
                    ensure_ascii=True,
                ),
            }
        )

    out_path = Path(args.out)
    write_parquet(rows, out_path)

    meta_path = Path(args.meta_out)
    meta_patch = {
        "instr_wrong_rows": len(rows),
        "instr_wrong_path": out_path.as_posix(),
        "instr_wrong_seed": int(args.seed),
        "instr_wrong_fields": ["instr_blank", "instr_shuffle", "instr_swap", "swap_valid", "swap_meta"],
    }
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        old.update(meta_patch)
        write_json(meta_path, old)
    else:
        write_json(meta_path, meta_patch)
    print(f"Wrote {out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
