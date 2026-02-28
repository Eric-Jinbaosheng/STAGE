import argparse
import json
from pathlib import Path
from typing import Dict, List

from h2r_schema.config_utils import load_simple_yaml
from h2r_schema.io_utils import read_jsonl, try_write_parquet, write_jsonl
from h2r_schema.pipeline import build_summary_records, load_episode_jsonl, save_summary_jsonl
from h2r_schema.relations import RelationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Week1 summaries dataset from proc episode JSONL files.")
    parser.add_argument("--proc-episodes-dir", required=True, help="Directory with proc episode JSONL")
    parser.add_argument("--summaries-dir", required=True, help="Output summaries directory")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--config", default="", help="Optional summaries YAML")
    return parser.parse_args()


def flatten_summary_row(row: Dict) -> Dict:
    rel = row["relations"]
    sig = row.get("signals", {})
    return {
        "episode_id": row["episode_id"],
        "t": float(row["timestamp"]),
        "d_hand_obj": float(sig.get("d_hand_obj", 0.0)),
        "d_grip_obj": float(sig.get("d_grip_obj", 0.0)),
        "approaching_hand_obj": float(rel.get("approaching_hand_object", 0.0)),
        "approaching_grip_obj": float(rel.get("approaching_gripper_object", 0.0)),
        "aligned_grip_obj": float(rel.get("aligned_gripper_object", 0.0)),
        "contact_possible": float(rel.get("contact_possible", 0.0)),
        "contact_confirmed": float(rel.get("contact_confirmed", 0.0)),
        "held_by_human": float(rel.get("held_by_human", 0.0)),
        "held_by_robot": float(rel.get("held_by_robot", 0.0)),
        "human_released": float(rel.get("human_released", 0.0)),
        "occluded_obj": float(rel.get("occluded_object", 0.0)),
    }


def main() -> None:
    args = parse_args()
    proc_episodes_dir = Path(args.proc_episodes_dir)
    summaries_dir = Path(args.summaries_dir)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(proc_episodes_dir.rglob("*.jsonl"))
    if not files:
        raise ValueError(f"No proc episode files under: {proc_episodes_dir}")

    relation_config = None
    if args.config:
        relation_config = RelationConfig.from_dict(load_simple_yaml(Path(args.config)))

    all_rows: List[Dict] = []
    lengths: List[int] = []
    for file in files:
        episode_id = file.stem
        frames = load_episode_jsonl(file)
        records = build_summary_records(
            episode_id=episode_id,
            frames=frames,
            window=args.window,
            relation_config=relation_config,
        )
        out_file = summaries_dir / f"{episode_id}.jsonl"
        save_summary_jsonl(out_file, records)
        flat = [flatten_summary_row(x) for x in records]
        all_rows.extend(flat)
        lengths.append(len(records))
        print(f"Built {out_file} ({len(records)} rows)")

    parquet_path = summaries_dir / "summaries.parquet"
    ok, reason = try_write_parquet(all_rows, parquet_path)
    if not ok:
        fallback = summaries_dir / "summaries.jsonl"
        write_jsonl(fallback, all_rows)
        print(f"Parquet unavailable ({reason}); wrote fallback JSONL: {fallback}")
        fmt = "jsonl"
    else:
        print(f"Wrote: {parquet_path}")
        fmt = "parquet"

    meta = {
        "num_episodes": len(lengths),
        "num_rows": len(all_rows),
        "window": args.window,
        "format": fmt,
        "min_episode_len": min(lengths) if lengths else 0,
        "max_episode_len": max(lengths) if lengths else 0,
        "avg_episode_len": (sum(lengths) / len(lengths)) if lengths else 0.0,
    }
    meta_path = summaries_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
