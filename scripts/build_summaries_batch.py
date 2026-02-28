import argparse
from pathlib import Path

from h2r_schema.config_utils import load_simple_yaml
from h2r_schema.pipeline import build_summary_records, load_episode_jsonl, save_summary_jsonl
from h2r_schema.relations import RelationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch build relation summaries from unified JSONL episodes.")
    parser.add_argument("--input-dir", required=True, help="Directory with unified *.jsonl episode files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write summary *.jsonl files.")
    parser.add_argument("--window", type=int, default=10, help="Temporal window size.")
    parser.add_argument("--config", default="", help="Optional summaries YAML config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    relation_config = None
    if args.config:
        relation_config = RelationConfig.from_dict(load_simple_yaml(Path(args.config)))
    files = sorted(input_dir.rglob("*.jsonl"))
    if not files:
        raise ValueError(f"No .jsonl files found under: {input_dir}")

    total = 0
    frames_total = 0
    for file in files:
        episode_id = file.stem
        frames = load_episode_jsonl(file)
        records = build_summary_records(
            episode_id=episode_id,
            frames=frames,
            window=args.window,
            relation_config=relation_config,
        )
        out_path = output_dir / file.relative_to(input_dir)
        save_summary_jsonl(out_path, records)
        total += 1
        frames_total += len(records)
        print(f"Built {out_path} ({len(records)} frames)")

    print(f"Done. Episodes: {total}, frames: {frames_total}")


if __name__ == "__main__":
    main()
