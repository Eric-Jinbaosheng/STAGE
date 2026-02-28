import argparse
from pathlib import Path

from h2r_schema.config_utils import load_simple_yaml
from h2r_schema.pipeline import build_summary_records, load_episode_jsonl, save_summary_jsonl
from h2r_schema.relations import RelationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build H2R relation summaries from unified episode JSONL.")
    parser.add_argument("--input", required=True, help="Input episode JSONL path.")
    parser.add_argument("--output", required=True, help="Output summary JSONL path.")
    parser.add_argument("--episode-id", required=False, default="", help="Override episode_id.")
    parser.add_argument("--window", type=int, default=10, help="Temporal window size for relation smoothing.")
    parser.add_argument("--config", default="", help="Optional summaries YAML config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    episode_id = args.episode_id or in_path.stem

    frames = load_episode_jsonl(in_path)
    relation_config = None
    if args.config:
        config_raw = load_simple_yaml(Path(args.config))
        relation_config = RelationConfig.from_dict(config_raw)
    records = build_summary_records(
        episode_id=episode_id,
        frames=frames,
        window=args.window,
        relation_config=relation_config,
    )
    save_summary_jsonl(out_path, records)
    print(f"Built {len(records)} summary frames: {out_path}")


if __name__ == "__main__":
    main()
