import argparse
import json
from pathlib import Path

from h2r_schema.adapters.handover_sim import HandoverSimMap, convert_episode_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Handover-Sim style records to unified episode JSONL.")
    parser.add_argument("--input", required=True, help="Input file or directory (.json/.jsonl).")
    parser.add_argument("--output-dir", required=True, help="Output directory for unified episode JSONL files.")
    parser.add_argument(
        "--mapping",
        required=False,
        default="",
        help="Optional mapping JSON path to override source key paths.",
    )
    return parser.parse_args()


def load_mapping(mapping_path: str) -> HandoverSimMap:
    if not mapping_path:
        return HandoverSimMap()
    path = Path(mapping_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Mapping file must be a JSON object")
    return HandoverSimMap.from_dict(raw)


def iter_input_files(path: Path):
    if path.is_file():
        yield path
        return
    for ext in ("*.jsonl", "*.json"):
        for file in path.rglob(ext):
            yield file


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    mapping = load_mapping(args.mapping)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    files = list(iter_input_files(input_path))
    if not files:
        raise ValueError(f"No .json/.jsonl files found under: {input_path}")

    converted = 0
    frames_total = 0
    for file in files:
        rel = file.relative_to(input_path) if input_path.is_dir() else Path(file.name)
        out_path = (output_dir / rel).with_suffix(".jsonl")
        count = convert_episode_file(file, out_path, mapping)
        converted += 1
        frames_total += count
        print(f"Converted {file} -> {out_path} ({count} frames)")

    print(f"Done. Episodes: {converted}, frames: {frames_total}")


if __name__ == "__main__":
    main()

