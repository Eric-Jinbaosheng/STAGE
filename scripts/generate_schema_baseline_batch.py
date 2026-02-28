import argparse
from pathlib import Path

from generate_schema_baseline import read_jsonl, to_schema_row, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch generate baseline schema from summary JSONL files.")
    parser.add_argument("--input-dir", required=True, help="Input summaries directory")
    parser.add_argument("--output-dir", required=True, help="Output schema directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    files = sorted(in_dir.rglob("*.jsonl"))
    if not files:
        raise ValueError(f"No .jsonl files under {in_dir}")

    total_rows = 0
    for file in files:
        rows = read_jsonl(file)
        out_rows = [to_schema_row(row) for row in rows]
        out_path = out_dir / file.relative_to(in_dir)
        write_jsonl(out_path, out_rows)
        total_rows += len(out_rows)
        print(f"Built {out_path} ({len(out_rows)} rows)")
    print(f"Done. Files: {len(files)}, rows: {total_rows}")


if __name__ == "__main__":
    main()

