import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluate schema validity.")
    parser.add_argument("--input-dir", required=True, help="Schema directory")
    return parser.parse_args()


def eval_file(path: Path):
    total = 0
    valid = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            if row.get("checker", {}).get("valid_schema", False):
                valid += 1
    return total, valid


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    files = sorted(in_dir.rglob("*.jsonl"))
    if not files:
        raise ValueError(f"No .jsonl files under {in_dir}")

    g_total = 0
    g_valid = 0
    for file in files:
        total, valid = eval_file(file)
        g_total += total
        g_valid += valid
        rate = valid / total if total else 0.0
        print(f"{file}: {valid}/{total} ({rate:.4f})")
    print(f"global: {g_valid}/{g_total} ({(g_valid / g_total) if g_total else 0.0:.4f})")


if __name__ == "__main__":
    main()

