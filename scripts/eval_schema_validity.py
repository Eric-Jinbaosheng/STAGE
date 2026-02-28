import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate schema validity from schema JSONL.")
    parser.add_argument("--input", required=True, help="Schema JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    total = 0
    valid = 0
    violations = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            checker = row.get("checker", {})
            if checker.get("valid_schema", False):
                valid += 1
            for v in checker.get("violations", []):
                violations[v] = violations.get(v, 0) + 1

    print(f"total={total}")
    print(f"valid={valid}")
    print(f"valid_rate={valid / total if total else 0.0:.4f}")
    if violations:
        print("violations:")
        for k in sorted(violations):
            print(f"  {k}: {violations[k]}")


if __name__ == "__main__":
    main()

