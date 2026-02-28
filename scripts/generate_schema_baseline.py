import argparse
import json
from pathlib import Path
from typing import Dict, List

from h2r_schema.schema_baseline import baseline_affordances, build_uncertainty, infer_phase, run_checker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline schema JSONL from summaries JSONL.")
    parser.add_argument("--input", required=True, help="Input summaries JSONL path.")
    parser.add_argument("--output", required=True, help="Output schema JSONL path.")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def to_schema_row(summary_row: Dict) -> Dict:
    rel = summary_row.get("relations", {})
    uncertainty = build_uncertainty(rel)
    phase = infer_phase(rel)
    affordances = baseline_affordances(rel, uncertainty)
    schema = {
        "episode_id": summary_row.get("episode_id", ""),
        "timestamp": summary_row.get("timestamp", 0.0),
        "phase": phase,
        "entities": summary_row.get("entities", {}),
        "relations": rel,
        "constraints": [
            {
                "name": "no_retract_before_secure",
                "satisfied": not (phase == "retract" and rel.get("held_by_robot", 0.0) < 0.6),
                "confidence": 0.8,
            }
        ],
        "affordances": affordances,
        "uncertainty": uncertainty,
    }
    return run_checker(schema)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    rows = read_jsonl(in_path)
    out_rows = [to_schema_row(row) for row in rows]
    write_jsonl(out_path, out_rows)
    valid = sum(1 for r in out_rows if r.get("checker", {}).get("valid_schema", False))
    print(f"Built {len(out_rows)} schema rows: {out_path}")
    print(f"Schema validity: {valid}/{len(out_rows)}")


if __name__ == "__main__":
    main()

