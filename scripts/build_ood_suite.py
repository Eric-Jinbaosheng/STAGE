import argparse
from pathlib import Path
from typing import List

from inject_unexpected_ood import (
    event_window,
    inject_ambiguous_contact,
    inject_not_releasing,
    inject_occlusion,
    inject_withdrawal,
    load_jsonl,
    save_jsonl,
)
import random


EVENTS: List[str] = ["not_releasing", "withdrawal", "occlusion", "ambiguous_contact"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 4-event OOD suite from unified episode JSONL files.")
    parser.add_argument("--input-dir", required=True, help="Directory with unified .jsonl episodes")
    parser.add_argument("--output-dir", required=True, help="Directory to write OOD variants")
    parser.add_argument("--start-ratio", type=float, default=0.55)
    parser.add_argument("--duration-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def apply_event(event: str, rows, start: int, end: int, rng: random.Random) -> None:
    if event == "not_releasing":
        inject_not_releasing(rows, start, end)
    elif event == "withdrawal":
        inject_withdrawal(rows, start, end)
    elif event == "occlusion":
        inject_occlusion(rows, start, end)
    elif event == "ambiguous_contact":
        inject_ambiguous_contact(rows, start, end, rng)


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    files = sorted(in_dir.rglob("*.jsonl"))
    if not files:
        raise ValueError(f"No .jsonl files under {in_dir}")

    rng = random.Random(args.seed)
    total = 0
    for file in files:
        rel = file.relative_to(in_dir)
        raw = load_jsonl(file)
        start, end = event_window(len(raw), args.start_ratio, args.duration_ratio)
        for event in EVENTS:
            rows = [dict(r) for r in raw]
            apply_event(event, rows, start, end, rng)
            out_path = out_dir / event / rel
            save_jsonl(out_path, rows)
            total += 1
            print(f"Wrote {out_path}")
    print(f"Done. Generated {total} OOD episodes")


if __name__ == "__main__":
    main()

