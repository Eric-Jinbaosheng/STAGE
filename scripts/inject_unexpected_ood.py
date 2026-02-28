import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject unexpected OOD events into unified episode JSONL.")
    parser.add_argument("--input", required=True, help="Unified input episode JSONL")
    parser.add_argument("--output", required=True, help="Output OOD episode JSONL")
    parser.add_argument(
        "--event",
        required=True,
        choices=["not_releasing", "withdrawal", "occlusion", "ambiguous_contact"],
        help="OOD event type to inject",
    )
    parser.add_argument("--start-ratio", type=float, default=0.55, help="Start position in [0,1]")
    parser.add_argument("--duration-ratio", type=float, default=0.25, help="Event duration ratio in [0,1]")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No frames in {path}")
    return rows


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def event_window(n: int, start_ratio: float, duration_ratio: float) -> tuple[int, int]:
    start = int(max(0.0, min(1.0, start_ratio)) * n)
    dur = max(1, int(max(0.01, min(1.0, duration_ratio)) * n))
    end = min(n, start + dur)
    return start, end


def inject_not_releasing(rows: List[Dict], start: int, end: int) -> None:
    for i in range(start, end):
        rows[i]["contact_signal_hand_object"] = max(rows[i].get("contact_signal_hand_object", 0.0), 0.85)
        rows[i]["hand_object_distance"] = min(rows[i].get("hand_object_distance", 0.1), 0.025)
        rows[i]["meta"] = {**rows[i].get("meta", {}), "ood_event": "not_releasing"}


def inject_withdrawal(rows: List[Dict], start: int, end: int) -> None:
    for i in range(start, end):
        rows[i]["hand_object_distance"] = min(1.2, rows[i].get("hand_object_distance", 0.1) + 0.06)
        hv = rows[i].get("hand_velocity", [0.0, 0.0, 0.0])
        rows[i]["hand_velocity"] = [abs(float(hv[0])) + 0.06, abs(float(hv[1])) + 0.02, abs(float(hv[2])) + 0.01]
        rows[i]["contact_signal_hand_object"] = max(0.0, rows[i].get("contact_signal_hand_object", 0.0) * 0.4)
        rows[i]["meta"] = {**rows[i].get("meta", {}), "ood_event": "withdrawal"}


def inject_occlusion(rows: List[Dict], start: int, end: int) -> None:
    for i in range(start, end):
        rows[i]["visibility_score"] = max(0.0, rows[i].get("visibility_score", 1.0) * 0.2)
        rows[i]["meta"] = {**rows[i].get("meta", {}), "ood_event": "occlusion"}


def inject_ambiguous_contact(rows: List[Dict], start: int, end: int, rng: random.Random) -> None:
    for i in range(start, end):
        rows[i]["gripper_object_distance"] = min(rows[i].get("gripper_object_distance", 0.2), 0.028)
        rows[i]["contact_signal_gripper_object"] = rng.uniform(0.05, 0.3)
        gv = rows[i].get("gripper_velocity", [0.0, 0.0, 0.0])
        rows[i]["gripper_velocity"] = [float(gv[0]) * 0.5, float(gv[1]) * 0.5, float(gv[2]) * 0.5]
        rows[i]["meta"] = {**rows[i].get("meta", {}), "ood_event": "ambiguous_contact"}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    in_path = Path(args.input)
    out_path = Path(args.output)
    rows = load_jsonl(in_path)
    start, end = event_window(len(rows), args.start_ratio, args.duration_ratio)

    if args.event == "not_releasing":
        inject_not_releasing(rows, start, end)
    elif args.event == "withdrawal":
        inject_withdrawal(rows, start, end)
    elif args.event == "occlusion":
        inject_occlusion(rows, start, end)
    elif args.event == "ambiguous_contact":
        inject_ambiguous_contact(rows, start, end, rng)

    save_jsonl(out_path, rows)
    print(f"Injected {args.event} on frames [{start}, {end}) -> {out_path}")


if __name__ == "__main__":
    main()

