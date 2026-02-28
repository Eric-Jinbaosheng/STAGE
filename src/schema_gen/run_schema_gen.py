import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema_check.checker import run_checker
from schema_gen.repair_loop import run_repair_loop


ACTIONS = [
    "HOLD_COMPLIANT",
    "BACKOFF_SMALL",
    "VIEWPOINT_CHANGE",
    "REALIGN",
    "CLOSE_GENTLE",
    "PROMPT_HUMAN_RELEASE",
]


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def tri_from_score(score: float, lo: float = 0.35, hi: float = 0.65):
    if score >= hi:
        return True
    if score <= lo:
        return False
    return "unknown"


def infer_phase(rel: Dict, state: Dict) -> str:
    app_g = float(rel.get("approaching_gripper_object", 0.0))
    aligned = float(rel.get("aligned_gripper_object", 0.0))
    c = state["contact_confirmed"]
    released = state["human_released"]
    secured = state["object_secured"]
    if secured is True and released is True:
        return "secure"
    if c is True and released is False:
        return "transfer"
    if c is True:
        return "contact"
    if aligned > 0.65:
        return "align"
    if app_g > 0.5:
        return "reach"
    return "offer"


def build_affordances(state: Dict, rel: Dict) -> List[Dict]:
    occluded = state["occluded"]
    contact = state["contact_confirmed"]
    released = state["human_released"]
    secured = state["object_secured"]

    base = [
        {
            "name": "HOLD_COMPLIANT",
            "params": {"duration_s": 0.5},
            "preconditions": ["always"],
            "expected_observation": "state stabilized",
            "score": 0.4,
        },
        {
            "name": "BACKOFF_SMALL",
            "params": {"distance_m": 0.03},
            "preconditions": ["risk_or_conflict"],
            "expected_observation": "safer separation",
            "score": 0.2,
        },
        {
            "name": "VIEWPOINT_CHANGE",
            "params": {"delta_deg": 20},
            "preconditions": ["occluded=true"],
            "expected_observation": "visibility improves",
            "score": 0.1,
        },
        {
            "name": "REALIGN",
            "params": {"max_delta_deg": 10},
            "preconditions": ["approach=true", "not_secured"],
            "expected_observation": "alignment increases",
            "score": 0.2,
        },
        {
            "name": "CLOSE_GENTLE",
            "params": {"effort": 0.25},
            "preconditions": ["contact_confirmed=true"],
            "expected_observation": "held_by_robot increases",
            "score": 0.05,
        },
        {
            "name": "PROMPT_HUMAN_RELEASE",
            "params": {"mode": "audio"},
            "preconditions": ["human_released=false"],
            "expected_observation": "release event triggered",
            "score": 0.05,
        },
    ]

    if occluded is True:
        for a in base:
            if a["name"] == "VIEWPOINT_CHANGE":
                a["score"] = 0.95
            if a["name"] == "HOLD_COMPLIANT":
                a["score"] = 0.85
            if a["name"] == "CLOSE_GENTLE":
                a["score"] = 0.0
    elif contact is True and secured is not True:
        for a in base:
            if a["name"] == "CLOSE_GENTLE":
                a["score"] = 0.88
            if a["name"] == "REALIGN":
                a["score"] = 0.45
    elif released is False and contact is True:
        for a in base:
            if a["name"] == "PROMPT_HUMAN_RELEASE":
                a["score"] = 0.9
            if a["name"] == "HOLD_COMPLIANT":
                a["score"] = 0.8
    else:
        for a in base:
            if a["name"] == "REALIGN":
                a["score"] = 0.65
            if a["name"] == "HOLD_COMPLIANT":
                a["score"] = 0.5

    return base


def choose_next_action(affordances: List[Dict]) -> str:
    ranked = sorted(affordances, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return ranked[0]["name"]


def summaries_to_schema(row: Dict, dt: float) -> Dict:
    rel = row.get("relations", {})
    contact = tri_from_score(float(rel.get("contact_confirmed", 0.0)))
    released = tri_from_score(float(rel.get("human_released", 0.0)))
    secured = tri_from_score(float(rel.get("held_by_robot", 0.0)))
    occluded = tri_from_score(float(rel.get("occluded_object", 0.0)))
    state = {
        "contact_confirmed": contact,
        "human_released": released,
        "object_secured": secured,
        "occluded": occluded,
    }
    phase = infer_phase(rel, state)
    affordances = build_affordances(state, rel)
    chosen = choose_next_action(affordances)
    schema = {
        "meta": {
            "episode_id": str(row.get("episode_id", "")),
            "t": float(row.get("timestamp", 0.0)),
            "dt": float(dt),
        },
        "phase": phase,
        "state": state,
        "constraints": [
            "no_retract_before_object_secured",
            "no_close_without_contact_confirmed",
            "prefer_conservative_action_when_occluded",
        ],
        "affordances": affordances,
        "next_action": {
            "chosen": chosen,
            "reason": "highest scored affordance from summaries evidence",
        },
    }
    return schema


def run_generation(
    rows: List[Dict],
    schema_path: str,
    max_retries: int,
) -> List[Dict]:
    out = []
    prev_t = None
    for row in rows:
        t = float(row.get("timestamp", 0.0))
        dt = 0.01 if prev_t is None else max(1e-3, t - prev_t)
        prev_t = t
        first = summaries_to_schema(row, dt=dt)

        first_report = run_checker(first, schema_path)
        repaired, repair_report, repair_count = run_repair_loop(
            first,
            checker_fn=lambda x: run_checker(x, schema_path),
            max_retries=max_retries,
        )
        out.append(
            {
                "schema": repaired,
                "first_pass_valid": bool(first_report["valid"]),
                "final_valid": bool(repair_report["valid"]),
                "repair_count": int(repair_count),
                "first_report": first_report,
                "final_report": repair_report,
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate interaction schema from summaries.")
    p.add_argument("--input", required=True, help="Input summary JSONL file or directory.")
    p.add_argument("--output", required=True, help="Output schema JSONL file.")
    p.add_argument("--schema-spec", default="schema_spec/interaction_schema.json")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--sample-frames", type=int, default=0, help="If >0, sample this many frames.")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def collect_rows(input_path: Path) -> List[Dict]:
    if input_path.is_file():
        return read_jsonl(input_path)
    rows: List[Dict] = []
    for file in sorted(input_path.rglob("*.jsonl")):
        rows.extend(read_jsonl(file))
    return rows


def main() -> None:
    args = parse_args()
    rows = collect_rows(Path(args.input))
    if not rows:
        raise ValueError(f"No summaries found in: {args.input}")

    if args.sample_frames > 0 and args.sample_frames < len(rows):
        rng = random.Random(args.seed)
        rows = rng.sample(rows, args.sample_frames)
        rows = sorted(rows, key=lambda x: (x.get("episode_id", ""), float(x.get("timestamp", 0.0))))

    generated = run_generation(rows, schema_path=args.schema_spec, max_retries=args.max_retries)
    out_rows = []
    for g in generated:
        row = {
            **g["schema"],
            "_metrics": {
                "first_pass_valid": g["first_pass_valid"],
                "final_valid": g["final_valid"],
                "repair_count": g["repair_count"],
            },
            "_checker": g["final_report"],
        }
        out_rows.append(row)
    write_jsonl(Path(args.output), out_rows)

    n = len(generated)
    v1 = sum(1 for g in generated if g["first_pass_valid"])
    v2 = sum(1 for g in generated if g["final_valid"])
    avg_rep = sum(g["repair_count"] for g in generated) / max(1, n)
    print(f"rows={n}")
    print(f"first_pass_validity={v1/n:.4f}")
    print(f"after_repair_validity={v2/n:.4f}")
    print(f"avg_repair_count={avg_rep:.4f}")


if __name__ == "__main__":
    main()
