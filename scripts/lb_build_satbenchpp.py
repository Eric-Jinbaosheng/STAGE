#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


TASK_ACTIONS = ["MOVE_TO", "APPROACH", "GRASP", "PICK"]


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def schema(target: str, reason: str, next_action: str = "MOVE_TO", first_subgoal: str | None = None) -> Dict[str, Any]:
    s = {
        "target_object": target,
        "target_exists": True,
        "phase": "approach",
        "human_contact": None,
        "human_released": None,
        "robot_contact": None,
        "robot_grasp_stable": None,
        "allowed_actions": TASK_ACTIONS,
        "blocked_actions": [],
        "next_action": next_action,
        "reason": reason,
        "confidence": 1.0,
    }
    if first_subgoal:
        s["first_subgoal"] = first_subgoal
        s["next_symbolic_action"] = first_subgoal
    return s


def relation(pos_a: List[float], pos_b: List[float]) -> str:
    dx = float(pos_a[0]) - float(pos_b[0])
    dy = float(pos_a[1]) - float(pos_b[1])
    dz = float(pos_a[2]) - float(pos_b[2])
    if abs(dz) > 0.06 and abs(dz) > max(abs(dx), abs(dy)) * 0.7:
        return "above" if dz > 0 else "below"
    if abs(dx) >= abs(dy):
        return "right of" if dx > 0 else "left of"
    return "behind" if dy > 0 else "in front of"


def relation_phrase(rel: str) -> str:
    if rel in {"left of", "right of", "in front of"}:
        return f"to the {rel}"
    return rel


def object_action(obj: str) -> str:
    o = obj.lower()
    if "drawer" in o:
        return "open"
    return "pick up"


def first_subgoal_action(obj: str) -> str:
    return "OPEN" if "drawer" in obj.lower() else "MOVE_TO"


def compositional_rows(base_rows: List[Dict[str, Any]], per_family: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    families = ["attribute_target", "relation_only", "attribute_relation_compositional"]
    for fam in families:
        for r in base_rows[:per_family]:
            orig = r["original_target_object"]
            cf = r["counterfactual_target_object"]
            po = r["original_target_pos"]
            pc = r["counterfactual_target_pos"]
            rel_orig = relation(po, pc)
            rel_cf = relation(pc, po)
            if fam == "attribute_target":
                orig_inst = f"{object_action(orig)} the {orig}"
                cf_inst = f"{object_action(cf)} the {cf}"
                anchor = None
            elif fam == "relation_only":
                orig_inst = f"{object_action(orig)} the object {relation_phrase(rel_orig)} the {cf}"
                cf_inst = f"{object_action(cf)} the object {relation_phrase(rel_cf)} the {orig}"
                anchor = orig
            else:
                orig_inst = f"{object_action(orig)} the {orig} that is {relation_phrase(rel_orig)} the {cf}"
                cf_inst = f"{object_action(cf)} the {cf} that is {relation_phrase(rel_cf)} the {orig}"
                anchor = orig
            eid = r["example_id"].replace("::target_swap", f"::satbenchpp_{fam}")
            out.append(make_row(r, eid, "satbenchpp_compositional", fam, orig_inst, cf_inst, cf, orig, anchor))
    return out


def temporal_rows(base_rows: List[Dict[str, Any]], max_rows: int) -> List[Dict[str, Any]]:
    patterns = ["first_target_order_swap", "open_or_pick_before_other", "move_before_manipulate"]
    out: List[Dict[str, Any]] = []
    for idx, r in enumerate(base_rows[:max_rows]):
        pat = patterns[idx % len(patterns)]
        orig = r["original_target_object"]
        cf = r["counterfactual_target_object"]
        if pat == "first_target_order_swap":
            orig_inst = f"first move to the {orig}, then move to the {cf}"
            cf_inst = f"first move to the {cf}, then move to the {orig}"
        elif pat == "open_or_pick_before_other":
            orig_inst = f"first {object_action(orig)} the {orig}, then {object_action(cf)} the {cf}"
            cf_inst = f"first {object_action(cf)} the {cf}, then {object_action(orig)} the {orig}"
        else:
            orig_inst = f"move to the {orig} before interacting with the {cf}"
            cf_inst = f"move to the {cf} before interacting with the {orig}"
        eid = r["example_id"].replace("::target_swap", f"::satbenchpp_temporal_{pat}")
        row = make_row(r, eid, "satbenchpp_temporal_procedural", pat, orig_inst, cf_inst, cf, orig, None)
        row["first_subgoal_original"] = {"target_object": orig, "next_symbolic_action": first_subgoal_action(orig)}
        row["first_subgoal_counterfactual"] = {"target_object": cf, "next_symbolic_action": first_subgoal_action(cf)}
        row["gold_schema"] = schema(
            cf,
            "gold temporal/procedural schema; counterfactual first subgoal is the action boundary target",
            first_subgoal_action(cf),
            first_subgoal_action(cf),
        )
        row["original_schema"] = schema(
            orig,
            "gold temporal/procedural schema; original first subgoal",
            first_subgoal_action(orig),
            first_subgoal_action(orig),
        )
        out.append(row)
    return out


def make_row(
    r: Dict[str, Any],
    eid: str,
    split: str,
    family: str,
    orig_inst: str,
    cf_inst: str,
    cf_target: str,
    orig_target: str,
    anchor: str | None,
) -> Dict[str, Any]:
    obs = dict(r.get("observation_state") or {})
    obs.update(
        {
            "ee_pos": r.get("ee_pos"),
            "original_target_pos": r.get("original_target_pos"),
            "counterfactual_target_pos": r.get("counterfactual_target_pos"),
            "target_angle_deg": r.get("target_angle_deg"),
            "target_distance": r.get("target_distance"),
        }
    )
    return {
        "example_id": eid,
        "source_example_id": r.get("example_id"),
        "observation_id": r.get("observation_id"),
        "obs_ptr": r.get("image_source") or r.get("obs_ptr"),
        "dataset": "libero",
        "perturbation_type": "target_swap",
        "satbenchpp_split": split,
        "satbenchpp_family": family,
        "original_instruction": orig_inst,
        "counterfactual_instruction": cf_inst,
        "instruction": cf_inst,
        "original_target_object": orig_target,
        "counterfactual_target_object": cf_target,
        "target_object": cf_target,
        "anchor_object": anchor,
        "counterfactual_valid": True,
        "scene_objects": sorted({orig_target, cf_target}),
        "observation_state": obs,
        "gold_schema": schema(cf_target, f"gold SAT-Bench++ schema for {split}/{family}"),
        "original_schema": schema(orig_target, f"gold SAT-Bench++ original schema for {split}/{family}"),
        "target_distance": r.get("target_distance"),
        "target_angle_deg": r.get("target_angle_deg"),
        "smoke_test": False,
    }


def oracle_prediction(row: Dict[str, Any], which: str) -> Dict[str, Any]:
    sch = row["gold_schema"] if which == "counterfactual" else row["original_schema"]
    return {
        "example_id": row["example_id"],
        "method": f"oracle_satbenchpp_{which}_schema",
        "parsed_schema": sch,
        "raw_model_output": json.dumps(sch, sort_keys=True),
        "target_correct": True,
        "satbenchpp_oracle": True,
        "satbenchpp_split": row["satbenchpp_split"],
        "satbenchpp_family": row["satbenchpp_family"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default="outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/analysis/predictions_with_positions.jsonl")
    ap.add_argument("--out-dir", default="outputs/linguistic_blindness/satbenchpp_v1")
    ap.add_argument("--per-compositional-family", type=int, default=200)
    ap.add_argument("--temporal-n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=29)
    args = ap.parse_args()

    rows = [
        r
        for r in read_jsonl(args.positions)
        if r.get("position_extraction_status") == "ok"
        and r.get("original_target_pos") is not None
        and r.get("counterfactual_target_pos") is not None
    ]
    random.Random(args.seed).shuffle(rows)
    needed = max(args.per_compositional_family, args.temporal_n)
    if len(rows) < needed:
        raise ValueError(f"Need {needed} positioned rows, found {len(rows)}")

    bench = compositional_rows(rows, args.per_compositional_family) + temporal_rows(rows, args.temporal_n)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "benchmark.jsonl", bench)
    write_jsonl(out / "oracle_schema_counterfactual.jsonl", [oracle_prediction(r, "counterfactual") for r in bench])
    write_jsonl(out / "oracle_schema_original.jsonl", [oracle_prediction(r, "original") for r in bench])
    stats = {
        "num_examples": len(bench),
        "source_positions": args.positions,
        "counts_by_split": dict(Counter(r["satbenchpp_split"] for r in bench)),
        "counts_by_family": dict(Counter(r["satbenchpp_family"] for r in bench)),
        "note": "Fixed-observation SAT-Bench++ benchmark. Observation and robot state are held fixed; only instruction semantics and the gold action-boundary schema target/subgoal change.",
    }
    (out / "benchmark_statistics.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out_dir": str(out), **stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
