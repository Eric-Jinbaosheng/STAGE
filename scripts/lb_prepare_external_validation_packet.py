#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sample(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    return rows[: min(n, len(rows))]


def norm_scene_objects(row: Dict[str, Any]) -> str:
    objs = row.get("scene_objects") or row.get("objects") or []
    return json.dumps(objs, ensure_ascii=False)


def sheet_row(row: Dict[str, Any], subset: str, idx: int) -> Dict[str, Any]:
    schema = row.get("gold_schema") or row.get("qwen_schema_counterfactual") or {}
    original_schema = row.get("original_schema") or {}
    target = row.get("counterfactual_target_object") or schema.get("target_object") or row.get("target_object") or ""
    first_subgoal = schema.get("next_action") or ""
    return {
        "annotation_id": f"{subset}_{idx:04d}",
        "subset": subset,
        "example_id": row.get("example_id", ""),
        "observation_id": row.get("observation_id", ""),
        "obs_ptr": row.get("obs_ptr") or row.get("image_source") or "",
        "scene_objects": norm_scene_objects(row),
        "instruction": row.get("counterfactual_instruction") or row.get("instruction") or "",
        "original_instruction": row.get("original_instruction", ""),
        "internal_executable_label": row.get("counterfactual_valid", row.get("target_exists", "")),
        "internal_target": target,
        "internal_first_subgoal": first_subgoal,
        "internal_should_execute": "ALLOW" if schema.get("target_exists", True) and not schema.get("blocked_actions") else "",
        "annotator_id": "",
        "is_instruction_executable": "",
        "intended_target": "",
        "first_required_subgoal": "",
        "robot_decision_execute_ask_hold_abort": "",
        "confidence_1_to_5": "",
        "notes": "",
        "internal_original_target": row.get("original_target_object") or original_schema.get("target_object") or "",
        "satbenchpp_split": row.get("satbenchpp_split", ""),
        "satbenchpp_family": row.get("satbenchpp_family", ""),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["annotation_id"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare external label validation packet for invalid + SAT-Bench++ splits.")
    ap.add_argument("--invalid", default="outputs/linguistic_blindness/exp3_invalid_instruction_clean_600/benchmark.jsonl")
    ap.add_argument("--satbenchpp", default="outputs/linguistic_blindness/satbenchpp_v1/benchmark.jsonl")
    ap.add_argument("--out-dir", default="outputs/linguistic_blindness/external_validation_packet_300")
    ap.add_argument("--n-invalid", type=int, default=100)
    ap.add_argument("--n-compositional", type=int, default=100)
    ap.add_argument("--n-temporal", type=int, default=100)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()

    invalid = read_jsonl(Path(args.invalid))
    satpp = read_jsonl(Path(args.satbenchpp))
    comp = [r for r in satpp if r.get("satbenchpp_split") == "satbenchpp_compositional"]
    temp = [r for r in satpp if r.get("satbenchpp_split") == "satbenchpp_temporal_procedural"]

    rows: List[Dict[str, Any]] = []
    for subset, picked in [
        ("invalid", sample(invalid, args.n_invalid, args.seed)),
        ("compositional", sample(comp, args.n_compositional, args.seed + 1)),
        ("temporal_procedural", sample(temp, args.n_temporal, args.seed + 2)),
    ]:
        rows.extend(sheet_row(r, subset, i) for i, r in enumerate(picked))

    out = Path(args.out_dir)
    write_csv(out / "external_validation_sheet.csv", rows)
    protocol = """# External Validation Protocol

Ask three non-author annotators to independently fill one copy of `external_validation_sheet.csv`.

For each row, annotators should answer:
- `is_instruction_executable`: yes/no/ambiguous
- `intended_target`: the object referred to by the instruction, or none/ambiguous
- `first_required_subgoal`: the first action/subgoal the robot should attempt
- `robot_decision_execute_ask_hold_abort`: execute/ask/hold/abort
- `confidence_1_to_5`: integer confidence

Do not expose the `internal_*` columns while annotators fill the sheet. Keep them
for scoring agreement with internal labels after annotation.
"""
    (out / "annotation_protocol.md").write_text(protocol, encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps({
        "num_rows": len(rows),
        "invalid_rows": len([r for r in rows if r["subset"] == "invalid"]),
        "compositional_rows": len([r for r in rows if r["subset"] == "compositional"]),
        "temporal_procedural_rows": len([r for r in rows if r["subset"] == "temporal_procedural"]),
        "seed": args.seed,
        "note": "Packet only. Agreement requires completed non-author annotation sheets.",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
