#!/usr/bin/env python3
"""Build normal-instruction benchmark for Exp4 preservation.

The examples reuse the exact 600 frozen Exp2 observations but set `instruction`
to the original normal instruction and `gold_schema` to the benchmark original
schema. Example ids are kept identical to Exp2 so Qwen normal predictions can be
joined with Exp2 rows directly.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from linguistic_blindness.utils.io import command_string, read_jsonl, utc_timestamp, write_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Exp4 normal-instruction benchmark from frozen Exp2 clean 600.")
    p.add_argument("--base-benchmark", default="outputs/linguistic_blindness/benchmark_val_full_clean/counterfactual_examples.jsonl")
    p.add_argument("--exp2-predictions", default="outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/predictions.jsonl")
    p.add_argument("--out-dir", default="outputs/linguistic_blindness/exp4_normal_instruction_clean_600")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp2_rows = read_jsonl(args.exp2_predictions)
    exp2_ids = [str(r["example_id"]) for r in exp2_rows]
    base_by_id = {str(r.get("example_id")): r for r in read_jsonl(args.base_benchmark)}
    missing = [eid for eid in exp2_ids if eid not in base_by_id]
    if missing:
        raise SystemExit(f"Missing {len(missing)} Exp2 examples in base benchmark; first={missing[0]}")

    rows: List[Dict[str, Any]] = []
    for eid in exp2_ids:
        base = dict(base_by_id[eid])
        if base.get("dataset") != "libero":
            raise SystemExit(f"Non-LIBERO Exp2 source: {eid} dataset={base.get('dataset')}")
        base["instruction"] = base.get("original_instruction")
        base["counterfactual_instruction"] = base.get("original_instruction")
        base["counterfactual_target_object"] = base.get("original_target_object")
        base["target_object"] = base.get("original_target_object")
        base["perturbation_type"] = "normal_instruction"
        base["gold_schema"] = base.get("original_schema")
        base["counterfactual_valid"] = True
        base["exp4_source"] = "exp2_target_swap_clean_600_original_instruction"
        base["source"] = "exp4_normal_instruction_from_exp2_clean_600"
        rows.append(base)

    summary = [{
        "split": "normal_instruction_clean_600",
        "num_examples": len(rows),
        "num_observations": len({r.get("observation_id") for r in rows}),
        "num_original_instructions": len({r.get("original_instruction") for r in rows}),
        "num_schema_labels": sum(1 for r in rows if r.get("gold_schema")),
        "perturbation_types": "normal_instruction",
    }]
    write_jsonl(out_dir / "benchmark.jsonl", rows)
    write_json(out_dir / "benchmark_statistics.json", {"rows": summary})
    write_csv(out_dir / "benchmark_statistics.csv", summary)
    write_json(out_dir / "run_metadata.json", {
        "timestamp_utc": utc_timestamp(),
        "command": command_string(),
        "base_benchmark": args.base_benchmark,
        "exp2_predictions": args.exp2_predictions,
        "num_examples": len(rows),
        "python": sys.version,
        "platform": platform.platform(),
    })
    print(json.dumps({"out_dir": str(out_dir), "num_examples": len(rows), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
