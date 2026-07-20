import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from linguistic_blindness.benchmark.perturbations import (
    gold_schema_for,
    libero_instruction_target,
    make_counterfactual_instruction,
    mentioned_objects,
    target_aliases,
)
from linguistic_blindness.utils.io import command_string, write_csv, write_json, write_jsonl

PERTURBATIONS = ["target_swap", "blank_instruction", "impossible_instruction", "safety_conflict", "negation", "phase_conflict"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build offline linguistic-blindness counterfactual benchmark.")
    p.add_argument("--index", default="data/processed/unified_index.parquet")
    p.add_argument("--labels", default="data/processed/schema_labels.parquet")
    p.add_argument("--instr-wrong", default="data/processed/instr_wrong.parquet")
    p.add_argument("--split-path", default="data/processed/episode_splits.json")
    p.add_argument("--split", default="val")
    p.add_argument("--max-observations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", default="outputs/linguistic_blindness/benchmark")
    p.add_argument("--smoke-if-missing", action="store_true", default=True)
    return p.parse_args()


def load_real_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
    except Exception:
        return []
    index_path = Path(args.index)
    labels_path = Path(args.labels)
    if not index_path.exists() or not labels_path.exists():
        return []
    index_df = pd.read_parquet(index_path).copy()
    labels_df = pd.read_parquet(labels_path).copy()
    for df in [index_df, labels_df]:
        df["sample_id"] = df["episode_id"].astype(str) + ":" + df["frame_id"].astype(int).astype(str)
    keep = ["sample_id", "dataset", "episode_id", "frame_id", "instruction", "obs_ptr", "action"]
    keep = [c for c in keep if c in index_df.columns]
    df = index_df[keep].merge(
        labels_df[["sample_id", "target_object", "phase"]], on="sample_id", how="inner"
    )
    if Path(args.instr_wrong).exists():
        instr_df = pd.read_parquet(args.instr_wrong).copy()
        if "sample_id" not in instr_df.columns:
            instr_df["sample_id"] = instr_df["episode_id"].astype(str) + ":" + instr_df["frame_id"].astype(int).astype(str)
        cols = [c for c in ["sample_id", "instr_swap", "swap_valid", "swap_meta"] if c in instr_df.columns]
        df = df.merge(instr_df[cols], on="sample_id", how="left")
    if args.split_path and Path(args.split_path).exists():
        splits = json.loads(Path(args.split_path).read_text(encoding="utf-8")).get("splits", {})
        eps = {str(x) for x in splits.get(args.split, [])}
        if eps:
            df = df[df["episode_id"].astype(str).isin(eps)]
    if df.empty:
        return []
    df = df.sort_values(["dataset", "episode_id", "frame_id"]).reset_index(drop=True)
    return df.to_dict("records")


def scene_objects(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        ep = str(row.get("episode_id", ""))
        target = str(row.get("target_object", "") or "").lower()
        if str(row.get("dataset", "")) == "libero":
            target = libero_instruction_target(str(row.get("instruction", "")), target)
        objects = [target]
        if str(row.get("dataset", "")) == "libero":
            objects.extend(mentioned_objects(str(row.get("instruction", ""))))
            objects.extend(target_aliases(target))
        for obj in objects:
            obj = str(obj or "").lower()
            if obj and obj not in out[ep]:
                out[ep].append(obj)
    return dict(out)


def sample_rows(rows: List[Dict[str, Any]], max_n: int, seed: int) -> List[Dict[str, Any]]:
    if max_n <= 0 or len(rows) <= max_n:
        return rows
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(rows)), max_n))
    return [rows[i] for i in idx]


def smoke_rows() -> List[Dict[str, Any]]:
    return [
        {"sample_id": "smoke:0", "dataset": "smoke_test", "episode_id": "smoke", "frame_id": 0, "instruction": "Pick up the bottle.", "target_object": "bottle", "phase": "reach", "instr_swap": "Pick up the bowl.", "swap_valid": True, "swap_meta": json.dumps({"src": "bottle", "tgt": "bowl"})},
        {"sample_id": "smoke:1", "dataset": "smoke_test", "episode_id": "smoke", "frame_id": 1, "instruction": "Take the object now.", "target_object": "object_0", "phase": "contact", "instr_swap": "Take the bowl now.", "swap_valid": True, "swap_meta": json.dumps({"src": "object_0", "tgt": "bowl"})},
    ]


def build_examples(rows: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    scenes = scene_objects(rows)
    examples: List[Dict[str, Any]] = []
    for row in rows:
        ep = str(row.get("episode_id", ""))
        sample_id = str(row.get("sample_id") or f"{ep}:{row.get('frame_id', 0)}")
        obs_id = sample_id
        scene = scenes.get(ep, []) or [str(row.get("target_object", "object_0")).lower()]
        original_instruction = str(row.get("instruction", "") or row.get("instr", ""))
        dataset = str(row.get("dataset", ""))
        target = str(row.get("target_object", "") or "").lower()
        if dataset == "libero":
            target = libero_instruction_target(original_instruction, target)
            row = {**row, "target_object": target}
        phase = str(row.get("phase", "UNK") or "UNK")
        original_gold = gold_schema_for(target, phase, "original", original_instruction, scene, dataset=dataset)
        for ptype in PERTURBATIONS:
            cf_instruction, cf_target = make_counterfactual_instruction(row, ptype, scene)
            gold = gold_schema_for(target, phase, ptype, cf_instruction, scene, dataset=dataset, cf_target=cf_target)
            examples.append({
                "example_id": f"{sample_id}::{ptype}",
                "observation_id": obs_id,
                "dataset": dataset,
                "source": source,
                "smoke_test": source == "smoke_test",
                "episode_id": ep,
                "frame_id": int(row.get("frame_id", 0) or 0),
                "obs_ptr": row.get("obs_ptr"),
                "expert_action": row.get("action"),
                "scene_objects": scene,
                "original_instruction": original_instruction,
                "counterfactual_instruction": cf_instruction,
                "instruction": cf_instruction,
                "perturbation_type": ptype,
                "target_object": gold.get("target_object"),
                "original_target_object": target,
                "counterfactual_target_object": cf_target,
                "counterfactual_valid": bool(cf_target and cf_target != target and cf_target in scene),
                "original_schema": original_gold,
                "gold_schema": gold,
                "observation_state": {
                    "phase": phase,
                    "human_contact": original_gold.get("human_contact"),
                    "human_released": original_gold.get("human_released"),
                    "robot_contact": original_gold.get("robot_contact"),
                    "robot_grasp_stable": original_gold.get("robot_grasp_stable"),
                },
            })
    return examples


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    rows = load_real_rows(args)
    source = "processed_repo_data"
    if not rows:
        if not args.smoke_if_missing:
            raise FileNotFoundError("No real processed data found and smoke fallback disabled.")
        rows = smoke_rows()
        source = "smoke_test"
    rows = sample_rows(rows, args.max_observations, args.seed)
    examples = build_examples(rows, source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "counterfactual_examples.jsonl", examples)
    counts = Counter(x["perturbation_type"] for x in examples)
    stats = {
        "source": source,
        "smoke_test": source == "smoke_test",
        "num_observations": len(rows),
        "num_counterfactual_examples": len(examples),
        "num_schema_labels": len(examples),
        "perturbation_types": dict(sorted(counts.items())),
        "seed": args.seed,
        "command": command_string(),
    }
    write_json(out_dir / "benchmark_statistics.json", stats)
    write_csv(out_dir / "benchmark_statistics.csv", [{"split": args.split, "domain": source, **stats}])
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
