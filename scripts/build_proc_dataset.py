import argparse
import json
from pathlib import Path
from typing import Dict, List

from h2r_schema.adapters.handover_sim import HandoverSimMap, convert_episode_file
from h2r_schema.io_utils import read_jsonl, try_write_parquet, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Week1 proc dataset from raw Handover-Sim exports.")
    parser.add_argument("--raw-dir", required=True, help="Raw episode dir (.json/.jsonl)")
    parser.add_argument("--proc-dir", required=True, help="Proc output dir")
    parser.add_argument("--mapping", default="", help="Optional mapping JSON")
    return parser.parse_args()


def load_mapping(path: str) -> HandoverSimMap:
    if not path:
        return HandoverSimMap()
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return HandoverSimMap.from_dict(raw if isinstance(raw, dict) else {})


def iter_files(root: Path):
    for ext in ("*.jsonl", "*.json"):
        for p in root.rglob(ext):
            yield p


def flatten_unified_row(episode_id: str, row: Dict, prev_t: float | None) -> Dict:
    t = float(row.get("timestamp", 0.0))
    dt = 0.0 if prev_t is None else max(0.0, t - prev_t)
    return {
        "episode_id": episode_id,
        "t": t,
        "dt": dt,
        "hand_pose": row["human_hand_pose"]["position_xyz"] + row["human_hand_pose"]["orientation_xyzw"],
        "object_pose": row["object_pose"]["position_xyz"] + row["object_pose"]["orientation_xyzw"],
        "gripper_pose": row["gripper_pose"]["position_xyz"] + row["gripper_pose"]["orientation_xyzw"],
        "gripper_width": float(row.get("gripper_opening", 0.0)),
        "contact_flag": float(row.get("contact_signal_gripper_object", 0.0)),
        "visibility": float(row.get("visibility_score", 1.0)),
        "success_label": row.get("meta", {}).get("success_label", -1),
    }


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    proc_dir = Path(args.proc_dir)
    proc_episodes_dir = proc_dir / "episodes"
    proc_episodes_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(args.mapping)
    raw_files = sorted(iter_files(raw_dir))
    if not raw_files:
        raise ValueError(f"No raw files under: {raw_dir}")

    all_rows: List[Dict] = []
    lengths: List[int] = []
    dts: List[float] = []

    for raw_file in raw_files:
        episode_id = raw_file.stem
        out_unified = proc_episodes_dir / f"{episode_id}.jsonl"
        n = convert_episode_file(raw_file, out_unified, mapping)
        unified_rows = read_jsonl(out_unified)
        prev_t = None
        for row in unified_rows:
            flat = flatten_unified_row(episode_id, row, prev_t)
            all_rows.append(flat)
            dts.append(float(flat["dt"]))
            prev_t = float(flat["t"])
        lengths.append(n)
        print(f"Converted {raw_file} -> {out_unified} ({n} frames)")

    parquet_path = proc_dir / "episodes.parquet"
    ok, reason = try_write_parquet(all_rows, parquet_path)
    if not ok:
        fallback = proc_dir / "episodes.jsonl"
        write_jsonl(fallback, all_rows)
        print(f"Parquet unavailable ({reason}); wrote fallback JSONL: {fallback}")
        proc_format = "jsonl"
    else:
        print(f"Wrote: {parquet_path}")
        proc_format = "parquet"

    fps_vals = [1.0 / x for x in dts if x > 0]
    meta = {
        "num_episodes": len(lengths),
        "num_frames": len(all_rows),
        "min_episode_len": min(lengths) if lengths else 0,
        "max_episode_len": max(lengths) if lengths else 0,
        "avg_episode_len": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "avg_dt": (sum(x for x in dts if x > 0) / max(1, len([x for x in dts if x > 0]))),
        "avg_fps": (sum(fps_vals) / len(fps_vals)) if fps_vals else 0.0,
        "proc_format": proc_format,
    }
    meta_path = proc_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()

