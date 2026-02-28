import argparse
import json
from pathlib import Path
from typing import Dict, List

from common_io import ensure_dir, write_json, write_parquet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build handover index parquet from raw episode JSON files.")
    p.add_argument("--raw-dir", default="data/raw/handoversim")
    p.add_argument("--out", default="data/processed/handover_index.parquet")
    p.add_argument("--meta-out", default="data/processed/meta.json")
    return p.parse_args()


def pose7(pose_obj: Dict) -> List[float]:
    return [*pose_obj["position"], *pose_obj["orientation"]]


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    meta_path = Path(args.meta_out)
    ensure_dir(out_path.parent)

    rows: List[Dict] = []
    files = sorted(raw_dir.glob("*.json"))
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        frames = payload.get("frames", [])
        for i, fr in enumerate(frames):
            rows.append(
                {
                    "dataset": "handover_sim",
                    "episode_id": fr.get("episode_id", file.stem),
                    "frame_id": i,
                    "t": float(fr.get("timestamp", i * 0.01)),
                    "instruction": "handover object to robot",
                    "obs_ptr": f"{file.as_posix()}#frames/{i}",
                    "proprio": "",
                    "action": "",
                    "object_ids": "object_0",
                    "object_states": "",
                    "camera_name": "sim_default",
                    "hand_pose": pose7(fr["human"]["hand"]),
                    "object_pose": pose7(fr["object"]),
                    "gripper_pose": pose7(fr["robot"]["gripper"]),
                    "gripper_width": float(fr["robot"]["gripper"]["opening"]),
                    "contact_flag": float(fr["signals"]["contact_gripper_object"]),
                    "visibility": float(fr["vision"].get("visibility_score", 1.0)),
                    "d_hand_obj": float(fr["signals"]["hand_object_distance"]),
                    "d_grip_obj": float(fr["signals"]["gripper_object_distance"]),
                }
            )

    write_parquet(rows, out_path)
    meta_patch = {
        "handover_index_rows": len(rows),
        "handover_episodes": len(files),
        "handover_index_path": out_path.as_posix(),
    }
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        old.update(meta_patch)
        write_json(meta_path, old)
    else:
        write_json(meta_path, meta_patch)
    print(f"Wrote {out_path} rows={len(rows)} episodes={len(files)}")


if __name__ == "__main__":
    main()

