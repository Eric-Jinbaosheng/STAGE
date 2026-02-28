import argparse
import json
from pathlib import Path
from typing import Dict, List

from common_io import ensure_dir, write_json, write_parquet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build LIBERO frame-level index parquet.")
    p.add_argument("--raw-dir", default="data/libero_raw")
    p.add_argument("--out", default="data/processed/libero_index.parquet")
    p.add_argument("--meta-out", default="data/processed/meta.json")
    p.add_argument(
        "--datasets",
        default="libero_spatial,libero_object,libero_goal",
        help="Comma-separated top-level dataset folders to include.",
    )
    return p.parse_args()


def _rows_from_npz(file: Path) -> List[Dict]:
    import numpy as np

    rows: List[Dict] = []
    data = np.load(file, allow_pickle=True)
    keys = set(data.files)
    n = 0
    if "actions" in keys:
        n = int(len(data["actions"]))
    elif "images" in keys:
        n = int(len(data["images"]))
    elif "obs" in keys:
        n = int(len(data["obs"]))
    instruction = str(data["instruction"]) if "instruction" in keys else "unknown instruction"
    task_id = str(data["task_id"]) if "task_id" in keys else file.stem
    for i in range(n):
        action = data["actions"][i].tolist() if "actions" in keys else []
        rows.append(
            {
                "dataset": "libero",
                "episode_id": file.stem,
                "frame_id": i,
                "t": float(i),
                "instruction": instruction,
                "task_id": task_id,
                "obs_ptr": f"{file.as_posix()}#frame/{i}",
                "proprio": "",
                "action": json.dumps(action, ensure_ascii=True),
                "object_ids": "",
                "object_states": "",
                "camera_name": "unknown",
            }
        )
    return rows


def _rows_from_hdf5(file: Path) -> List[Dict]:
    import h5py

    rows: List[Dict] = []
    with h5py.File(file, "r") as f:
        if "data" not in f:
            return rows
        g = f["data"]
        problem_info_raw = g.attrs.get("problem_info", "{}")
        if isinstance(problem_info_raw, bytes):
            problem_info_raw = problem_info_raw.decode("utf-8")
        try:
            problem_info = json.loads(problem_info_raw)
        except Exception:
            problem_info = {}
        instruction = str(problem_info.get("language_instruction", "unknown instruction"))
        env_name = str(g.attrs.get("env_name", "libero_task"))
        task_id = file.stem

        demo_keys = sorted([k for k in g.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))
        for demo_key in demo_keys:
            demo = g[demo_key]
            n = int(demo["actions"].shape[0]) if "actions" in demo else int(demo["states"].shape[0])
            for i in range(n):
                action = demo["actions"][i].tolist() if "actions" in demo else []
                camera_name = "agentview_rgb" if ("obs" in demo and "agentview_rgb" in demo["obs"]) else "unknown"
                gripper_opening = None
                ee_pos = None
                if "obs" in demo:
                    if "gripper_states" in demo["obs"]:
                        gs = demo["obs"]["gripper_states"][i]
                        gripper_opening = float(abs(float(gs[0])) + abs(float(gs[1])))
                    if "ee_pos" in demo["obs"]:
                        ee_pos = demo["obs"]["ee_pos"][i].tolist()
                rows.append(
                    {
                        "dataset": "libero",
                        "episode_id": f"{file.stem}:{demo_key}",
                        "frame_id": i,
                        "t": float(i),
                        "instruction": instruction,
                        "task_id": task_id,
                        "obs_ptr": f"{file.as_posix()}#/data/{demo_key}/obs/{camera_name}/{i}",
                        "proprio": f"{file.as_posix()}#/data/{demo_key}/robot_states/{i}",
                        "action": json.dumps(action, ensure_ascii=True),
                        "object_ids": "",
                        "object_states": f"{file.as_posix()}#/data/{demo_key}/states/{i}",
                        "camera_name": camera_name,
                        "env_name": env_name,
                        "gripper_opening": gripper_opening if gripper_opening is not None else -1.0,
                        "action_gripper": float(action[-1]) if len(action) > 0 else 0.0,
                        "ee_pos": json.dumps(ee_pos if ee_pos is not None else [], ensure_ascii=True),
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    meta_path = Path(args.meta_out)
    ensure_dir(out_path.parent)
    ensure_dir(raw_dir)
    allowed = {x.strip() for x in args.datasets.split(",") if x.strip()}

    rows: List[Dict] = []
    files = []
    for ext in ("*.npz", "*.h5", "*.hdf5", "*.jsonl"):
        for file in sorted(raw_dir.rglob(ext)):
            top = file.relative_to(raw_dir).parts[0] if len(file.relative_to(raw_dir).parts) > 0 else ""
            if allowed and top not in allowed:
                continue
            files.append(file)

    for file in files:
        if file.suffix.lower() == ".npz":
            rows.extend(_rows_from_npz(file))
        elif file.suffix.lower() in (".h5", ".hdf5"):
            rows.extend(_rows_from_hdf5(file))
        # jsonl kept as extension point.

    write_parquet(rows, out_path)
    meta_patch = {
        "libero_index_rows": len(rows),
        "libero_files_found": len(files),
        "libero_datasets_included": sorted(allowed),
        "libero_index_path": out_path.as_posix(),
    }
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        old.update(meta_patch)
        write_json(meta_path, old)
    else:
        write_json(meta_path, meta_patch)

    if not files:
        print(f"No LIBERO raw files found under {raw_dir}. Wrote empty index: {out_path}")
    else:
        print(f"Wrote {out_path} rows={len(rows)} from files={len(files)}")


if __name__ == "__main__":
    main()
