import json
from pathlib import Path
from typing import Dict, List, Optional

from .relations import RelationConfig, compute_relations
from .types import FrameRecord, parse_frame


def load_episode_jsonl(path: Path) -> List[FrameRecord]:
    frames: List[FrameRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                frame = parse_frame(raw)
                frames.append(frame)
            except Exception as exc:
                raise ValueError(f"Failed parsing line {i} in {path}: {exc}") from exc
    if not frames:
        raise ValueError(f"No valid frames found in {path}")
    return frames


def build_summary_records(
    episode_id: str,
    frames: List[FrameRecord],
    window: int = 10,
    relation_config: Optional[RelationConfig] = None,
) -> List[Dict]:
    relation_scores = compute_relations(frames, window=window, config=relation_config)
    output: List[Dict] = []
    for frame, rel in zip(frames, relation_scores):
        output.append(
            {
                "episode_id": episode_id,
                "timestamp": frame.timestamp,
                "entities": {
                    "human_hand_pose": {
                        "position_xyz": frame.human_hand_pose.position_xyz,
                        "orientation_xyzw": frame.human_hand_pose.orientation_xyzw,
                    },
                    "object_pose": {
                        "position_xyz": frame.object_pose.position_xyz,
                        "orientation_xyzw": frame.object_pose.orientation_xyzw,
                    },
                    "gripper_pose": {
                        "position_xyz": frame.gripper_pose.position_xyz,
                        "orientation_xyzw": frame.gripper_pose.orientation_xyzw,
                    },
                },
                "relations": rel,
                "signals": {
                    "d_hand_obj": frame.hand_object_distance,
                    "d_grip_obj": frame.gripper_object_distance,
                    "visibility_score": frame.visibility_score,
                    "contact_signal_hand_object": frame.contact_signal_hand_object,
                    "contact_signal_gripper_object": frame.contact_signal_gripper_object,
                    "gripper_opening": frame.gripper_opening,
                },
                "meta": frame.meta or {},
            }
        )
    return output


def save_summary_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
