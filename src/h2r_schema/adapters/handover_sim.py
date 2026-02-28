from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .common import coerce_float, coerce_vec, get_path_value, read_json, read_jsonl, write_jsonl


@dataclass
class HandoverSimMap:
    timestamp: str = "timestamp"
    human_hand_pos: str = "human.hand.position"
    human_hand_quat: str = "human.hand.orientation"
    object_pos: str = "object.position"
    object_quat: str = "object.orientation"
    gripper_pos: str = "robot.gripper.position"
    gripper_quat: str = "robot.gripper.orientation"
    hand_object_distance: str = "signals.hand_object_distance"
    gripper_object_distance: str = "signals.gripper_object_distance"
    hand_velocity: str = "human.hand.velocity"
    gripper_velocity: str = "robot.gripper.velocity"
    object_velocity: str = "object.velocity"
    visibility_score: str = "vision.visibility_score"
    contact_signal_hand_object: str = "signals.contact_hand_object"
    contact_signal_gripper_object: str = "signals.contact_gripper_object"
    gripper_opening: str = "robot.gripper.opening"

    @staticmethod
    def from_dict(raw: Dict) -> "HandoverSimMap":
        obj = HandoverSimMap()
        for k, v in raw.items():
            if hasattr(obj, k) and isinstance(v, str):
                setattr(obj, k, v)
        return obj


def _pose(position: List[float], orientation: List[float]) -> Dict:
    return {"position_xyz": position, "orientation_xyzw": orientation}


def convert_frame(raw: Dict, idx: int, mapping: HandoverSimMap) -> Dict:
    timestamp = coerce_float(get_path_value(raw, mapping.timestamp), float(idx))

    human_pos = coerce_vec(get_path_value(raw, mapping.human_hand_pos), 3, [0.0, 0.0, 0.0])
    human_quat = coerce_vec(get_path_value(raw, mapping.human_hand_quat), 4, [0.0, 0.0, 0.0, 1.0])
    object_pos = coerce_vec(get_path_value(raw, mapping.object_pos), 3, [0.0, 0.0, 0.0])
    object_quat = coerce_vec(get_path_value(raw, mapping.object_quat), 4, [0.0, 0.0, 0.0, 1.0])
    gripper_pos = coerce_vec(get_path_value(raw, mapping.gripper_pos), 3, [0.0, 0.0, 0.0])
    gripper_quat = coerce_vec(get_path_value(raw, mapping.gripper_quat), 4, [0.0, 0.0, 0.0, 1.0])

    return {
        "timestamp": timestamp,
        "human_hand_pose": _pose(human_pos, human_quat),
        "object_pose": _pose(object_pos, object_quat),
        "gripper_pose": _pose(gripper_pos, gripper_quat),
        "hand_object_distance": coerce_float(get_path_value(raw, mapping.hand_object_distance), 1.0),
        "gripper_object_distance": coerce_float(get_path_value(raw, mapping.gripper_object_distance), 1.0),
        "hand_velocity": coerce_vec(get_path_value(raw, mapping.hand_velocity), 3, [0.0, 0.0, 0.0]),
        "gripper_velocity": coerce_vec(get_path_value(raw, mapping.gripper_velocity), 3, [0.0, 0.0, 0.0]),
        "object_velocity": coerce_vec(get_path_value(raw, mapping.object_velocity), 3, [0.0, 0.0, 0.0]),
        "visibility_score": coerce_float(get_path_value(raw, mapping.visibility_score), 1.0),
        "contact_signal_hand_object": coerce_float(get_path_value(raw, mapping.contact_signal_hand_object), 0.0),
        "contact_signal_gripper_object": coerce_float(get_path_value(raw, mapping.contact_signal_gripper_object), 0.0),
        "gripper_opening": coerce_float(get_path_value(raw, mapping.gripper_opening), 1.0),
        "meta": {"source": "handover-sim"},
    }


def convert_episode_file(input_path: Path, output_path: Path, mapping: HandoverSimMap) -> int:
    if input_path.suffix.lower() == ".jsonl":
        raw_frames = read_jsonl(input_path)
    elif input_path.suffix.lower() == ".json":
        payload = read_json(input_path)
        if isinstance(payload, list):
            raw_frames = payload
        elif isinstance(payload, dict) and isinstance(payload.get("frames"), list):
            raw_frames = payload["frames"]
        else:
            raise ValueError(f"Unsupported JSON structure: {input_path}")
    else:
        raise ValueError(f"Unsupported file extension: {input_path}")

    out_frames = [convert_frame(frame, idx=i, mapping=mapping) for i, frame in enumerate(raw_frames)]
    write_jsonl(output_path, out_frames)
    return len(out_frames)

