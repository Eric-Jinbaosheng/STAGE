from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .common import coerce_float, coerce_vec, get_path_value, read_json, read_jsonl, write_jsonl


@dataclass
class GenH2RMap:
    timestamp: str = "t"
    human_hand_pos: str = "obs.human_hand.pos"
    human_hand_quat: str = "obs.human_hand.quat"
    object_pos: str = "obs.object.pos"
    object_quat: str = "obs.object.quat"
    gripper_pos: str = "obs.robot_gripper.pos"
    gripper_quat: str = "obs.robot_gripper.quat"
    hand_object_distance: str = "signals.d_hand_obj"
    gripper_object_distance: str = "signals.d_gripper_obj"
    hand_velocity: str = "obs.human_hand.vel"
    gripper_velocity: str = "obs.robot_gripper.vel"
    object_velocity: str = "obs.object.vel"
    visibility_score: str = "vision.obj_visibility"
    contact_signal_hand_object: str = "signals.contact_human_obj"
    contact_signal_gripper_object: str = "signals.contact_robot_obj"
    gripper_opening: str = "obs.robot_gripper.opening"

    @staticmethod
    def from_dict(raw: Dict) -> "GenH2RMap":
        obj = GenH2RMap()
        for k, v in raw.items():
            if hasattr(obj, k) and isinstance(v, str):
                setattr(obj, k, v)
        return obj


def _pose(position: List[float], orientation: List[float]) -> Dict:
    return {"position_xyz": position, "orientation_xyzw": orientation}


def convert_frame(raw: Dict, idx: int, mapping: GenH2RMap) -> Dict:
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
        "meta": {"source": "genh2r"},
    }


def convert_episode_file(input_path: Path, output_path: Path, mapping: GenH2RMap) -> int:
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

