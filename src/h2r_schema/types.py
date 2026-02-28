from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Pose:
    position_xyz: List[float]
    orientation_xyzw: List[float]


@dataclass
class FrameRecord:
    timestamp: float
    human_hand_pose: Pose
    object_pose: Pose
    gripper_pose: Pose
    hand_object_distance: float
    gripper_object_distance: float
    hand_velocity: List[float]
    gripper_velocity: List[float]
    object_velocity: List[float]
    visibility_score: float
    contact_signal_hand_object: float
    contact_signal_gripper_object: float
    gripper_opening: float
    force_torque: Optional[List[float]] = None
    meta: Optional[Dict] = None


def _require_vec(name: str, value: List[float], expected: int) -> None:
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError(f"{name} must be a list of length {expected}")


def _require_num(name: str, value) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")


def parse_pose(raw: Dict, prefix: str) -> Pose:
    if not isinstance(raw, dict):
        raise ValueError(f"{prefix} pose must be an object")
    position = raw.get("position_xyz")
    orientation = raw.get("orientation_xyzw")
    _require_vec(f"{prefix}.position_xyz", position, 3)
    _require_vec(f"{prefix}.orientation_xyzw", orientation, 4)
    return Pose(position_xyz=[float(x) for x in position], orientation_xyzw=[float(x) for x in orientation])


def parse_frame(raw: Dict) -> FrameRecord:
    required_numeric = [
        "timestamp",
        "hand_object_distance",
        "gripper_object_distance",
        "visibility_score",
        "contact_signal_hand_object",
        "contact_signal_gripper_object",
        "gripper_opening",
    ]
    for key in required_numeric:
        _require_num(key, raw.get(key))

    hand_velocity = raw.get("hand_velocity")
    gripper_velocity = raw.get("gripper_velocity")
    object_velocity = raw.get("object_velocity")
    _require_vec("hand_velocity", hand_velocity, 3)
    _require_vec("gripper_velocity", gripper_velocity, 3)
    _require_vec("object_velocity", object_velocity, 3)

    return FrameRecord(
        timestamp=float(raw["timestamp"]),
        human_hand_pose=parse_pose(raw.get("human_hand_pose"), "human_hand_pose"),
        object_pose=parse_pose(raw.get("object_pose"), "object_pose"),
        gripper_pose=parse_pose(raw.get("gripper_pose"), "gripper_pose"),
        hand_object_distance=float(raw["hand_object_distance"]),
        gripper_object_distance=float(raw["gripper_object_distance"]),
        hand_velocity=[float(x) for x in hand_velocity],
        gripper_velocity=[float(x) for x in gripper_velocity],
        object_velocity=[float(x) for x in object_velocity],
        visibility_score=float(raw["visibility_score"]),
        contact_signal_hand_object=float(raw["contact_signal_hand_object"]),
        contact_signal_gripper_object=float(raw["contact_signal_gripper_object"]),
        gripper_opening=float(raw["gripper_opening"]),
        force_torque=raw.get("force_torque"),
        meta=raw.get("meta", {}),
    )

