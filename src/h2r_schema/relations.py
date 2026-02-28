from dataclasses import dataclass
from typing import Dict, List, Optional

from .math_utils import clamp01, cosine_similarity, mean, safe_norm, sigmoid, sub
from .types import FrameRecord


@dataclass
class RelationConfig:
    approach_dist_scale: float = 0.06
    approach_speed_scale: float = 0.18
    aligned_angle_scale: float = 0.45
    contact_distance_threshold: float = 0.025
    contact_soft_distance_threshold: float = 0.05
    low_rel_speed_threshold: float = 0.08
    rigid_position_threshold: float = 0.01
    gripper_closed_threshold: float = 0.35
    release_drop_threshold: float = 0.35
    occlusion_threshold: float = 0.35
    contact_signal_threshold: float = 0.5

    @staticmethod
    def from_dict(raw: Dict[str, float]) -> "RelationConfig":
        cfg = RelationConfig()
        for key, value in raw.items():
            if hasattr(cfg, key):
                setattr(cfg, key, float(value))
        return cfg


def _orientation_alignment_score(a: List[float], b: List[float]) -> float:
    # Use absolute quaternion dot as a practical similarity proxy.
    qdot = abs(sum(x * y for x, y in zip(a, b)))
    return clamp01(qdot)


def _approaching_score(
    prev_dist: Optional[float], cur_dist: float, velocity: List[float], toward_vec: List[float], cfg: RelationConfig
) -> float:
    if prev_dist is None:
        return 0.0
    dist_drop = prev_dist - cur_dist
    dist_term = sigmoid(dist_drop / max(cfg.approach_dist_scale, 1e-6))
    direction_term = (cosine_similarity(velocity, toward_vec) + 1.0) / 2.0
    speed_term = clamp01(safe_norm(velocity) / max(cfg.approach_speed_scale, 1e-6))
    return clamp01(0.4 * dist_term + 0.3 * direction_term + 0.3 * speed_term)


def _contact_possible_score(dist: float, rel_speed: float, cfg: RelationConfig) -> float:
    near = clamp01((cfg.contact_soft_distance_threshold - dist) / max(cfg.contact_soft_distance_threshold, 1e-6))
    slow = clamp01((cfg.low_rel_speed_threshold - rel_speed) / max(cfg.low_rel_speed_threshold, 1e-6))
    return clamp01(0.7 * near + 0.3 * slow)


def _contact_confirmed_score(signal: float, dist: float, rel_speed: float, cfg: RelationConfig) -> float:
    signal_term = 1.0 if signal >= cfg.contact_signal_threshold else signal
    geom_term = 1.0 if dist <= cfg.contact_distance_threshold else clamp01(
        (cfg.contact_soft_distance_threshold - dist) / max(cfg.contact_soft_distance_threshold, 1e-6)
    )
    impact_term = clamp01(rel_speed / max(cfg.low_rel_speed_threshold, 1e-6))
    return clamp01(max(signal_term, 0.7 * geom_term + 0.3 * impact_term))


def _held_score(rigid_delta_pos: float, contact_score: float, clamp_by_gripper: Optional[float], cfg: RelationConfig) -> float:
    rigid = clamp01((cfg.rigid_position_threshold - rigid_delta_pos) / max(cfg.rigid_position_threshold, 1e-6))
    base = clamp01(0.6 * rigid + 0.4 * contact_score)
    if clamp_by_gripper is None:
        return base
    closed = 1.0 if clamp_by_gripper <= cfg.gripper_closed_threshold else 0.0
    return clamp01(0.7 * base + 0.3 * closed)


def compute_relations(
    frames: List[FrameRecord],
    window: int = 10,
    config: Optional[RelationConfig] = None,
) -> List[Dict[str, float]]:
    cfg = config or RelationConfig()
    scores: List[Dict[str, float]] = []

    prev_hand_dist: Optional[float] = None
    prev_gripper_dist: Optional[float] = None
    prev_held_by_human = 0.0
    prev_hand_obj_rel_pos: Optional[List[float]] = None
    prev_gripper_obj_rel_pos: Optional[List[float]] = None

    held_human_hist: List[float] = []
    held_robot_hist: List[float] = []

    for frame in frames:
        hand_to_obj = sub(frame.object_pose.position_xyz, frame.human_hand_pose.position_xyz)
        gripper_to_obj = sub(frame.object_pose.position_xyz, frame.gripper_pose.position_xyz)
        hand_obj_rel_speed = safe_norm(sub(frame.hand_velocity, frame.object_velocity))
        gripper_obj_rel_speed = safe_norm(sub(frame.gripper_velocity, frame.object_velocity))

        approaching_hand = _approaching_score(
            prev_dist=prev_hand_dist,
            cur_dist=frame.hand_object_distance,
            velocity=frame.hand_velocity,
            toward_vec=hand_to_obj,
            cfg=cfg,
        )
        approaching_gripper = _approaching_score(
            prev_dist=prev_gripper_dist,
            cur_dist=frame.gripper_object_distance,
            velocity=frame.gripper_velocity,
            toward_vec=gripper_to_obj,
            cfg=cfg,
        )

        orientation_score = _orientation_alignment_score(
            frame.gripper_pose.orientation_xyzw, frame.object_pose.orientation_xyzw
        )
        align_dist_term = clamp01((cfg.contact_soft_distance_threshold - frame.gripper_object_distance) / max(
            cfg.contact_soft_distance_threshold, 1e-6
        ))
        aligned = clamp01(0.75 * orientation_score + 0.25 * align_dist_term)

        contact_possible = _contact_possible_score(
            dist=frame.gripper_object_distance, rel_speed=gripper_obj_rel_speed, cfg=cfg
        )
        contact_confirmed = _contact_confirmed_score(
            signal=frame.contact_signal_gripper_object,
            dist=frame.gripper_object_distance,
            rel_speed=gripper_obj_rel_speed,
            cfg=cfg,
        )

        hand_obj_rel_pos = sub(frame.human_hand_pose.position_xyz, frame.object_pose.position_xyz)
        gripper_obj_rel_pos = sub(frame.gripper_pose.position_xyz, frame.object_pose.position_xyz)
        hand_rigid_delta = safe_norm(sub(hand_obj_rel_pos, prev_hand_obj_rel_pos)) if prev_hand_obj_rel_pos else 1.0
        gripper_rigid_delta = safe_norm(sub(gripper_obj_rel_pos, prev_gripper_obj_rel_pos)) if prev_gripper_obj_rel_pos else 1.0

        held_by_human = _held_score(
            rigid_delta_pos=hand_rigid_delta,
            contact_score=frame.contact_signal_hand_object,
            clamp_by_gripper=None,
            cfg=cfg,
        )
        held_by_robot = _held_score(
            rigid_delta_pos=gripper_rigid_delta,
            contact_score=contact_confirmed,
            clamp_by_gripper=frame.gripper_opening,
            cfg=cfg,
        )

        held_human_hist.append(held_by_human)
        held_robot_hist.append(held_by_robot)
        if len(held_human_hist) > window:
            held_human_hist.pop(0)
        if len(held_robot_hist) > window:
            held_robot_hist.pop(0)

        human_released = 0.0
        held_drop = prev_held_by_human - held_by_human
        if held_drop > 0:
            release_by_drop = clamp01(held_drop / max(cfg.release_drop_threshold, 1e-6))
            release_by_history = clamp01(mean(held_human_hist) - held_by_human + 0.5)
            human_released = clamp01(0.7 * release_by_drop + 0.3 * release_by_history)

        occluded = clamp01((cfg.occlusion_threshold - frame.visibility_score) / max(cfg.occlusion_threshold, 1e-6))

        scores.append(
            {
                "approaching_hand_object": approaching_hand,
                "approaching_gripper_object": approaching_gripper,
                "aligned_gripper_object": aligned,
                "contact_possible": contact_possible,
                "contact_confirmed": contact_confirmed,
                "held_by_human": held_by_human,
                "held_by_robot": held_by_robot,
                "human_released": human_released,
                "occluded_object": occluded,
                "meta_hand_obj_rel_speed": hand_obj_rel_speed,
                "meta_gripper_obj_rel_speed": gripper_obj_rel_speed,
            }
        )

        prev_hand_dist = frame.hand_object_distance
        prev_gripper_dist = frame.gripper_object_distance
        prev_held_by_human = held_by_human
        prev_hand_obj_rel_pos = hand_obj_rel_pos
        prev_gripper_obj_rel_pos = gripper_obj_rel_pos

    return scores
