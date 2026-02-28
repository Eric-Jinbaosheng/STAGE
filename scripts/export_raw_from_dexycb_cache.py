import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def euler_xyz_to_quat(euler_xyz: np.ndarray) -> np.ndarray:
    # Convert XYZ Euler to quaternion [x, y, z, w].
    x, y, z = float(euler_xyz[0]), float(euler_xyz[1]), float(euler_xyz[2])
    cx = np.cos(x * 0.5)
    sx = np.sin(x * 0.5)
    cy = np.cos(y * 0.5)
    sy = np.sin(y * 0.5)
    cz = np.cos(z * 0.5)
    sz = np.sin(z * 0.5)
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    return np.array([qx, qy, qz, qw], dtype=np.float32)


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def finite_velocity(seq: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(seq, dtype=np.float32)
    if len(seq) > 1:
        vel[1:] = (seq[1:] - seq[:-1]) / max(dt, 1e-6)
        vel[0] = vel[1]
    return vel


def build_gripper_pose(
    hand_pos: np.ndarray, obj_pos: np.ndarray, obj_quat: np.ndarray, approach_offset: float
) -> Tuple[np.ndarray, np.ndarray]:
    # Pseudo robot gripper: placed opposite to human hand around object.
    vec = normalize(obj_pos - hand_pos)
    if float(np.linalg.norm(vec)) < 1e-8:
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    gripper_pos = obj_pos + vec * approach_offset
    gripper_quat = obj_quat.copy()
    return gripper_pos.astype(np.float32), gripper_quat.astype(np.float32)


def pose_to_raw_frames(
    pose_y: np.ndarray,
    pose_m: np.ndarray,
    object_index: int,
    episode_id: str,
    dt: float,
) -> List[Dict]:
    # pose_y: [T, num_obj, 6] => [tx,ty,tz, ex,ey,ez]
    # pose_m: [T, num_hand, 51] => [tx,ty,tz, euler(48)]
    t_len = pose_y.shape[0]

    obj_pos_seq = pose_y[:, object_index, :3].astype(np.float32)
    obj_eul_seq = pose_y[:, object_index, 3:6].astype(np.float32)
    obj_quat_seq = np.array([euler_xyz_to_quat(e) for e in obj_eul_seq], dtype=np.float32)

    hand_pos_seq = pose_m[:, 0, :3].astype(np.float32)
    hand_eul_seq = pose_m[:, 0, 3:6].astype(np.float32)
    hand_quat_seq = np.array([euler_xyz_to_quat(e) for e in hand_eul_seq], dtype=np.float32)

    gripper_pos_seq = np.zeros_like(obj_pos_seq, dtype=np.float32)
    gripper_quat_seq = np.zeros_like(obj_quat_seq, dtype=np.float32)
    for i in range(t_len):
        gp, gq = build_gripper_pose(hand_pos_seq[i], obj_pos_seq[i], obj_quat_seq[i], approach_offset=0.08)
        gripper_pos_seq[i] = gp
        gripper_quat_seq[i] = gq

    hand_vel_seq = finite_velocity(hand_pos_seq, dt)
    obj_vel_seq = finite_velocity(obj_pos_seq, dt)
    gripper_vel_seq = finite_velocity(gripper_pos_seq, dt)

    rows: List[Dict] = []
    for i in range(t_len):
        hand_obj_dist = float(np.linalg.norm(hand_pos_seq[i] - obj_pos_seq[i]))
        grip_obj_dist = float(np.linalg.norm(gripper_pos_seq[i] - obj_pos_seq[i]))
        contact_h = 1.0 if hand_obj_dist < 0.035 else 0.0
        rel_speed_g = float(np.linalg.norm(gripper_vel_seq[i] - obj_vel_seq[i]))
        contact_g = 1.0 if (grip_obj_dist < 0.03 and rel_speed_g < 0.25) else 0.0
        gripper_opening = float(min(1.0, max(0.0, grip_obj_dist / 0.08)))

        rows.append(
            {
                "episode_id": episode_id,
                "timestamp": float(i * dt),
                "human": {
                    "hand": {
                        "position": hand_pos_seq[i].tolist(),
                        "orientation": hand_quat_seq[i].tolist(),
                        "velocity": hand_vel_seq[i].tolist(),
                    }
                },
                "object": {
                    "position": obj_pos_seq[i].tolist(),
                    "orientation": obj_quat_seq[i].tolist(),
                    "velocity": obj_vel_seq[i].tolist(),
                },
                "robot": {
                    "gripper": {
                        "position": gripper_pos_seq[i].tolist(),
                        "orientation": gripper_quat_seq[i].tolist(),
                        "velocity": gripper_vel_seq[i].tolist(),
                        "opening": gripper_opening,
                    }
                },
                "vision": {"visibility_score": 1.0},
                "signals": {
                    "hand_object_distance": hand_obj_dist,
                    "gripper_object_distance": grip_obj_dist,
                    "contact_hand_object": contact_h,
                    "contact_gripper_object": contact_g,
                },
                "meta": {"source": "dex-ycb-cache", "episode_id": episode_id},
            }
        )
    return rows


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export raw handover-like episodes from DexYCB cache.")
    p.add_argument("--cache-dir", required=True, help="Path to dex-ycb-cache folder.")
    p.add_argument("--out-dir", required=True, help="Output dir for raw episodes.")
    p.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to export.")
    p.add_argument("--start-index", type=int, default=0, help="Start cache index.")
    p.add_argument("--dt", type=float, default=0.01, help="Frame dt for exported sequence.")
    p.add_argument("--max-frames", type=int, default=800, help="Cap frames per episode.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    idx = args.start_index
    while exported < args.num_episodes:
        pose_file = cache_dir / f"pose_{idx:03d}.npz"
        meta_file = cache_dir / f"meta_{idx:03d}.json"
        if not pose_file.exists() or not meta_file.exists():
            break

        pose = np.load(pose_file, allow_pickle=True)
        pose_y = np.array(pose["pose_y"], dtype=np.float32)
        pose_m = np.array(pose["pose_m"], dtype=np.float32)
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        obj_idx = int(meta.get("ycb_grasp_ind", 0))
        obj_idx = max(0, min(obj_idx, pose_y.shape[1] - 1))

        if args.max_frames > 0:
            pose_y = pose_y[: args.max_frames]
            pose_m = pose_m[: args.max_frames]

        ep_id = f"ep_{exported + 1:04d}"
        rows = pose_to_raw_frames(pose_y, pose_m, obj_idx, ep_id, args.dt)
        out_file = out_dir / f"{ep_id}.json"
        write_json(out_file, {"frames": rows})
        print(f"Exported {out_file} ({len(rows)} frames) from cache idx {idx:03d}")

        exported += 1
        idx += 1

    print(f"Done. Exported episodes: {exported}")


if __name__ == "__main__":
    main()

