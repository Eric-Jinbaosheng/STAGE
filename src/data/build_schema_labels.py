import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from common_io import write_json, write_parquet


OBJECT_VOCAB = [
    "mug",
    "bowl",
    "can",
    "bottle",
    "box",
    "plate",
    "pot",
    "stove",
    "basket",
    "book",
    "drawer",
    "spoon",
    "block",
    "object_0",
    "UNK",
]

SCHEMA_LABEL_RULESET_VERSION = "week1_frozen_libero_phase_v2"
TARGET_OBJECT_RULESET_VERSION = "week1_frozen_target_object_v2"

OBJECT_PATTERNS = [
    ("bowl", [r"\bbowl\b", r"\bramekin\b"]),
    ("plate", [r"\bplate\b", r"\bdish\b"]),
    ("bottle", [r"\bbottle\b", r"\bwine bottle\b"]),
    ("mug", [r"\bmug\b", r"\bcup\b"]),
    ("pot", [r"\bmoka pot\b", r"\bmoka pots\b", r"\bpot\b", r"\bpots\b", r"\bfrying pan\b", r"\bpan\b", r"\bkettle\b"]),
    ("box", [r"\bbox\b", r"\bcookie box\b"]),
    ("can", [r"\bcan\b"]),
    ("can", [r"\balphabet soup\b"]),
    ("bottle", [r"\btomato sauce\b", r"\bbbq sauce\b", r"\bketchup\b", r"\bmilk\b", r"\borange juice\b", r"\bsalad dressing\b"]),
    ("box", [r"\bchocolate pudding\b", r"\bcream cheese\b", r"\bbutter\b"]),
    ("basket", [r"\bbasket\b", r"\bcaddy\b"]),
    ("book", [r"\bbook\b"]),
    ("stove", [r"\bstove\b", r"\bburner\b"]),
    ("drawer", [r"\bdrawer\b", r"\bcabinet\b"]),
    ("spoon", [r"\bspoon\b"]),
    ("block", [r"\bblock\b"]),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build schema labels parquet from LIBERO/Handover indexes.")
    p.add_argument("--libero-index", default="data/processed/libero_index.parquet")
    p.add_argument("--handover-index", default="data/processed/handover_index.parquet")
    p.add_argument("--out", default="data/processed/schema_labels.parquet")
    p.add_argument("--meta-out", default="data/processed/meta.json")
    p.add_argument("--vocab-out", default="data/processed/object_vocab.json")
    return p.parse_args()


def parse_target_from_instruction(text: str) -> str:
    low = (text or "").lower()
    for target, patterns in OBJECT_PATTERNS:
        for pat in patterns:
            if re.search(pat, low):
                return target
    return "UNK"


def tri(v: float, th_hi: float = 0.65, th_lo: float = 0.35) -> str:
    if v >= th_hi:
        return "T"
    if v <= th_lo:
        return "F"
    return "UNK"


def build_affordance_mask(state_contact: str, state_secured: str, state_occ: str) -> Dict[str, int]:
    mask = {
        "HOLD": 1,
        "BACKOFF_SMALL": 1,
        "VIEWPOINT_CHANGE": 1 if state_occ in ("T", "UNK") else 0,
        "REALIGN": 1 if state_secured != "T" else 0,
        "CLOSE_GENTLE": 1 if state_contact == "T" and state_occ != "T" else 0,
        "RETRACT": 1 if state_secured == "T" else 0,
        "PROMPT": 1 if state_secured != "T" else 0,
    }
    return mask


def phase_from_segments(progress: float, segments: List[tuple]) -> str:
    for upper, phase in segments:
        if progress < upper:
            return phase
    return segments[-1][1] if segments else "UNK"


def phase_handover(row: pd.Series) -> str:
    d = float(row.get("d_grip_obj", 1.0))
    c = float(row.get("contact_flag", 0.0))
    w = float(row.get("gripper_width", 1.0))
    if c > 0.6 and w < 0.35:
        return "secure"
    if c > 0.6:
        return "contact"
    if d < 0.04:
        return "align"
    if d < 0.10:
        return "reach"
    return "reach"


def phase_libero(
    frame_id: int,
    max_frame: int,
    gripper_opening: float,
    action_gripper: float,
    prev_opening: float,
    target_object: str,
    instruction: str,
) -> str:
    if max_frame <= 1:
        return "UNK"
    p = frame_id / max_frame
    low_inst = instruction.lower()
    mentions_drawer = ("drawer" in low_inst) or ("cabinet" in low_inst)
    mentions_stove = ("stove" in low_inst) or ("burner" in low_inst)
    is_composite = (" and " in low_inst) or (" then " in low_inst)

    # Pure articulated / switch-like tasks: no grasp segment.
    if target_object in {"drawer", "stove"}:
        return phase_from_segments(
            p,
            [
                (0.22, "reach"),
                (0.93, "manipulate"),
                (1.01, "done"),
            ],
        )

    # Composite articulated + pick/place tasks.
    if is_composite and (mentions_drawer or mentions_stove):
        return phase_from_segments(
            p,
            [
                (0.16, "reach"),
                (0.34, "manipulate"),
                (0.48, "reach"),
                (0.62, "grasp"),
                (0.90, "manipulate"),
                (0.98, "place"),
                (1.01, "done"),
            ],
        )

    # Push-like tasks: no real grasp, mostly contact-rich manipulation.
    if "push " in low_inst:
        return phase_from_segments(
            p,
            [
                (0.28, "reach"),
                (0.86, "manipulate"),
                (0.97, "place"),
                (1.01, "done"),
            ],
        )

    # Pick tasks that start from inside / on top of articulated furniture often
    # need a longer pre-grasp approach/manipulate period.
    if "pick up " in low_inst and ("drawer" in low_inst or "cabinet" in low_inst or "stove" in low_inst):
        return phase_from_segments(
            p,
            [
                (0.18, "reach"),
                (0.34, "manipulate"),
                (0.52, "grasp"),
                (0.84, "manipulate"),
                (0.97, "place"),
                (1.01, "done"),
            ],
        )

    # Standard pick-and-place tasks.
    if "pick up " in low_inst:
        return phase_from_segments(
            p,
            [
                (0.20, "reach"),
                (0.38, "grasp"),
                (0.76, "manipulate"),
                (0.97, "place"),
                (1.01, "done"),
            ],
        )

    # Standard put tasks start with an object already or nearly grasped.
    if "put " in low_inst:
        return phase_from_segments(
            p,
            [
                (0.12, "reach"),
                (0.28, "grasp"),
                (0.74, "manipulate"),
                (0.97, "place"),
                (1.01, "done"),
            ],
        )

    is_closing = action_gripper < -0.5 or (prev_opening >= 0 and gripper_opening >= 0 and (prev_opening - gripper_opening) > 0.003)
    is_closed = (gripper_opening >= 0) and (gripper_opening < 0.03)
    is_narrow = (gripper_opening >= 0) and (gripper_opening < 0.06)
    is_opening = action_gripper > 0.5 or (prev_opening >= 0 and gripper_opening >= 0 and (gripper_opening - prev_opening) > 0.003)

    if p < 0.18:
        return "reach"
    if p < 0.38 and (is_closing or is_narrow):
        return "grasp"
    if p < 0.78 and (is_closed or is_narrow or not is_opening):
        return "manipulate"
    if p < 0.98:
        return "place"
    return "done"


def build_from_handover(df: pd.DataFrame) -> List[Dict]:
    rows: List[Dict] = []
    if df.empty or "episode_id" not in df.columns or "frame_id" not in df.columns:
        return rows
    for _, r in df.iterrows():
        contact = tri(float(r.get("contact_flag", 0.0)))
        secured = tri(1.0 - float(r.get("gripper_width", 1.0)))
        occluded = tri(1.0 - float(r.get("visibility", 1.0)))
        mask = build_affordance_mask(contact, secured, occluded)
        rows.append(
            {
                "dataset": "handover_sim",
                "episode_id": r["episode_id"],
                "frame_id": int(r["frame_id"]),
                "target_object": "object_0",
                "phase": phase_handover(r),
                "contact_confirmed": contact,
                "object_secured": secured,
                "occluded": occluded,
                "affordance_mask": json.dumps(mask, ensure_ascii=True),
            }
        )
    return rows


def build_from_libero(df: pd.DataFrame) -> List[Dict]:
    rows: List[Dict] = []
    if df.empty or "episode_id" not in df.columns or "frame_id" not in df.columns:
        return rows
    grp_max = df.groupby("episode_id")["frame_id"].max().to_dict() if len(df) else {}
    for ep, sub in df.groupby("episode_id", sort=False):
        sub = sub.sort_values("frame_id")
        prev_open = -1.0
        max_frame = int(grp_max.get(ep, len(sub)))
        for _, r in sub.iterrows():
            fid = int(r["frame_id"])
            target = parse_target_from_instruction(str(r.get("instruction", "")))
            contact = "UNK"
            # Use gripper openness as a weak grasp proxy only for phase, not strong state.
            grip_open = float(r.get("gripper_opening", -1.0))
            action_grip = float(r.get("action_gripper", 0.0))
            secured = "T" if (grip_open >= 0 and grip_open < 0.03 and fid / max(1, max_frame) > 0.25) else "UNK"
            occluded = "UNK"
            mask = build_affordance_mask(contact, secured, occluded)
            rows.append(
                {
                    "dataset": "libero",
                    "episode_id": ep,
                    "frame_id": fid,
                    "target_object": target,
                    "phase": phase_libero(
                        fid,
                        max_frame,
                        grip_open,
                        action_grip,
                        prev_open,
                        target,
                        str(r.get("instruction", "")),
                    ),
                    "contact_confirmed": contact,
                    "object_secured": secured,
                    "occluded": occluded,
                    "affordance_mask": json.dumps(mask, ensure_ascii=True),
                }
            )
            prev_open = grip_open
    return rows


def main() -> None:
    args = parse_args()
    rows: List[Dict] = []
    if Path(args.handover_index).exists():
        h = pd.read_parquet(args.handover_index)
        rows.extend(build_from_handover(h))
    if Path(args.libero_index).exists():
        l = pd.read_parquet(args.libero_index)
        rows.extend(build_from_libero(l))

    out_path = Path(args.out)
    write_parquet(rows, out_path)
    write_json(Path(args.vocab_out), {"object_vocab": OBJECT_VOCAB})

    meta_patch = {
        "schema_labels_rows": len(rows),
        "schema_labels_path": out_path.as_posix(),
        "schema_label_ruleset_version": SCHEMA_LABEL_RULESET_VERSION,
        "target_object_ruleset_version": TARGET_OBJECT_RULESET_VERSION,
    }
    meta_path = Path(args.meta_out)
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        old.update(meta_patch)
        write_json(meta_path, old)
    else:
        write_json(meta_path, meta_patch)

    print(f"Wrote {out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
