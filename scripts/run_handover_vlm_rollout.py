import argparse
import json
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
HANDOVER_ROOT = REPO_ROOT / "third_party" / "handover-sim"
HANDOVER_EXAMPLES = HANDOVER_ROOT / "examples"

for path in (REPO_ROOT, HANDOVER_ROOT, HANDOVER_EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from handover.config import cfg as default_cfg
from handover.benchmark_runner import BenchmarkRunner
from handover.benchmark_wrapper import EpisodeStatus
from run_benchmark_yang_icra2021 import YangICRA2021Policy, start_conf
from src.h2r_schema.relations import RelationConfig, compute_relations
from src.h2r_schema.types import FrameRecord, Pose


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run VLM-side handover rollout on true Handover-Sim benchmark.")
    p.add_argument("--model-name", required=True)
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--task-profile", default="", help="Optional JSON profile for task transfer (thresholds/target/instruction overrides).")
    p.add_argument("--instruction-template", default="handover {target} to robot")
    p.add_argument("--target-object-key", default="", help="Preferred object key in obs['ycb_bodies']; empty means auto-resolve.")
    p.add_argument("--target-object-index", type=int, default=0, help="Fallback target object index when key is not set.")
    p.add_argument("--setup", default="s0")
    p.add_argument("--split", default="val")
    p.add_argument("--scene-index", type=int, default=0)
    p.add_argument("--scene-indices", default="")
    p.add_argument("--env-id", default="HandoverStateEnv-v1")
    p.add_argument("--policy-time-wait", type=float, default=0.0, help="Delay before robot policy starts moving.")
    p.add_argument("--contact-force-thresh", type=float, default=0.5, help="Benchmark contact force threshold.")
    p.add_argument("--release-force-thresh", type=float, default=0.5, help="Release contact force threshold.")
    p.add_argument("--mano-restitution", type=float, default=0.1, help="MANO link restitution (lower -> less bounce).")
    p.add_argument("--res-dir", required=True)
    p.add_argument("--out-path", default="")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--video-fps", type=int, default=20)
    p.add_argument("--video-every-n", type=int, default=1, help="Record every N frames.")
    p.add_argument("--video-max-frames", type=int, default=0, help="Maximum recorded frames per episode (0 means unlimited).")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--schema-window-sec", type=float, default=1.5, help="Structured summary sliding window length in seconds.")
    p.add_argument("--schema-window-min-frames", type=int, default=10, help="Minimum frames kept in structured summary window.")
    p.add_argument(
        "--intervention-mode",
        choices=["minimal", "aggressive"],
        default="minimal",
        help="minimal: keep native Yang low-level control; aggressive: enable lock/close-stage control interventions.",
    )
    p.add_argument("--vlm-interval", type=int, default=25, help="Run VLM every N frames; other frames use low-level policy.")
    p.add_argument("--enable-event-trigger", action="store_true", help="Trigger VLM on sudden state changes.")
    p.add_argument("--event-dgrip-delta", type=float, default=0.01)
    p.add_argument("--event-hand-obj-delta", type=float, default=0.012)
    p.add_argument("--max-joint-step", type=float, default=0.03, help="Max per-frame joint delta for q0..q6.")
    p.add_argument("--slowdown-dist", type=float, default=0.12, help="Slow down when gripper-object distance below this.")
    p.add_argument("--near-step-scale", type=float, default=0.35, help="Step scale applied within slowdown-dist.")
    p.add_argument("--assist-close-dist", type=float, default=0.06, help="Activate gripper-close assist when d_grip_obj <= this.")
    p.add_argument("--assist-close-release-dist", type=float, default=0.085, help="Release close assist when far again.")
    p.add_argument("--assist-close-hold-frames", type=int, default=90, help="Minimum frames to keep gripper close after assist triggers.")
    p.add_argument("--grasp-lock-enter-dist", type=float, default=0.09, help="Activate grasp lock when d_grip_obj <= this.")
    p.add_argument("--grasp-lock-exit-dist", type=float, default=0.13, help="Release grasp lock when d_grip_obj >= this and hold elapsed.")
    p.add_argument("--grasp-lock-hold-frames", type=int, default=220, help="Minimum lock duration in frames.")
    p.add_argument("--force-grasp-dist", type=float, default=0.06, help="Geometry-based force-grasp distance.")
    p.add_argument("--force-grasp-hold-frames", type=int, default=35, help="Need this many near frames to force-grasp.")
    p.add_argument("--force-grasp-latch-dist", type=float, default=0.09, help="Latch when once entering this near field.")
    p.add_argument(
        "--close-stage-enter-dist",
        type=float,
        default=0.09,
        help="Enter near-field close stage when hand/object distance is below this.",
    )
    p.add_argument(
        "--close-stage-hold-frames",
        type=int,
        default=8,
        help="Need this many consecutive near frames to trigger close stage.",
    )
    p.add_argument(
        "--close-stage-max-frames",
        type=int,
        default=80,
        help="Max frames spent in close stage before forced commit/backoff.",
    )
    p.add_argument("--disable-human-ready-gate", action="store_true", help="Disable wait-for-human readiness gate.")
    p.add_argument("--human-ready-max-dist", type=float, default=0.16, help="Min MANO-to-object distance threshold.")
    p.add_argument("--disable-human-stable-gate", action="store_true", help="Disable hand-motion stability gate.")
    p.add_argument("--human-stable-max-speed", type=float, default=0.10, help="Require hand speed <= this threshold.")
    p.add_argument("--human-stable-min-frames", type=int, default=16, help="Consecutive stable frames required.")
    p.add_argument("--approach-keepout-dist", type=float, default=0.14, help="Do not let gripper get closer than this while human hand is still moving.")
    p.add_argument("--handover-hand-obj-near-dist", type=float, default=0.12, help="Treat hand as carrying/controlling object when hand-object distance is below this.")
    p.add_argument("--no-approach-moving-dist", type=float, default=0.35, help="Hard block approach when hand/object is moving and gripper-object distance is below this.")
    p.add_argument("--motion-lock-release-stable-frames", type=int, default=45, help="Require this many stable frames to release motion-lock.")
    p.add_argument("--motion-lock-obj-speed-max", type=float, default=0.03, help="Object speed threshold for releasing motion-lock.")
    p.add_argument("--schema-reacquire-retreat-step", type=float, default=0.015, help="Joint retreat step when schema triggers reacquire mode.")
    p.add_argument("--pre-release-keepout-dist", type=float, default=0.20, help="Hard keepout distance to object before HUMAN_RELEASED is confirmed.")
    p.add_argument("--release-fallback-frame", type=int, default=3200, help="After this frame, allow progression with stable geometry even if HUMAN_RELEASED is not explicitly confirmed.")
    p.add_argument("--wait-hover-z-offset", type=float, default=0.18, help="Hover height above hand/object during wait states.")
    p.add_argument("--wait-hover-xy-blend", type=float, default=0.70, help="XY blend between hand and object centers for hover target.")
    p.add_argument("--wait-hover-max-joint-step", type=float, default=0.02, help="Per-joint step cap for wait-hover IK action.")
    p.add_argument("--wait-update-interval", type=int, default=12, help="Update wait-hover IK every N frames.")
    p.add_argument("--wait-target-deadband", type=float, default=0.03, help="Do not refresh wait-hover target if movement is below this distance.")
    p.add_argument(
        "--human-ready-max-wait-frame",
        type=int,
        default=500,
        help="After this frame, stop blocking on wait_human even if human-ready is false.",
    )
    p.add_argument("--disable-geom-gate", action="store_true", help="Disable geometry-consistency hard gate.")
    p.add_argument("--geom-contact-max-dist", type=float, default=0.05)
    p.add_argument("--geom-secure-max-dist", type=float, default=0.04)
    p.add_argument("--max-vlm-calls", type=int, default=24, help="Hard budget: max VLM calls per episode.")
    p.add_argument("--min-vlm-gap", type=int, default=20, help="Minimum frame gap between two VLM calls.")
    p.add_argument("--near-thresh", type=float, default=0.18, help="Only trigger VLM when near handover region.")
    p.add_argument("--uncertainty-high-thresh", type=float, default=0.75)
    p.add_argument("--uncertainty-low-thresh", type=float, default=0.40)
    p.add_argument("--release-commit-thresh", type=float, default=0.55)
    p.add_argument("--robot-hold-commit-thresh", type=float, default=0.65)
    p.add_argument("--near-contact-commit-dist", type=float, default=0.052, help="Geometric near-contact distance for commit fallback.")
    p.add_argument("--near-contact-commit-hold-frames", type=int, default=22, help="Required consecutive near-contact frames before commit fallback.")
    p.add_argument(
        "--safety-hand-keepout-dist",
        type=float,
        default=0.12,
        help="Before human release is confirmed, hold if gripper-hand distance is below this.",
    )
    p.add_argument("--commit-min-frame", type=int, default=120, help="Disallow commit before this frame.")
    p.add_argument(
        "--commit-vlm-votes",
        type=int,
        default=2,
        help="Require this many VLM secure votes before commit.",
    )
    p.add_argument(
        "--force-commit-frame",
        type=int,
        default=900,
        help="After this frame, allow commit with weaker condition if schema stays secure.",
    )
    p.add_argument(
        "--hard-force-commit-frame",
        type=int,
        default=0,
        help="Controller-level hard commit fallback; <=0 disables.",
    )
    p.add_argument(
        "--force-abort-frame",
        type=int,
        default=12000,
        help="Abort before env timeout if still not committed.",
    )
    p.add_argument(
        "--cold-start-mode",
        choices=["always", "near", "off"],
        default="near",
        help="When to allow the first VLM call.",
    )
    p.add_argument("--use-schema-checker", action="store_true")
    p.add_argument("--use-consistency-rerank", action="store_true")
    p.add_argument("--enable-non-target-keepout", action="store_true", help="Block approach when gripper is too close to non-target objects.")
    p.add_argument("--non-target-keepout-dist", type=float, default=0.09)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


def prompt_from_instruction(instruction: str, structured_window: str = "") -> str:
    structured_block = structured_window.strip() if structured_window else "N/A"
    return (
        "You are a robot policy assistant.\n"
        "Given the instruction and the structured window summary, output the interaction schema in the exact format.\n"
        "Output exactly eight lines.\n"
        "Do not output JSON.\n"
        "Do not output code.\n"
        "Do not explain anything.\n"
        "Use this exact field order:\n"
        "TARGET=<...>\n"
        "PHASE=<...>\n"
        "SECURED=<T|F|UNK>\n"
        "CONTACT=<T|F|UNK>\n"
        "OCCLUDED=<T|F|UNK>\n"
        "HUMAN_RELEASED=<T|F|UNK>\n"
        "CONFLICT=<float in [0,1]>\n"
        "AFF=[HOLD=0/1,BACKOFF_SMALL=0/1,VIEWPOINT_CHANGE=0/1,REALIGN=0/1,CLOSE_GENTLE=0/1,RETRACT=0/1,PROMPT=0/1]\n\n"
        f"INSTRUCTION: {instruction}\n"
        "STRUCTURED_WINDOW:\n"
        f"{structured_block}\n"
        "OUTPUT_SCHEMA:\n"
    )


def render_multimodal_prompt(processor: Any, prompt_text: str, add_generation_prompt: bool = True) -> str:
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return prompt_text


def normalize_tri(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"TRUE", "1"}:
        return "T"
    if text in {"FALSE", "0"}:
        return "F"
    if text in {"T", "F", "UNK"}:
        return text
    return "UNK"


def _vote(values: List[str], default: str = "UNK") -> str:
    clean = [str(v) for v in values if str(v)]
    if not clean:
        return default
    counts: Dict[str, int] = {}
    for v in clean:
        counts[v] = counts.get(v, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _vote_aff(keys: List[str], maps: List[Dict[str, int]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key in keys:
        vals = [1 if int(m.get(key, 0)) else 0 for m in maps]
        out[key] = 1 if sum(vals) >= max(1, len(vals) - sum(vals)) else 0
    return out


class VLMYangPolicy(YangICRA2021Policy):
    def __init__(self, cfg, args: argparse.Namespace):
        super().__init__(cfg, time_wait=float(args.policy_time_wait))
        self._args = args
        self._task_profile: Dict[str, Any] = {}
        if str(args.task_profile).strip():
            profile_path = Path(args.task_profile).expanduser().resolve()
            with profile_path.open("r", encoding="utf-8") as f:
                self._task_profile = dict(json.load(f) or {})
        self._instruction_template = str(
            self._task_profile.get("instruction_template", args.instruction_template)
        )
        self._target_object_key_pref = str(
            self._task_profile.get("target_object_key", args.target_object_key)
        ).strip()
        self._target_object_index_pref = max(
            0,
            int(self._task_profile.get("target_object_index", args.target_object_index)),
        )
        self._target_object_key_runtime: Optional[str] = None
        self._instruction = self._instruction_template.format(target=f"object_{self._target_object_index_pref}")
        self._intervention_mode = str(args.intervention_mode)
        self._sim_dt = float(getattr(cfg.SIM, "TIME_STEP", 1.0 / 240.0))
        sec = max(0.5, float(args.schema_window_sec))
        min_frames = max(2, int(args.schema_window_min_frames))
        self._schema_window_frames = max(min_frames, int(round(sec / max(self._sim_dt, 1e-4))))
        self._frame_buffer: deque[FrameRecord] = deque(maxlen=self._schema_window_frames)
        self._prev_hand_pos: Optional[np.ndarray] = None
        self._prev_gripper_pos: Optional[np.ndarray] = None
        self._prev_obj_pos: Optional[np.ndarray] = None
        self._rel_cfg = RelationConfig()
        self._vlm_interval = max(1, int(args.vlm_interval))
        self._enable_event_trigger = bool(args.enable_event_trigger)
        self._event_dgrip_delta = float(args.event_dgrip_delta)
        self._event_hand_obj_delta = float(args.event_hand_obj_delta)
        self._max_joint_step = float(args.max_joint_step)
        self._slowdown_dist = float(args.slowdown_dist)
        self._near_step_scale = float(args.near_step_scale)
        self._assist_close_dist = float(args.assist_close_dist)
        self._assist_close_release_dist = float(args.assist_close_release_dist)
        self._assist_close_hold_frames = max(1, int(args.assist_close_hold_frames))
        self._grasp_lock_enter_dist = float(args.grasp_lock_enter_dist)
        self._grasp_lock_exit_dist = float(args.grasp_lock_exit_dist)
        self._grasp_lock_hold_frames = max(1, int(args.grasp_lock_hold_frames))
        self._force_grasp_dist = float(args.force_grasp_dist)
        self._force_grasp_hold_frames = max(1, int(args.force_grasp_hold_frames))
        self._force_grasp_latch_dist = float(args.force_grasp_latch_dist)
        self._close_stage_enter_dist = float(args.close_stage_enter_dist)
        self._close_stage_hold_frames = max(1, int(args.close_stage_hold_frames))
        self._close_stage_max_frames = max(1, int(args.close_stage_max_frames))
        self._geom_gate_enabled = not bool(args.disable_geom_gate)
        self._geom_contact_max_dist = float(args.geom_contact_max_dist)
        self._geom_secure_max_dist = float(args.geom_secure_max_dist)
        self._human_ready_gate_enabled = not bool(args.disable_human_ready_gate)
        self._human_ready_max_dist = float(args.human_ready_max_dist)
        self._human_stable_gate_enabled = not bool(args.disable_human_stable_gate)
        self._human_stable_max_speed = float(args.human_stable_max_speed)
        self._human_stable_min_frames = max(1, int(args.human_stable_min_frames))
        self._approach_keepout_dist = float(args.approach_keepout_dist)
        self._handover_hand_obj_near_dist = float(args.handover_hand_obj_near_dist)
        self._no_approach_moving_dist = float(args.no_approach_moving_dist)
        self._motion_lock_release_stable_frames = max(1, int(args.motion_lock_release_stable_frames))
        self._motion_lock_obj_speed_max = float(args.motion_lock_obj_speed_max)
        self._schema_reacquire_retreat_step = float(args.schema_reacquire_retreat_step)
        self._pre_release_keepout_dist = float(args.pre_release_keepout_dist)
        self._release_fallback_frame = max(0, int(args.release_fallback_frame))
        self._wait_hover_z_offset = float(args.wait_hover_z_offset)
        self._wait_hover_xy_blend = float(np.clip(args.wait_hover_xy_blend, 0.0, 1.0))
        self._wait_hover_max_joint_step = float(max(1e-4, args.wait_hover_max_joint_step))
        self._wait_update_interval = max(1, int(args.wait_update_interval))
        self._wait_target_deadband = float(max(1e-4, args.wait_target_deadband))
        self._human_ready_max_wait_frame = max(0, int(args.human_ready_max_wait_frame))
        self._max_vlm_calls = max(1, int(args.max_vlm_calls))
        self._min_vlm_gap = max(1, int(args.min_vlm_gap))
        self._near_thresh = float(args.near_thresh)
        self._uncertainty_high_thresh = float(args.uncertainty_high_thresh)
        self._uncertainty_low_thresh = float(args.uncertainty_low_thresh)
        self._release_commit_thresh = float(args.release_commit_thresh)
        self._robot_hold_commit_thresh = float(args.robot_hold_commit_thresh)
        self._near_contact_commit_dist = float(args.near_contact_commit_dist)
        self._near_contact_commit_hold_frames = max(1, int(args.near_contact_commit_hold_frames))
        self._safety_hand_keepout_dist = float(args.safety_hand_keepout_dist)
        self._commit_min_frame = max(0, int(args.commit_min_frame))
        self._commit_vlm_votes = max(1, int(args.commit_vlm_votes))
        self._force_commit_frame = max(0, int(args.force_commit_frame))
        self._hard_force_commit_frame = max(0, int(args.hard_force_commit_frame))
        self._force_abort_frame = max(0, int(args.force_abort_frame))
        self._enable_non_target_keepout = bool(args.enable_non_target_keepout)
        self._non_target_keepout_dist = float(args.non_target_keepout_dist)
        self._cold_start_mode = str(args.cold_start_mode)
        self._last_schema: Dict[str, Any] = {}
        self._last_phase = "reach"
        self._last_decision = "continue"
        self._last_vlm_frame = -1
        self._vlm_calls = 0
        self._secure_vote_count = 0
        self._occluded_vote_count = 0
        self._geom_commit_counter = 0
        self._release_vote_count = 0
        self._last_event_feat: Optional[Dict[str, float]] = None
        self._last_relation_scores: Dict[str, float] = {}
        self._policy_aborted = False
        self._abort_reason = ""
        self._grasp_lock_active = False
        self._grasp_lock_until_frame = -1
        self._force_grasp_counter = 0
        self._force_grasp_latched = False
        self._force_grasp_latch_start = -1
        self._close_stage_near_counter = 0
        self._close_stage_active = False
        self._close_stage_start_frame = -1
        self._stuck_counter = 0
        self._prev_cfg: Optional[np.ndarray] = None
        self._assist_close_active = False
        self._assist_close_until_frame = -1
        self._near_contact_commit_counter = 0
        self._human_stable_counter = 0
        self._prev_event_hand_center: Optional[np.ndarray] = None
        self._prev_event_obj_center: Optional[np.ndarray] = None
        self._motion_lock_active = False
        self._motion_lock_stable_counter = 0
        self._release_fallback_stable_counter = 0
        self._wait_action_cache: Optional[np.ndarray] = None
        self._wait_target_cache: Optional[np.ndarray] = None
        self._wait_last_update_frame = -1
        self._target_object_index_runtime = self._target_object_index_pref

        threshold_overrides = dict(self._task_profile.get("threshold_overrides", {}) or {})
        if threshold_overrides:
            for key, value in threshold_overrides.items():
                attr = f"_{key}"
                if hasattr(self, attr):
                    cur = getattr(self, attr)
                    try:
                        setattr(self, attr, type(cur)(value))
                    except Exception:
                        setattr(self, attr, value)

        from peft import PeftModel
        from src.data.schema_text_core import (
            PRIMITIVE_ORDER,
            parse_schema_text,
            render_schema_fields,
            repair_schema_prediction,
        )
        from src.model.load_vlm import VLMConfig, load_processor, load_vlm_model

        self._PRIMITIVE_ORDER = PRIMITIVE_ORDER
        self._parse_schema_text = parse_schema_text
        self._render_schema_fields = render_schema_fields
        self._repair_schema_prediction = repair_schema_prediction

        torch_dtype = "bfloat16" if args.bf16 else ("float16" if args.fp16 else "auto")
        vlm_cfg = VLMConfig(
            model_name_or_path=args.model_name,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self._processor = load_processor(args.model_name, trust_remote_code=args.trust_remote_code)
        base_model = load_vlm_model(vlm_cfg)
        self._model = PeftModel.from_pretrained(base_model, args.adapter_path)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    @property
    def name(self):
        return "vlm-checker-consistency-yang-wrapper"

    def reset(self):
        super().reset()
        self._last_schema = {}
        self._last_phase = "reach"
        self._last_decision = "continue"
        self._last_vlm_frame = -1
        self._vlm_calls = 0
        self._secure_vote_count = 0
        self._occluded_vote_count = 0
        self._geom_commit_counter = 0
        self._release_vote_count = 0
        self._last_event_feat = None
        self._last_relation_scores = {}
        self._policy_aborted = False
        self._abort_reason = ""
        self._grasp_lock_active = False
        self._grasp_lock_until_frame = -1
        self._force_grasp_counter = 0
        self._force_grasp_latched = False
        self._force_grasp_latch_start = -1
        self._close_stage_near_counter = 0
        self._close_stage_active = False
        self._close_stage_start_frame = -1
        self._stuck_counter = 0
        self._prev_cfg = None
        self._assist_close_active = False
        self._assist_close_until_frame = -1
        self._near_contact_commit_counter = 0
        self._human_stable_counter = 0
        self._prev_event_hand_center = None
        self._prev_event_obj_center = None
        self._motion_lock_active = False
        self._motion_lock_stable_counter = 0
        self._release_fallback_stable_counter = 0
        self._wait_action_cache = None
        self._wait_target_cache = None
        self._wait_last_update_frame = -1
        self._target_object_key_runtime = None
        self._target_object_index_runtime = self._target_object_index_pref
        self._frame_buffer.clear()
        self._prev_hand_pos = None
        self._prev_gripper_pos = None
        self._prev_obj_pos = None

    def _current_image(self) -> np.ndarray:
        rgba = self._runner_env.render_offscreen()
        img = np.asarray(rgba, dtype=np.uint8)
        if img.ndim == 3 and img.shape[-1] >= 3:
            img = img[..., :3]
        return img

    def _ordered_object_keys(self, obs) -> List[Any]:
        keys = list(obs["ycb_bodies"].keys())
        # Keep environment insertion order: index 0 corresponds to the grasped target
        # in Handover-Sim scene construction.
        return keys

    def _schema_target_to_index(self, schema_target: str) -> Optional[int]:
        s = str(schema_target or "").strip()
        if not s:
            return None
        m = re.search(r"(\d+)$", s)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _resolve_target_object_key(self, obs, schema_target: str = "") -> Any:
        ordered = self._ordered_object_keys(obs)
        ordered_str = [str(k) for k in ordered]
        if not ordered:
            raise RuntimeError("No ycb_bodies found in observation.")
        if self._target_object_key_pref and self._target_object_key_pref in ordered_str:
            key = ordered[ordered_str.index(self._target_object_key_pref)]
        elif self._target_object_key_runtime and self._target_object_key_runtime in ordered_str:
            key = ordered[ordered_str.index(self._target_object_key_runtime)]
        elif str(schema_target).strip():
            st = str(schema_target).strip()
            if st in ordered_str:
                key = ordered[ordered_str.index(st)]
            else:
                idx = self._schema_target_to_index(st)
                key = ordered[min(max(0, idx or 0), len(ordered) - 1)]
        else:
            key = ordered[min(self._target_object_index_pref, len(ordered) - 1)]
        self._target_object_key_runtime = str(key)
        self._target_object_index_runtime = int(ordered.index(key))
        return key

    def _target_body(self, obs, schema_target: str = ""):
        key = self._resolve_target_object_key(obs, schema_target=schema_target)
        return obs["ycb_bodies"][key], str(key)

    def _target_object_pose7(self, obs, schema_target: str = "") -> Tuple[np.ndarray, str]:
        obj_body, key = self._target_body(obs, schema_target=schema_target)
        obj_pose7 = np.asarray(obj_body.link_state[0, 6, 0:7].numpy(), dtype=np.float32)
        return obj_pose7, key

    def _update_instruction_from_target(self, obs, schema_target: str = "") -> None:
        key = self._resolve_target_object_key(obs, schema_target=schema_target)
        self._instruction = self._instruction_template.format(target=str(key))

    def _generate_schema(self, image: np.ndarray, instruction: str, structured_window: str = "") -> Dict[str, Any]:
        prompt_text = prompt_from_instruction(instruction, structured_window=structured_window)
        rendered_prompt = render_multimodal_prompt(self._processor, prompt_text, add_generation_prompt=True)
        encoded = self._processor(
            text=[rendered_prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        for key, value in list(encoded.items()):
            if torch.is_tensor(value):
                encoded[key] = value.to(self._device)
        prompt_len = int(encoded["input_ids"].shape[1])
        out = self._model.generate(
            **encoded,
            max_new_tokens=self._args.max_new_tokens,
            min_new_tokens=min(16, self._args.max_new_tokens),
            do_sample=False,
        )
        generated = out[:, prompt_len:]
        decoded = self._processor.batch_decode(generated, skip_special_tokens=True)
        text = decoded[0].strip() if decoded else ""
        parsed = self._parse_schema_text(text)
        if self._args.use_schema_checker:
            parsed = self._repair_schema_prediction(parsed, instruction=instruction, dataset="handover_sim")
        return parsed

    def _obs_to_frame_record(self, obs) -> FrameRecord:
        obj_pose7, target_key = self._target_object_pose7(obs, schema_target=self._last_schema.get("TARGET", ""))
        obj_pos = obj_pose7[0:3]
        obj_quat = obj_pose7[3:7]

        gr_pose7 = np.asarray(
            obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:7].numpy(),
            dtype=np.float32,
        )
        gr_pos = gr_pose7[0:3]
        gr_quat = gr_pose7[3:7]

        mano_body = obs.get("mano_body", None)
        if mano_body is not None:
            hand_xyz = np.asarray(mano_body.link_state[0, :, 0:3].numpy(), dtype=np.float32)
            hand_pos = hand_xyz.mean(axis=0) if hand_xyz.size else obj_pos.copy()
        else:
            hand_pos = obj_pos.copy()
        hand_quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        hand_vel = (
            ((hand_pos - self._prev_hand_pos) / max(self._sim_dt, 1e-4)).tolist()
            if self._prev_hand_pos is not None
            else [0.0, 0.0, 0.0]
        )
        gr_vel = (
            ((gr_pos - self._prev_gripper_pos) / max(self._sim_dt, 1e-4)).tolist()
            if self._prev_gripper_pos is not None
            else [0.0, 0.0, 0.0]
        )
        obj_vel = (
            ((obj_pos - self._prev_obj_pos) / max(self._sim_dt, 1e-4)).tolist()
            if self._prev_obj_pos is not None
            else [0.0, 0.0, 0.0]
        )

        self._prev_hand_pos = hand_pos.copy()
        self._prev_gripper_pos = gr_pos.copy()
        self._prev_obj_pos = obj_pos.copy()

        d_hand_obj = float(np.linalg.norm(hand_pos - obj_pos))
        d_grip_obj = float(np.linalg.norm(gr_pos - obj_pos))
        visibility_score = float(np.clip(1.0 - d_hand_obj / 0.6, 0.0, 1.0))
        contact_hand_obj = 1.0 if d_hand_obj <= 0.03 else 0.0
        contact_grip_obj = 1.0 if d_grip_obj <= 0.05 else 0.0
        current_cfg = obs["panda_body"].dof_state[0, :, 0].numpy().copy()
        gripper_opening = float(max(current_cfg[7], current_cfg[8]) / 0.04)

        return FrameRecord(
            timestamp=float(int(obs["frame"]) * self._sim_dt),
            human_hand_pose=Pose(position_xyz=hand_pos.tolist(), orientation_xyzw=hand_quat.tolist()),
            object_pose=Pose(position_xyz=obj_pos.tolist(), orientation_xyzw=obj_quat.tolist()),
            gripper_pose=Pose(position_xyz=gr_pos.tolist(), orientation_xyzw=gr_quat.tolist()),
            hand_object_distance=d_hand_obj,
            gripper_object_distance=d_grip_obj,
            hand_velocity=[float(x) for x in hand_vel],
            gripper_velocity=[float(x) for x in gr_vel],
            object_velocity=[float(x) for x in obj_vel],
            visibility_score=visibility_score,
            contact_signal_hand_object=float(contact_hand_obj),
            contact_signal_gripper_object=float(contact_grip_obj),
            gripper_opening=float(np.clip(gripper_opening, 0.0, 1.0)),
            meta={"frame": int(obs["frame"]), "target_key": target_key},
        )

    def _build_structured_window_summary(self, obs) -> str:
        frame_rec = self._obs_to_frame_record(obs)
        self._frame_buffer.append(frame_rec)
        records = list(self._frame_buffer)
        if not records:
            return "N/A"
        rel = compute_relations(records, window=min(len(records), self._schema_window_frames), config=self._rel_cfg)[-1]
        self._last_relation_scores = dict(rel)
        cur = records[-1]
        visibility_state = "occluded" if rel["occluded_object"] >= 0.5 else "visible"
        contact_state = "confirmed" if rel["contact_confirmed"] >= 0.5 else ("possible" if rel["contact_possible"] >= 0.5 else "none")
        grasp_state_human = "held" if rel["held_by_human"] >= 0.5 else "not_held"
        grasp_state_robot = "held" if rel["held_by_robot"] >= 0.5 else "not_held"
        conflict_score = float(
            np.clip(
                0.45 * float(rel["occluded_object"])
                + 0.30 * abs(float(rel["contact_possible"]) - float(rel["contact_confirmed"]))
                + 0.25 * max(0.0, float(rel["held_by_human"]) + float(rel["held_by_robot"]) - 1.2),
                0.0,
                1.0,
            )
        )

        lines = [
            "ENTITY_TABLE:",
            (
                f"human_hand: pos={np.round(cur.human_hand_pose.position_xyz,4).tolist()} "
                f"vel={np.round(cur.hand_velocity,4).tolist()}"
            ),
            (
                f"gripper: pos={np.round(cur.gripper_pose.position_xyz,4).tolist()} "
                f"vel={np.round(cur.gripper_velocity,4).tolist()} opening={cur.gripper_opening:.3f}"
            ),
            (
                f"object(target={cur.meta.get('target_key','object_0')}): pos={np.round(cur.object_pose.position_xyz,4).tolist()} "
                f"vel={np.round(cur.object_velocity,4).tolist()}"
            ),
            f"visibility_state={visibility_state}",
            f"contact_state={contact_state}",
            f"grasp_state_human={grasp_state_human}",
            f"grasp_state_robot={grasp_state_robot}",
            "RELATION_CANDIDATES(score|evidence):",
            (
                f"approaching(hand,object)={rel['approaching_hand_object']:.3f} | "
                f"evidence=d_hand_obj:{cur.hand_object_distance:.3f},v_hand:{np.linalg.norm(cur.hand_velocity):.3f}"
            ),
            (
                f"approaching(gripper,object)={rel['approaching_gripper_object']:.3f} | "
                f"evidence=d_grip_obj:{cur.gripper_object_distance:.3f},v_grip:{np.linalg.norm(cur.gripper_velocity):.3f}"
            ),
            (
                f"aligned(gripper,object)={rel['aligned_gripper_object']:.3f} | "
                f"evidence=d_grip_obj:{cur.gripper_object_distance:.3f}"
            ),
            (
                f"contact_possible={rel['contact_possible']:.3f} | "
                f"evidence=rel_speed:{rel['meta_gripper_obj_rel_speed']:.3f},d_grip_obj:{cur.gripper_object_distance:.3f}"
            ),
            (
                f"contact_confirmed={rel['contact_confirmed']:.3f} | "
                f"evidence=contact_signal_gripper:{cur.contact_signal_gripper_object:.3f}"
            ),
            (
                f"held_by_human={rel['held_by_human']:.3f} | "
                f"evidence=contact_signal_hand:{cur.contact_signal_hand_object:.3f},d_hand_obj:{cur.hand_object_distance:.3f}"
            ),
            (
                f"held_by_robot={rel['held_by_robot']:.3f} | "
                f"evidence=gripper_opening:{cur.gripper_opening:.3f},contact_confirmed:{rel['contact_confirmed']:.3f}"
            ),
            (
                f"human_released={rel['human_released']:.3f} | "
                f"evidence=held_by_human:{rel['held_by_human']:.3f}"
            ),
            (
                f"occluded(object)={rel['occluded_object']:.3f} | "
                f"evidence=visibility_score:{cur.visibility_score:.3f}"
            ),
            (
                f"conflict_or_uncertainty={conflict_score:.3f} | "
                f"evidence=occluded:{rel['occluded_object']:.3f},|contact_possible-contact_confirmed|:"
                f"{abs(float(rel['contact_possible'])-float(rel['contact_confirmed'])):.3f}"
            ),
            f"WINDOW_INFO: frames={len(records)}, duration_sec≈{len(records)*self._sim_dt:.2f}",
        ]
        return "\n".join(lines)

    def _predict_schema(self, obs) -> Dict[str, Any]:
        self._update_instruction_from_target(obs, schema_target=self._last_schema.get("TARGET", ""))
        image = self._current_image()
        structured_window = self._build_structured_window_summary(obs)
        pred = self._generate_schema(image, self._instruction, structured_window=structured_window)
        self._update_instruction_from_target(obs, schema_target=str(pred.get("TARGET", "")))
        if self._args.use_consistency_rerank:
            blank_pred = self._generate_schema(image, "", structured_window=structured_window)
            final = dict(pred)
            final["TARGET"] = str(pred.get("TARGET", "object_0"))
            final["PHASE"] = _vote(
                [str(pred.get("PHASE", "reach")), str(blank_pred.get("PHASE", "reach"))],
                default=str(pred.get("PHASE", "reach")),
            )
            final["SECURED"] = _vote(
                [str(pred.get("SECURED", "UNK")), str(blank_pred.get("SECURED", "UNK"))],
                default=str(pred.get("SECURED", "UNK")),
            )
            final["CONTACT"] = _vote(
                [str(pred.get("CONTACT", "UNK")), str(blank_pred.get("CONTACT", "UNK"))],
                default=str(pred.get("CONTACT", "UNK")),
            )
            final["OCCLUDED"] = _vote(
                [str(pred.get("OCCLUDED", "UNK")), str(blank_pred.get("OCCLUDED", "UNK"))],
                default=str(pred.get("OCCLUDED", "UNK")),
            )
            final["HUMAN_RELEASED"] = _vote(
                [str(pred.get("HUMAN_RELEASED", "UNK")), str(blank_pred.get("HUMAN_RELEASED", "UNK"))],
                default=str(pred.get("HUMAN_RELEASED", "UNK")),
            )
            try:
                c1 = float(pred.get("CONFLICT", 0.5))
            except Exception:
                c1 = 0.5
            try:
                c2 = float(blank_pred.get("CONFLICT", 0.5))
            except Exception:
                c2 = 0.5
            final["CONFLICT"] = float(np.clip(0.5 * c1 + 0.5 * c2, 0.0, 1.0))
            final["AFF"] = _vote_aff(self._PRIMITIVE_ORDER, [dict(pred.get("AFF", {}) or {}), dict(blank_pred.get("AFF", {}) or {})])
            if self._args.use_schema_checker:
                final = self._repair_schema_prediction(final, instruction=self._instruction, dataset="handover_sim")
            return final
        return pred

    def _extract_event_feat(self, obs) -> Dict[str, float]:
        target_pose7, target_key = self._target_object_pose7(obs, schema_target=self._last_schema.get("TARGET", ""))
        obj_pos = target_pose7[0:3]
        if self._prev_event_obj_center is not None:
            obj_speed = float(np.linalg.norm(obj_pos - self._prev_event_obj_center) / max(self._sim_dt, 1e-4))
        else:
            obj_speed = 0.0
        self._prev_event_obj_center = obj_pos.copy()
        finger_a = np.asarray(obs["panda_body"].link_state[0, 9, 0:3].numpy(), dtype=np.float32)
        finger_b = np.asarray(obs["panda_body"].link_state[0, 10, 0:3].numpy(), dtype=np.float32)
        grip_center = 0.5 * (finger_a + finger_b)
        mano_body = obs.get("mano_body", None)
        if mano_body is not None:
            hand_xyz = np.asarray(mano_body.link_state[0, :, 0:3].numpy(), dtype=np.float32)
            if hand_xyz.size > 0:
                hand_center = np.asarray(hand_xyz.mean(axis=0), dtype=np.float32)
                if self._prev_event_hand_center is not None:
                    hand_speed = float(np.linalg.norm(hand_center - self._prev_event_hand_center) / max(self._sim_dt, 1e-4))
                else:
                    hand_speed = 0.0
                self._prev_event_hand_center = hand_center.copy()
                d_hand_obj = float(np.min(np.linalg.norm(hand_xyz - obj_pos.reshape(1, 3), axis=1)))
                d_grip_hand = float(np.min(np.linalg.norm(hand_xyz - grip_center.reshape(1, 3), axis=1)))
            else:
                d_hand_obj = float(np.linalg.norm(grip_center - obj_pos))
                d_grip_hand = float("inf")
                hand_speed = 0.0
        else:
            d_hand_obj = float(np.linalg.norm(grip_center - obj_pos))
            d_grip_hand = float("inf")
            hand_speed = 0.0
        ordered = self._ordered_object_keys(obs)
        non_target_dists: List[float] = []
        for key in ordered:
            if key == target_key:
                continue
            body = obs["ycb_bodies"][key]
            other_pos = np.asarray(body.link_state[0, 6, 0:3].numpy(), dtype=np.float32)
            non_target_dists.append(float(np.linalg.norm(grip_center - other_pos)))
        d_grip_non_target = float(min(non_target_dists)) if non_target_dists else float("inf")
        return {
            "d_grip_obj": float(np.linalg.norm(grip_center - obj_pos)),
            "d_hand_obj": d_hand_obj,
            "d_grip_hand": d_grip_hand,
            "d_grip_non_target": d_grip_non_target,
            "hand_speed": hand_speed,
            "obj_speed": obj_speed,
            "contact_proxy": 1.0 if float(np.linalg.norm(grip_center - obj_pos)) < 0.05 else 0.0,
            "target_key": target_key,
        }

    def _is_near_handover(self, feat: Dict[str, float]) -> bool:
        return min(float(feat["d_grip_obj"]), float(feat["d_hand_obj"])) <= self._near_thresh

    def _apply_geometry_gate(self, schema: Dict[str, Any], obs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        feat = self._extract_event_feat(obs)
        d_grip = float(feat["d_grip_obj"])
        d_hand = float(feat["d_hand_obj"])
        contact_proxy = int(feat["contact_proxy"]) == 1
        if not self._geom_gate_enabled:
            return dict(schema), {
                "geom_gate_enabled": False,
                "geom_gate_applied": False,
                "geom_gate_changes": [],
                "geom_d_grip_obj": d_grip,
                "geom_d_hand_obj": d_hand,
                "geom_contact_proxy": int(contact_proxy),
            }

        out = dict(schema)
        changes: List[str] = []
        cur_phase = str(out.get("PHASE", "reach"))
        cur_secured = normalize_tri(out.get("SECURED", "UNK"))
        cur_contact = normalize_tri(out.get("CONTACT", "UNK"))

        # Hard physical guard: if far from object or no contact proxy, cannot be CONTACT=T.
        if cur_contact == "T" and (d_grip > self._geom_contact_max_dist or not contact_proxy):
            out["CONTACT"] = "F"
            changes.append("CONTACT:T->F")

        # Hard physical guard: if farther than secure threshold, cannot be SECURED=T/PHASE=secure.
        if cur_secured == "T" and (d_grip > self._geom_secure_max_dist or not contact_proxy):
            out["SECURED"] = "F"
            changes.append("SECURED:T->F")
        if cur_phase == "secure" and (d_grip > self._geom_secure_max_dist or not contact_proxy):
            out["PHASE"] = "align"
            changes.append("PHASE:secure->align")

        return out, {
            "geom_gate_enabled": True,
            "geom_gate_applied": bool(changes),
            "geom_gate_changes": changes,
            "geom_d_grip_obj": d_grip,
            "geom_d_hand_obj": d_hand,
            "geom_contact_proxy": int(contact_proxy),
        }

    def _compute_human_object_min_dist(self, obs) -> float:
        mano_body = obs.get("mano_body", None)
        if mano_body is None:
            return float("inf")
        obj_pose7, _ = self._target_object_pose7(obs, schema_target=self._last_schema.get("TARGET", ""))
        obj_pos = np.asarray(obj_pose7[0:3], dtype=np.float32).reshape(1, 3)
        mano_xyz = np.asarray(mano_body.link_state[0, :, 0:3].numpy(), dtype=np.float32)
        if mano_xyz.size == 0:
            return float("inf")
        d = np.linalg.norm(mano_xyz - obj_pos, axis=1)
        return float(np.min(d))

    def _is_human_ready(self, obs) -> Tuple[bool, float]:
        d = self._compute_human_object_min_dist(obs)
        if not self._human_ready_gate_enabled:
            return True, d
        return bool(d <= self._human_ready_max_dist), d

    def _smooth_action(self, current_cfg: np.ndarray, proposed_action: np.ndarray, obs) -> np.ndarray:
        if self._intervention_mode == "minimal":
            return np.asarray(proposed_action, dtype=np.float32).copy()
        out = np.asarray(proposed_action, dtype=np.float32).copy()
        base_step = max(1e-4, float(self._max_joint_step))
        d_grip = float(self._extract_event_feat(obs)["d_grip_obj"])
        step = base_step * float(self._near_step_scale) if d_grip <= self._slowdown_dist else base_step
        dq = np.clip(out[:7] - current_cfg[:7], -step, step)
        out[:7] = current_cfg[:7] + dq
        return out

    def _safe_wait_action(self, current_cfg: np.ndarray, obs) -> np.ndarray:
        # Stable wait: retreat in joint-space to a fixed safe home (start_conf),
        # avoiding Cartesian back-path sweeps that can hit objects on the table.
        frame_now = int(obs["frame"])
        if self._wait_action_cache is not None and (frame_now - int(self._wait_last_update_frame)) < self._wait_update_interval:
            action = np.asarray(self._wait_action_cache, dtype=np.float32).copy()
            action[7:9] = 0.04
            return action

        action = np.asarray(current_cfg, dtype=np.float32).copy()
        home = np.asarray(start_conf, dtype=np.float32).copy()
        step = float(self._wait_hover_max_joint_step)
        dq = np.clip(home[:7] - action[:7], -step, step)
        action[:7] = action[:7] + dq
        action[7:9] = 0.04
        self._wait_action_cache = np.asarray(action, dtype=np.float32).copy()
        self._wait_last_update_frame = frame_now
        return action

    def _update_grasp_lock(self, obs) -> None:
        frame = int(obs["frame"])
        d_grip = float(self._extract_event_feat(obs)["d_grip_obj"])
        yang_in_approach, yang_at_grasp = self._get_yang_grasp_flags(obs)
        yang_near = bool(yang_in_approach or yang_at_grasp)
        if not self._grasp_lock_active and (d_grip <= self._grasp_lock_enter_dist or yang_near):
            self._grasp_lock_active = True
            self._grasp_lock_until_frame = frame + self._grasp_lock_hold_frames
            return
        if self._grasp_lock_active:
            hold_elapsed = frame >= self._grasp_lock_until_frame
            far_enough = d_grip >= self._grasp_lock_exit_dist
            if hold_elapsed and far_enough:
                self._grasp_lock_active = False

    def _get_yang_grasp_flags(self, obs) -> Tuple[bool, bool]:
        ee_pose = self._get_ee_pose(obs)
        in_approach = False
        at_grasp = False
        if (
            getattr(self._in_approach_region, "_start_pose", None) is not None
            and getattr(self._in_approach_region, "_final_pose", None) is not None
        ):
            in_approach = bool(self._in_approach_region(ee_pose))
        if getattr(self._at_grasp_pose, "_goal_pose", None) is not None:
            at_grasp = bool(self._at_grasp_pose(ee_pose))
        return in_approach, at_grasp

    def _should_run_vlm(self, obs) -> Tuple[bool, str]:
        frame = int(obs["frame"])
        feat = self._extract_event_feat(obs)
        near = self._is_near_handover(feat)

        if self._vlm_calls >= self._max_vlm_calls:
            self._last_event_feat = feat
            return False, "budget_exhausted"

        if self._last_vlm_frame < 0:
            self._last_event_feat = feat
            if self._cold_start_mode == "always":
                return True, "cold_start"
            if self._cold_start_mode == "near" and near:
                return True, "cold_start_near"
            return False, "cold_start_skip"

        if (frame - self._last_vlm_frame) < self._min_vlm_gap:
            self._last_event_feat = feat
            return False, "cooldown_reuse"

        if (frame - self._last_vlm_frame) >= self._vlm_interval:
            self._last_event_feat = feat
            return True, "periodic_interval"

        if not self._enable_event_trigger:
            self._last_event_feat = feat
            return False, "reuse_cache"
        if not near:
            self._last_event_feat = feat
            return False, "far_from_handover"
        if self._last_event_feat is None:
            self._last_event_feat = feat
            return False, "reuse_cache"
        dgrip_jump = abs(feat["d_grip_obj"] - self._last_event_feat["d_grip_obj"])
        dhand_jump = abs(feat["d_hand_obj"] - self._last_event_feat["d_hand_obj"])
        contact_toggle = int(feat["contact_proxy"]) != int(self._last_event_feat["contact_proxy"])
        self._last_event_feat = feat
        if dgrip_jump >= self._event_dgrip_delta or dhand_jump >= self._event_hand_obj_delta or contact_toggle:
            return True, "event_trigger"
        return False, "reuse_cache"

    def _schema_decision(self, schema: Dict[str, Any], obs, run_vlm: bool) -> str:
        phase = str(schema.get("PHASE", "reach"))
        secured = normalize_tri(schema.get("SECURED", "UNK"))
        contact = normalize_tri(schema.get("CONTACT", "UNK"))
        occluded = normalize_tri(schema.get("OCCLUDED", "UNK"))
        released = normalize_tri(schema.get("HUMAN_RELEASED", "UNK"))
        try:
            conflict_val = float(schema.get("CONFLICT", 0.5))
        except Exception:
            conflict_val = 0.5
        conflict = float(np.clip(conflict_val, 0.0, 1.0))
        aff = dict(schema.get("AFF", {}) or {})
        frame = int(obs["frame"])
        rel = dict(self._last_relation_scores or {})
        rel_released = float(rel.get("human_released", 0.0))
        rel_held_human = float(rel.get("held_by_human", 0.0))
        rel_held_robot = float(rel.get("held_by_robot", 0.0))
        rel_contact = float(rel.get("contact_confirmed", 0.0))
        rel_occluded = float(rel.get("occluded_object", 0.0))
        released_signal = (
            released == "T"
            or rel_released >= self._release_commit_thresh
            or (rel_held_human <= 0.35 and rel_held_robot >= self._robot_hold_commit_thresh)
        )

        # Update secure-vote count only when a fresh VLM output is produced.
        if run_vlm:
            is_secure_vote = phase == "secure" and secured == "T" and contact == "T"
            self._secure_vote_count = self._secure_vote_count + 1 if is_secure_vote else 0
            self._occluded_vote_count = self._occluded_vote_count + 1 if occluded == "T" else 0
            self._release_vote_count = self._release_vote_count + 1 if released_signal else 0

        # Conservative uncertainty gate: high conflict/occlusion prevents commit.
        if conflict >= self._uncertainty_high_thresh or rel_occluded >= self._uncertainty_high_thresh or occluded == "T":
            _ = aff
            return "continue"

        # Commit when transfer evidence is stable and uncertainty is low.
        if (
            frame >= self._commit_min_frame
            and self._release_vote_count >= 1
            and released_signal
            and conflict <= self._uncertainty_low_thresh
            and rel_contact >= 0.5
            and rel_held_robot >= self._robot_hold_commit_thresh
            and contact in {"T", "UNK"}
            and secured in {"T", "UNK"}
            and phase in {"contact", "secure", "transfer"}
        ):
            return "commit"

        # Secondary strict commit path: explicit secure votes plus low uncertainty.
        if (
            self._secure_vote_count >= self._commit_vlm_votes
            and frame >= self._commit_min_frame
            and secured == "T"
            and contact == "T"
            and phase == "secure"
            and conflict <= self._uncertainty_low_thresh
        ):
            return "commit"
        if (
            frame >= self._force_commit_frame
            and self._secure_vote_count >= 1
            and secured == "T"
            and contact == "T"
            and phase == "secure"
            and conflict <= self._uncertainty_high_thresh
        ):
            return "commit"
        _ = (self._occluded_vote_count, aff)
        return "continue"

    def forward(self, obs):
        self._update_grasp_lock(obs)
        run_vlm, reason = self._should_run_vlm(obs)
        human_ready, human_obj_min_dist = self._is_human_ready(obs)
        frame_now = int(obs["frame"])
        gate_info: Dict[str, Any] = {}
        if run_vlm:
            schema = self._predict_schema(obs)
            gated_schema, gate_info = self._apply_geometry_gate(schema, obs)
            self._last_schema = gated_schema
            self._last_phase = str(gated_schema.get("PHASE", "reach"))
            self._last_decision = self._schema_decision(gated_schema, obs, run_vlm=True)
            self._last_vlm_frame = int(obs["frame"])
            self._vlm_calls += 1
            self._last_event_feat = self._extract_event_feat(obs)
        elif self._last_schema:
            gated_schema, gate_info = self._apply_geometry_gate(self._last_schema, obs)
            self._last_schema = gated_schema
            self._last_phase = str(gated_schema.get("PHASE", "reach"))
            self._last_decision = self._schema_decision(gated_schema, obs, run_vlm=False)

        decision = self._last_decision
        allow_action_override = (self._intervention_mode != "minimal")
        feat_now = self._extract_event_feat(obs)
        d_grip_now = float(feat_now["d_grip_obj"])
        d_hand_now = float(feat_now["d_hand_obj"])
        d_grip_hand_now = float(feat_now.get("d_grip_hand", 1e6))
        d_grip_non_target_now = float(feat_now.get("d_grip_non_target", 1e6))
        target_key_now = str(feat_now.get("target_key", ""))
        hand_speed_now = float(feat_now.get("hand_speed", 0.0))
        obj_speed_now = float(feat_now.get("obj_speed", 0.0))
        yang_in_approach, yang_at_grasp = self._get_yang_grasp_flags(obs)
        if hand_speed_now <= self._human_stable_max_speed:
            self._human_stable_counter += 1
        else:
            self._human_stable_counter = 0
        human_stable = (not self._human_stable_gate_enabled) or (self._human_stable_counter >= self._human_stable_min_frames)
        if human_ready and human_stable and (frame_now >= self._release_fallback_frame) and not self._done and decision != "commit":
            if (not self._force_grasp_latched) and d_grip_now <= self._force_grasp_latch_dist:
                self._force_grasp_latched = True
                self._force_grasp_latch_start = frame_now
            if d_grip_now <= self._force_grasp_dist:
                self._force_grasp_counter += 1
            else:
                self._force_grasp_counter = 0
            latch_age = frame_now - int(self._force_grasp_latch_start) if self._force_grasp_latched else -1
            if self._force_grasp_counter >= self._force_grasp_hold_frames or (
                self._force_grasp_latched and latch_age >= self._force_grasp_hold_frames
            ):
                decision = "commit"
                self._last_decision = decision
        else:
            self._force_grasp_counter = 0
            if not human_ready:
                self._force_grasp_latched = False
                self._force_grasp_latch_start = -1
        if decision != "commit" and not self._done and int(obs["frame"]) >= self._force_abort_frame:
            # Do not hard-abort by frame budget; let env timeout or geometric commit decide.
            decision = "continue"
            self._last_decision = "continue"

        close_stage_near = (
            human_ready
            and human_stable
            and (
                d_grip_now <= self._close_stage_enter_dist
                and d_hand_now <= (self._close_stage_enter_dist * 1.5)
                and (yang_in_approach or yang_at_grasp)
            )
            and (self._vlm_calls > 0)
        )
        if close_stage_near and not self._done and decision not in {"abort", "commit"}:
            self._close_stage_near_counter += 1
        else:
            self._close_stage_near_counter = 0

        if (
            not self._close_stage_active
            and self._close_stage_near_counter >= self._close_stage_hold_frames
            and not self._done
            and decision not in {"abort", "commit"}
        ):
            self._close_stage_active = True
            self._close_stage_start_frame = frame_now

        if self._close_stage_active and not self._done:
            close_age = frame_now - int(self._close_stage_start_frame)
            if close_age >= self._close_stage_max_frames:
                decision = "commit"
                self._last_decision = "commit"
                self._close_stage_active = False
                self._close_stage_near_counter = 0

        # Geometry-driven commit fallback to avoid staying in ALIGN forever.
        geom_commit_cond = (
            human_ready
            and human_stable
            and d_grip_now <= max(0.07, self._close_stage_enter_dist)
            and (yang_in_approach or yang_at_grasp)
            and frame_now >= self._commit_min_frame
        )
        if geom_commit_cond and not self._done and decision not in {"abort", "commit"}:
            self._geom_commit_counter += 1
        else:
            self._geom_commit_counter = 0
        if self._geom_commit_counter >= 20 and not self._done and decision != "abort":
            decision = "commit"
            self._last_decision = "commit"
            self._close_stage_active = False
            self._close_stage_near_counter = 0

        info = {
            "pred_phase": self._last_phase,
            "decision": decision,
            "pred_schema_text": self._render_schema_fields(self._last_schema) if self._last_schema else "",
            "vlm_called": bool(run_vlm),
            "vlm_interval": int(self._vlm_interval),
            "vlm_reason": reason,
            "vlm_calls_so_far": int(self._vlm_calls),
            "vlm_budget": int(self._max_vlm_calls),
            "geom_gate_applied": bool(gate_info.get("geom_gate_applied", False)),
            "geom_gate_changes": list(gate_info.get("geom_gate_changes", [])),
            "geom_d_grip_obj": float(gate_info.get("geom_d_grip_obj", -1.0)),
            "geom_d_hand_obj": float(gate_info.get("geom_d_hand_obj", -1.0)),
            "geom_contact_proxy": int(gate_info.get("geom_contact_proxy", -1)),
            "geom_d_grip_hand": float(d_grip_hand_now),
            "human_ready": bool(human_ready),
            "human_stable": bool(human_stable),
            "human_stable_counter": int(self._human_stable_counter),
            "human_stable_max_speed": float(self._human_stable_max_speed),
            "hand_speed": float(hand_speed_now),
            "obj_speed": float(obj_speed_now),
            "human_obj_min_dist": float(human_obj_min_dist),
            "grasp_lock_active": bool(self._grasp_lock_active),
            "grasp_lock_until_frame": int(self._grasp_lock_until_frame),
            "force_grasp_counter": int(self._force_grasp_counter),
            "force_grasp_dist": float(self._force_grasp_dist),
            "force_grasp_latched": bool(self._force_grasp_latched),
            "force_grasp_latch_start": int(self._force_grasp_latch_start),
            "close_stage_near_counter": int(self._close_stage_near_counter),
            "close_stage_active": bool(self._close_stage_active),
            "close_stage_start_frame": int(self._close_stage_start_frame),
            "yang_in_approach": bool(yang_in_approach),
            "yang_at_grasp": bool(yang_at_grasp),
            "occluded_vote_count": int(self._occluded_vote_count),
            "geom_commit_counter": int(self._geom_commit_counter),
        }

        current_cfg = obs["panda_body"].dof_state[0, :, 0].numpy().copy()
        if self._prev_cfg is None:
            self._prev_cfg = current_cfg.copy()
        joint_motion = float(np.linalg.norm(current_cfg[:7] - self._prev_cfg[:7]))
        near_for_unstick = d_grip_now <= max(self._close_stage_enter_dist * 1.8, 0.16)
        if near_for_unstick and joint_motion <= 1e-4 and not self._done and decision not in {"abort", "commit"}:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
        self._prev_cfg = current_cfg.copy()
        if self._stuck_counter >= 45 and human_ready and not self._done and decision not in {"abort", "commit"}:
            if not self._close_stage_active:
                self._close_stage_active = True
                self._close_stage_start_frame = frame_now
            info["decision"] = "unstick_to_close"
        info["stuck_counter"] = int(self._stuck_counter)
        info["joint_motion"] = float(joint_motion)

        released = normalize_tri(self._last_schema.get("HUMAN_RELEASED", "UNK")) if self._last_schema else "UNK"
        phase_now = str(self._last_schema.get("PHASE", "reach")) if self._last_schema else "reach"
        try:
            conflict_now = float(self._last_schema.get("CONFLICT", 0.5)) if self._last_schema else 0.5
        except Exception:
            conflict_now = 0.5
        rel_released = float((self._last_relation_scores or {}).get("human_released", 0.0))
        release_confirmed = bool(released == "T" or rel_released >= self._release_commit_thresh)
        release_fallback_ready = bool(
            human_stable and obj_speed_now <= self._motion_lock_obj_speed_max and hand_speed_now <= self._human_stable_max_speed
        )
        if release_fallback_ready:
            self._release_fallback_stable_counter += 1
        else:
            self._release_fallback_stable_counter = 0
        release_gate_open = bool(
            release_confirmed
            or (
                frame_now >= self._release_fallback_frame
            )
        )
        if (not release_confirmed) and decision == "commit":
            decision = "continue"
            self._last_decision = "continue"
        moving_handover_object = (
            (d_hand_now <= self._handover_hand_obj_near_dist)
            and (hand_speed_now > self._human_stable_max_speed)
        ) or (obj_speed_now > max(self._motion_lock_obj_speed_max, 0.04))
        hard_block_moving_approach = bool(
            moving_handover_object
            and d_grip_now <= self._no_approach_moving_dist
        )
        if (not release_gate_open) and moving_handover_object:
            self._motion_lock_active = True
            self._motion_lock_stable_counter = 0
        if self._motion_lock_active and (not release_gate_open):
            stable_unlock = (
                hand_speed_now <= (0.7 * self._human_stable_max_speed)
                and obj_speed_now <= self._motion_lock_obj_speed_max
            )
            if stable_unlock:
                self._motion_lock_stable_counter += 1
            else:
                self._motion_lock_stable_counter = 0
            if self._motion_lock_stable_counter >= self._motion_lock_release_stable_frames:
                self._motion_lock_active = False
                self._motion_lock_stable_counter = 0
        if release_gate_open:
            self._motion_lock_active = False
            self._motion_lock_stable_counter = 0
        schema_reacquire_active = (
            (not self._done)
            and (not release_gate_open)
            and (
                self._motion_lock_active
                or ((not human_stable) and d_hand_now <= self._handover_hand_obj_near_dist)
            )
            and moving_handover_object
            and d_grip_now <= self._approach_keepout_dist
            and frame_now >= self._commit_min_frame
        )
        approach_keepout_active = (
            (not self._done)
            and (not release_gate_open)
            and self._motion_lock_active
            and (d_grip_now <= self._approach_keepout_dist)
            and moving_handover_object
            and frame_now >= self._commit_min_frame
        )
        info["moving_handover_object"] = bool(moving_handover_object)
        info["hard_block_moving_approach"] = bool(hard_block_moving_approach)
        info["no_approach_moving_dist"] = float(self._no_approach_moving_dist)
        info["schema_phase_now"] = str(phase_now)
        info["schema_conflict_now"] = float(conflict_now)
        info["target_key"] = target_key_now
        info["d_grip_non_target"] = d_grip_non_target_now
        info["non_target_keepout_dist"] = float(self._non_target_keepout_dist)
        info["schema_reacquire_active"] = bool(schema_reacquire_active)
        info["motion_lock_active"] = bool(self._motion_lock_active)
        info["motion_lock_stable_counter"] = int(self._motion_lock_stable_counter)
        info["motion_lock_release_stable_frames"] = int(self._motion_lock_release_stable_frames)
        info["motion_lock_obj_speed_max"] = float(self._motion_lock_obj_speed_max)
        info["approach_keepout_active"] = bool(approach_keepout_active)
        info["approach_keepout_dist"] = float(self._approach_keepout_dist)
        info["handover_hand_obj_near_dist"] = float(self._handover_hand_obj_near_dist)
        info["pre_release_keepout_dist"] = float(self._pre_release_keepout_dist)
        if (not self._done) and hard_block_moving_approach:
            action = self._safe_wait_action(current_cfg, obs)
            self._force_grasp_counter = 0
            self._force_grasp_latched = False
            self._force_grasp_latch_start = -1
            self._near_contact_commit_counter = 0
            self._close_stage_active = False
            self._close_stage_near_counter = 0
            self._geom_commit_counter = 0
            self._action_repeat = None
            info["decision"] = "hard_block_hand_moving"
            info["wait_human"] = True
            return action, info

        if allow_action_override and (approach_keepout_active or schema_reacquire_active):
            action = self._safe_wait_action(current_cfg, obs)
            self._force_grasp_counter = 0
            self._force_grasp_latched = False
            self._force_grasp_latch_start = -1
            self._near_contact_commit_counter = 0
            self._close_stage_active = False
            self._close_stage_near_counter = 0
            self._geom_commit_counter = 0
            self._action_repeat = None
            info["decision"] = "schema_reacquire_wait"
            info["wait_human"] = True
            return action, info
        if allow_action_override and (not self._done) and (not release_gate_open) and moving_handover_object and d_grip_now <= self._pre_release_keepout_dist:
            action = self._safe_wait_action(current_cfg, obs)
            self._force_grasp_counter = 0
            self._force_grasp_latched = False
            self._force_grasp_latch_start = -1
            self._near_contact_commit_counter = 0
            self._close_stage_active = False
            self._close_stage_near_counter = 0
            self._geom_commit_counter = 0
            self._action_repeat = None
            info["decision"] = "pre_release_keepout"
            info["wait_human"] = True
            return action, info
        hand_keepout_active = (
            (not self._done)
            and (not release_gate_open)
            and d_grip_hand_now <= self._safety_hand_keepout_dist
        )
        info["release_confirmed"] = bool(release_confirmed)
        info["release_gate_open"] = bool(release_gate_open)
        info["release_fallback_frame"] = int(self._release_fallback_frame)
        info["release_fallback_stable_counter"] = int(self._release_fallback_stable_counter)
        info["hand_keepout_active"] = bool(hand_keepout_active)
        info["safety_hand_keepout_dist"] = float(self._safety_hand_keepout_dist)
        if allow_action_override and hand_keepout_active:
            action = self._safe_wait_action(current_cfg, obs)
            info["decision"] = "safety_wait_release"
            info["wait_human"] = True
            return action, info

        near_contact_commit_cond = (
            (not self._done)
            and release_gate_open
            and decision not in {"abort", "commit"}
            and frame_now >= self._commit_min_frame
            and d_grip_now <= self._near_contact_commit_dist
            and bool(feat_now.get("contact_proxy", 0.0) >= 0.5)
            and (human_ready and human_stable)
        )
        if near_contact_commit_cond:
            self._near_contact_commit_counter += 1
        else:
            self._near_contact_commit_counter = 0
        info["near_contact_commit_counter"] = int(self._near_contact_commit_counter)
        info["near_contact_commit_dist"] = float(self._near_contact_commit_dist)
        if self._near_contact_commit_counter >= self._near_contact_commit_hold_frames:
            decision = "commit"
            self._last_decision = "commit"
            self._close_stage_active = False
            self._close_stage_near_counter = 0
            info["decision"] = "near_contact_commit"

        if (
            (not self._done)
            and (decision != "abort")
            and (self._hard_force_commit_frame > 0)
            and (frame_now >= self._hard_force_commit_frame)
        ):
            decision = "commit"
            self._last_decision = "commit"
            self._close_stage_active = False
            self._close_stage_near_counter = 0
            info["decision"] = "hard_force_commit_window"

        non_target_keepout_active = (
            self._enable_non_target_keepout
            and (not self._done)
            and (not release_gate_open)
            and (d_grip_non_target_now <= self._non_target_keepout_dist)
        )
        info["non_target_keepout_active"] = bool(non_target_keepout_active)
        if allow_action_override and non_target_keepout_active:
            action = self._safe_wait_action(current_cfg, obs)
            self._action_repeat = None
            info["decision"] = "non_target_keepout_wait"
            info["wait_human"] = True
            return action, info

        should_wait_human = (
            ((not human_ready) or (not human_stable))
            and (self._human_ready_gate_enabled or self._human_stable_gate_enabled)
            and (int(obs["frame"]) < self._human_ready_max_wait_frame)
            and (not self._force_grasp_latched)
            and (not self._done)
        )
        if allow_action_override and should_wait_human:
            # Do not pre-grasp before human hand approaches the object.
            action = self._safe_wait_action(current_cfg, obs)
            info["decision"] = "wait_human"
            info["wait_human"] = True
            return action, info

        if allow_action_override and decision == "abort" and not self._done:
            self._policy_aborted = True
            self._abort_reason = "vlm_schema_gate"
            action = current_cfg.copy()
            action[7:9] = 0.04
            info["policy_abort"] = True
            info["abort_reason"] = self._abort_reason
            return action, info

        # In minimal mode, do not intervene in low-level control; keep native Yang behavior.
        if self._intervention_mode == "minimal":
            if decision == "commit" and not self._done:
                self._done = True
                self._done_frame = None
                self._back = None
                self._close_stage_active = False
            if (not self._done) and (getattr(self, "_action_repeat", None) is None):
                if self._current_grasps is None:
                    self._current_grasps = self._load_grasps(obs)
                current_cfg_boot = self._get_current_cfg(obs)
                object_pose_boot = self._get_object_pose(obs)
                ee_pose_boot = self._get_ee_pose(obs)
                boot_action = self._get_reactive_policy_action(
                    current_cfg_boot,
                    object_pose_boot,
                    ee_pose_boot,
                )
                self._action_repeat = np.asarray(boot_action, dtype=np.float32).copy()
            action, base_info = super().forward(obs)
            if not self._done:
                near_contact = (d_grip_now <= self._assist_close_dist) and bool(feat_now.get("contact_proxy", 0.0) >= 0.5)
                if near_contact and (human_ready or release_confirmed):
                    self._assist_close_active = True
                    self._assist_close_until_frame = max(
                        self._assist_close_until_frame,
                        frame_now + self._assist_close_hold_frames,
                    )
                if self._assist_close_active:
                    if frame_now <= self._assist_close_until_frame:
                        action = np.asarray(action, dtype=np.float32).copy()
                        action[7:9] = 0.0
                    elif d_grip_now >= self._assist_close_release_dist:
                        self._assist_close_active = False
                info["assist_close_active"] = bool(self._assist_close_active)
                info["assist_close_until_frame"] = int(self._assist_close_until_frame)
                info["assist_close_dist"] = float(self._assist_close_dist)
                info["assist_close_release_dist"] = float(self._assist_close_release_dist)
            if base_info:
                info.update(base_info)
            return action, info

        if self._close_stage_active and decision != "abort" and not self._done:
            if self._current_grasps is None:
                self._current_grasps = self._load_grasps(obs)
            object_pose = self._get_object_pose(obs)
            ee_pose = self._get_ee_pose(obs)
            action = self._get_reactive_policy_action(current_cfg, object_pose, ee_pose)
            self._action_repeat = np.asarray(action, dtype=np.float32).copy()
            action = self._smooth_action(current_cfg, action, obs)
            action[7:9] = 0.0
            info["decision"] = "close_stage_push"
            return action, info

        if decision == "commit" and not self._done:
            # Mark done and let Yang base policy run its native close-gripper + retreat sequence.
            self._done = True
            self._done_frame = None
            self._back = None
            self._close_stage_active = False

        # Yang policy assumes action_repeat has been initialized by periodic planning.
        # When policy_time_wait=0 and the first frame does not align with repeat stride,
        # _action_repeat can still be None and crash. Initialize it lazily here.
        if decision == "continue" and not self._done and getattr(self, "_action_repeat", None) is None:
            if self._current_grasps is None:
                self._current_grasps = self._load_grasps(obs)
            current_cfg = self._get_current_cfg(obs)
            object_pose = self._get_object_pose(obs)
            ee_pose = self._get_ee_pose(obs)
            boot_action = self._get_reactive_policy_action(current_cfg, object_pose, ee_pose)
            self._action_repeat = boot_action.copy()
            boot_action = self._smooth_action(current_cfg, boot_action, obs)
            return boot_action, info

        # Near-object hysteresis lock to avoid rapid re-planning oscillation.
        if decision == "continue" and not self._done and self._grasp_lock_active:
            if getattr(self, "_action_repeat", None) is not None:
                lock_action = np.asarray(self._action_repeat, dtype=np.float32).copy()
            else:
                if self._current_grasps is None:
                    self._current_grasps = self._load_grasps(obs)
                object_pose = self._get_object_pose(obs)
                ee_pose = self._get_ee_pose(obs)
                lock_action = self._get_reactive_policy_action(current_cfg, object_pose, ee_pose)
                self._action_repeat = lock_action.copy()
            lock_action = self._smooth_action(current_cfg, lock_action, obs)
            info["decision"] = "lock_follow"
            return lock_action, info

        action, base_info = super().forward(obs)
        action = self._smooth_action(current_cfg, action, obs)
        if base_info:
            info.update(base_info)
        return action, info


def run_scene_with_policy(
    runner: BenchmarkRunner,
    idx: int,
    policy: VLMYangPolicy,
    save_video: bool = False,
    video_path: Optional[Path] = None,
    video_fps: int = 20,
    video_every_n: int = 1,
    video_max_frames: int = 0,
) -> Dict[str, Any]:
    obs = runner._env.reset(idx=idx)
    policy._runner_env = runner._env
    policy.reset()
    elapsed = []
    policy_trace = []
    action_trace = []
    vlm_calls = 0
    writer = None
    recorded_frames = 0
    if save_video:
        import imageio.v2 as imageio

        if video_path is None:
            raise ValueError("video_path must be provided when save_video=True")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(video_path), fps=max(1, int(video_fps)))

    def _finalize(result_dict: Dict[str, Any]) -> Dict[str, Any]:
        if writer is not None:
            writer.close()
            result_dict["video_path"] = str(video_path)
        return result_dict

    while True:
        (action, info), dt = runner._run_policy(policy, obs)
        if "obs_time" in info:
            dt -= info["obs_time"]
        action_trace.append(np.asarray(action, dtype=np.float32))
        elapsed.append(float(dt))
        policy_trace.append(
            {
                "frame": int(obs["frame"]),
                "pred_phase": str(info.get("pred_phase", "")),
                "decision": str(info.get("decision", "")),
                "pred_schema_text": str(info.get("pred_schema_text", "")),
                "vlm_called": bool(info.get("vlm_called", False)),
                "vlm_reason": str(info.get("vlm_reason", "")),
                "policy_abort": bool(info.get("policy_abort", False)),
                "abort_reason": str(info.get("abort_reason", "")),
            }
        )
        if bool(info.get("vlm_called", False)):
            vlm_calls += 1
        if writer is not None and (int(obs["frame"]) % max(1, int(video_every_n)) == 0):
            if int(video_max_frames) <= 0 or recorded_frames < int(video_max_frames):
                frame_rgb = policy._current_image()
                writer.append_data(frame_rgb)
                recorded_frames += 1
        if bool(info.get("policy_abort", False)):
            return _finalize(
                {
                "elapsed_time": np.asarray(elapsed),
                "elapsed_frame": int(obs["frame"]),
                "result": 0,
                "policy_abort": True,
                "abort_reason": str(info.get("abort_reason", "")),
                "last_pred_phase": str(info.get("pred_phase", "")),
                "policy_trace": policy_trace,
                "vlm_calls": int(vlm_calls),
                }
            )
        obs, _, _, env_info = runner._env.step(action)
        if env_info["status"] != 0:
            return _finalize(
                {
                "elapsed_time": np.asarray(elapsed),
                "elapsed_frame": int(runner._env.frame),
                "result": int(env_info["status"]),
                "policy_abort": False,
                "abort_reason": "",
                "last_pred_phase": policy._last_phase,
                "policy_trace": policy_trace,
                "vlm_calls": int(vlm_calls),
                }
            )


def main() -> None:
    args = parse_args()
    cfg = default_cfg.clone()
    cfg.defrost()
    cfg.ENV.ID = args.env_id
    cfg.BENCHMARK.CONTACT_FORCE_THRESH = float(args.contact_force_thresh)
    cfg.ENV.RELEASE_FORCE_THRESH = float(args.release_force_thresh)
    cfg.ENV.MANO_LINK_RESTITUTION = float(args.mano_restitution)
    cfg.SIM.RENDER = False
    cfg.ENV.RENDER_OFFSCREEN = True
    cfg.BENCHMARK.SETUP = args.setup
    cfg.BENCHMARK.SPLIT = args.split
    cfg.BENCHMARK.SAVE_RESULT = False
    cfg.freeze()

    runner = BenchmarkRunner(cfg)
    policy = VLMYangPolicy(cfg, args)
    res_dir = Path(args.res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    if args.scene_indices:
        indices = [int(x.strip()) for x in args.scene_indices.split(",") if x.strip()]
    else:
        indices = [args.scene_index]

    results = []
    for idx in indices:
        scene_id = runner._env.scene_ids[idx]
        print(f"{idx + 1:04d}/{runner._env.num_scenes:04d}: scene {scene_id}")
        scene_video_path = res_dir / f"scene_{idx:04d}.mp4"
        result = run_scene_with_policy(
            runner,
            idx,
            policy,
            save_video=bool(args.save_video),
            video_path=scene_video_path,
            video_fps=int(args.video_fps),
            video_every_n=max(1, int(args.video_every_n)),
            video_max_frames=max(0, int(args.video_max_frames)),
        )
        status = int(result["result"])
        print(f"time:   {float(np.sum(result['elapsed_time'])):6.2f}")
        print(f"frame:  {int(result['elapsed_frame']):5d}")
        if result["policy_abort"]:
            print("result:  policy_abort")
        elif status == EpisodeStatus.SUCCESS:
            print("result:  success")
        else:
            f1 = bool(status & EpisodeStatus.FAILURE_HUMAN_CONTACT == EpisodeStatus.FAILURE_HUMAN_CONTACT)
            f2 = bool(status & EpisodeStatus.FAILURE_OBJECT_DROP == EpisodeStatus.FAILURE_OBJECT_DROP)
            f3 = bool(status & EpisodeStatus.FAILURE_TIMEOUT == EpisodeStatus.FAILURE_TIMEOUT)
            print(f"result:  failure {int(f1)} {int(f2)} {int(f3)}")

        decision_trace = result["policy_trace"]
        decision_counts = {
            "continue": int(sum(1 for e in decision_trace if str(e.get("decision", "")) == "continue")),
            "commit": int(sum(1 for e in decision_trace if str(e.get("decision", "")) == "commit")),
            "abort": int(sum(1 for e in decision_trace if str(e.get("decision", "")) == "abort")),
        }
        first_commit_frame = next((int(e["frame"]) for e in decision_trace if str(e.get("decision", "")) == "commit"), -1)
        first_abort_frame = next((int(e["frame"]) for e in decision_trace if str(e.get("decision", "")) == "abort"), -1)

        results.append(
            {
                "scene_index": idx,
                "scene_id": int(scene_id),
                "elapsed_time_sec": float(np.sum(result["elapsed_time"])),
                "elapsed_frame": int(result["elapsed_frame"]),
                "status": status,
                "success": bool(status == EpisodeStatus.SUCCESS),
                "failure_human_contact": bool(status & EpisodeStatus.FAILURE_HUMAN_CONTACT == EpisodeStatus.FAILURE_HUMAN_CONTACT),
                "failure_object_drop": bool(status & EpisodeStatus.FAILURE_OBJECT_DROP == EpisodeStatus.FAILURE_OBJECT_DROP),
                "failure_timeout": bool(status & EpisodeStatus.FAILURE_TIMEOUT == EpisodeStatus.FAILURE_TIMEOUT),
                "policy_abort": bool(result["policy_abort"]),
                "abort_reason": str(result["abort_reason"]),
                "last_pred_phase": str(result["last_pred_phase"]),
                "vlm_calls": int(result.get("vlm_calls", 0)),
                "decision_counts": decision_counts,
                "first_commit_frame": int(first_commit_frame),
                "first_abort_frame": int(first_abort_frame),
                "decision_trace_preview": decision_trace[:30],
                "video_path": str(result.get("video_path", "")),
            }
        )

    summary = {
        "method": "vlm_full_rollout",
        "setup": args.setup,
        "split": args.split,
        "env_id": args.env_id,
        "scene_indices": indices,
        "num_scenes": len(indices),
        "success_rate": float(np.mean([x["success"] for x in results])) if results else 0.0,
        "policy_abort_rate": float(np.mean([x["policy_abort"] for x in results])) if results else 0.0,
        "timeout_rate": float(np.mean([x["failure_timeout"] for x in results])) if results else 0.0,
        "vlm_calls_total": int(sum(int(r.get("vlm_calls", 0)) for r in results)),
        "results": results,
    }
    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
