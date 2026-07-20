from __future__ import annotations

from typing import Any, Dict, List

from linguistic_blindness.benchmark.perturbations import infer_state_from_phase
from linguistic_blindness.benchmark.schema import InteractionSchema
from linguistic_blindness.models.base import BaseSchemaModel
from linguistic_blindness.utils.text import extract_target, has_prohibition, is_blank_instruction


class StructuredParser(BaseSchemaModel):
    """Deterministic lightweight parser for the MVP offline benchmark.

    It is intentionally simple: it predicts schema fields from instruction text,
    scene objects, and replay state flags. This is a baseline and not a learned
    result.
    """

    name = "structured_parser"

    def predict(self, example: Dict[str, Any]) -> Dict[str, Any]:
        instruction = str(example.get("instruction", ""))
        scene_objects = [str(x).lower() for x in example.get("scene_objects", [])]
        obs_state = example.get("observation_state", {}) or {}
        phase = str(obs_state.get("phase") or "UNK")
        state = infer_state_from_phase(phase, str(example.get("dataset", "")))
        target = extract_target(instruction, scene_objects) or extract_target(instruction)
        target_exists = bool(target and target in scene_objects)
        allowed: List[str] = ["MOVE_TO", "GRASP", "PICK", "PLACE", "WAIT", "ASK"]
        blocked: List[str] = []
        reason = "parsed from instruction and replay state"

        if is_blank_instruction(instruction):
            next_action = "WAIT"  # intentionally conservative baseline
            allowed = ["ASK", "PROMPT", "WAIT", "HOLD", "ABORT", "BACKOFF"]
            blocked = ["GRASP", "PICK", "PLACE", "RETRACT", "MOVE_TO", "TRANSFER"]
            target = None
            target_exists = False
            reason = "blank instruction"
        elif has_prohibition(instruction):
            next_action = "WAIT"
            allowed = ["ASK", "PROMPT", "WAIT", "HOLD", "ABORT", "BACKOFF"]
            blocked = ["GRASP", "PICK", "RETRACT", "TRANSFER"]
            reason = "prohibition detected"
        elif not target_exists:
            next_action = "MOVE_TO"  # deliberately exposes hallucination risk before checker/gate
            reason = "target parsed but not found"
        elif state.get("human_released") is False and phase in {"contact", "transfer"}:
            next_action = "RETRACT"  # deliberately exposes safety risk before checker/gate
            blocked = []
            reason = "unsafe naive handover action"
        elif phase in {"grasp", "manipulate", "contact"}:
            next_action = "GRASP"
        elif phase in {"place", "done", "secure"}:
            next_action = "PLACE" if phase == "place" else "HOLD"
        else:
            next_action = "MOVE_TO"

        return InteractionSchema(
            target_object=target,
            target_exists=target_exists,
            phase=phase,
            human_contact=state.get("human_contact"),
            human_released=state.get("human_released"),
            robot_contact=state.get("robot_contact"),
            robot_grasp_stable=state.get("robot_grasp_stable"),
            allowed_actions=allowed,
            blocked_actions=blocked,
            next_action=next_action,
            reason=reason,
            confidence=0.75,
        ).to_dict()


class BlindStructuredParser(StructuredParser):
    """A diagnostic baseline that ignores counterfactual language target changes."""

    name = "blind_structured_parser"

    def predict(self, example: Dict[str, Any]) -> Dict[str, Any]:
        patched = dict(example)
        patched["instruction"] = example.get("original_instruction", example.get("instruction", ""))
        pred = super().predict(patched)
        pred["reason"] = "uses original instruction; diagnostic blind baseline"
        return pred
