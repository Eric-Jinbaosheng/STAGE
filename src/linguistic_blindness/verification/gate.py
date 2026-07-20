from __future__ import annotations

from typing import Any, Dict

from linguistic_blindness.utils.text import is_blank_instruction, normalize_action


def gate_action(example: Dict[str, Any], schema: Dict[str, Any], flags: Dict[str, Any], confidence_threshold: float = 0.0) -> Dict[str, Any]:
    next_action = normalize_action(schema.get("next_action"))
    confidence = schema.get("confidence")
    low_conf = confidence is not None and float(confidence) < confidence_threshold
    reason = "preserved"
    gated = next_action

    if flags.get("safety_violation") or flags.get("release_block_missing"):
        gated, reason = "HOLD", "safety conflict"
    elif flags.get("impossible_execution"):
        gated, reason = "TARGET_NOT_FOUND", "impossible target"
    elif flags.get("blind_execution") or is_blank_instruction(example.get("instruction", "")):
        gated, reason = "ASK", "blank or underspecified instruction"
    elif flags.get("negation_failure"):
        gated, reason = "WAIT", "prohibition detected"
    elif flags.get("blocked_action_violation") or flags.get("action_constraint_inconsistency"):
        gated, reason = "WAIT", "action violates schema constraints"
    elif flags.get("phase_state_inconsistent"):
        gated, reason = "HOLD", "phase-state inconsistency"
    elif low_conf:
        gated, reason = "ASK", "low confidence"

    return {
        "final_gated_action": gated,
        "gate_changed_action": gated != next_action,
        "gate_reason": reason,
    }
