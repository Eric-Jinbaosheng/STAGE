from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from linguistic_blindness.utils.text import normalize_action

PHASES = {
    "UNK", "reach", "align", "contact", "transfer", "secure", "retract", "failure",
    "grasp", "manipulate", "place", "done", "offer", "wait",
}

REQUIRED_FIELDS = [
    "target_object", "target_exists", "phase", "human_contact", "human_released", "robot_contact",
    "robot_grasp_stable", "allowed_actions", "blocked_actions", "next_action", "reason", "confidence",
]


@dataclass
class InteractionSchema:
    target_object: Optional[str] = None
    target_exists: bool = True
    phase: str = "UNK"
    human_contact: Optional[bool] = None
    human_released: Optional[bool] = None
    robot_contact: Optional[bool] = None
    robot_grasp_stable: Optional[bool] = None
    allowed_actions: List[str] = field(default_factory=list)
    blocked_actions: List[str] = field(default_factory=list)
    next_action: str = "WAIT"
    reason: str = ""
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_object": self.target_object,
            "target_exists": bool(self.target_exists),
            "phase": self.phase,
            "human_contact": self.human_contact,
            "human_released": self.human_released,
            "robot_contact": self.robot_contact,
            "robot_grasp_stable": self.robot_grasp_stable,
            "allowed_actions": list(self.allowed_actions),
            "blocked_actions": list(self.blocked_actions),
            "next_action": self.next_action,
            "reason": self.reason,
            "confidence": self.confidence,
        }


def normalize_schema(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    raw = dict(payload or {})
    phase = str(raw.get("phase", raw.get("PHASE", "UNK")) or "UNK")
    if phase not in PHASES:
        phase = "UNK"
    out = InteractionSchema(
        target_object=_none_if_empty(raw.get("target_object", raw.get("TARGET"))),
        target_exists=bool(raw.get("target_exists", True)),
        phase=phase,
        human_contact=_tri_bool(raw.get("human_contact", raw.get("CONTACT"))),
        human_released=_tri_bool(raw.get("human_released", raw.get("HUMAN_RELEASED"))),
        robot_contact=_tri_bool(raw.get("robot_contact", raw.get("CONTACT"))),
        robot_grasp_stable=_tri_bool(raw.get("robot_grasp_stable", raw.get("SECURED"))),
        allowed_actions=[normalize_action(x) for x in _as_list(raw.get("allowed_actions", []))],
        blocked_actions=[normalize_action(x) for x in _as_list(raw.get("blocked_actions", []))],
        next_action=normalize_action(raw.get("next_action", raw.get("NEXT_ACTION", "WAIT"))),
        reason=str(raw.get("reason", "")),
        confidence=_float_or_none(raw.get("confidence", raw.get("CONFLICT"))),
    )
    return out.to_dict()


def schema_validity_errors(schema: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for field_name in REQUIRED_FIELDS:
        if field_name not in schema:
            errors.append(f"missing_{field_name}")
    if str(schema.get("phase", "UNK")) not in PHASES:
        errors.append("illegal_phase")
    if not isinstance(schema.get("allowed_actions", []), list):
        errors.append("allowed_actions_not_list")
    if not isinstance(schema.get("blocked_actions", []), list):
        errors.append("blocked_actions_not_list")
    return errors


def _none_if_empty(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.upper() in {"UNK", "NONE", "NULL"}:
        return None
    return text.lower()


def _tri_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().upper()
    if text in {"T", "TRUE", "1", "YES"}:
        return True
    if text in {"F", "FALSE", "0", "NO"}:
        return False
    return None


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        if not value.strip():
            return []
        return [x.strip() for x in value.split(",") if x.strip()]
    return [value]


def _float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    if out > 1.0 and out <= 100.0:
        out = out / 100.0
    return max(0.0, min(1.0, out))
