import json
from pathlib import Path
from typing import Dict, List, Tuple

from jsonschema import Draft202012Validator


TRI_TRUE = True
TRI_FALSE = False
TRI_UNKNOWN = "unknown"


def _is_true(v) -> bool:
    return v is True


def _is_false(v) -> bool:
    return v is False


def _load_schema(schema_path: Path) -> Dict:
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _get_affordance_names(instance: Dict) -> List[str]:
    names = []
    for a in instance.get("affordances", []):
        name = a.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def validate_schema_only(instance: Dict, schema: Dict) -> Tuple[bool, List[str]]:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
    messages = [f"{'/'.join(map(str, e.path))}: {e.message}" for e in errors]
    return len(messages) == 0, messages


def semantic_check(instance: Dict) -> Tuple[List[str], List[str]]:
    violations: List[str] = []
    blocked: List[str] = []

    state = instance.get("state", {})
    affordance_names = _get_affordance_names(instance)
    chosen = instance.get("next_action", {}).get("chosen")

    # Invariant 1: if not released, disallow release-dependent motions.
    if _is_false(state.get("human_released")):
        blocked.extend(["RETRACT", "LIFT", "PULL_TEST"])

    # Invariant 2: no close gentle when contact is not confirmed.
    if not _is_true(state.get("contact_confirmed")):
        blocked.append("CLOSE_GENTLE")
        if chosen == "CLOSE_GENTLE":
            violations.append("contact_not_confirmed_but_close_gentle")

    # Invariant 3: occlusion should favor conservative actions.
    if _is_true(state.get("occluded")):
        blocked.append("FAST_APPROACH")
        if chosen not in {"VIEWPOINT_CHANGE", "HOLD_COMPLIANT", "BACKOFF_SMALL"}:
            violations.append("occluded_but_non_conservative_action")

    # Invariant 4: no retract before object secured.
    if not _is_true(state.get("object_secured")):
        blocked.append("RETRACT")

    if chosen is not None and chosen not in affordance_names:
        violations.append("next_action_not_in_affordances")

    # Optional semantic consistency: if contact unknown, avoid strict closure.
    if state.get("contact_confirmed") == TRI_UNKNOWN and chosen == "CLOSE_GENTLE":
        violations.append("contact_unknown_but_close_gentle")

    allowed_actions = [a for a in affordance_names if a not in set(blocked)]
    return violations, allowed_actions


def run_checker(instance: Dict, schema_path: str) -> Dict:
    schema = _load_schema(Path(schema_path))
    ok_schema, schema_errors = validate_schema_only(instance, schema)
    violations: List[str] = []
    allowed_actions: List[str] = []
    if ok_schema:
        violations, allowed_actions = semantic_check(instance)

    is_valid = ok_schema and not violations
    if not ok_schema:
        error_type = "STRUCTURE"
    elif violations:
        error_type = "SEMANTIC"
    else:
        error_type = "NONE"

    return {
        "valid": is_valid,
        "error_type": error_type,
        "schema_errors": schema_errors,
        "violations": violations,
        "allowed_actions": allowed_actions,
    }

