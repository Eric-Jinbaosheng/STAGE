from copy import deepcopy
from typing import Callable, Dict, Tuple


def apply_minimal_repair(instance: Dict, checker_report: Dict) -> Dict:
    fixed = deepcopy(instance)
    violations = set(checker_report.get("violations", []))
    allowed = checker_report.get("allowed_actions", [])

    affordances = fixed.get("affordances", [])
    aff_names = [a.get("name") for a in affordances]
    chosen = fixed.get("next_action", {}).get("chosen")

    # Keep changes minimal: update only broken fields.
    if "next_action_not_in_affordances" in violations:
        if aff_names:
            fixed.setdefault("next_action", {})["chosen"] = aff_names[0]
            fixed["next_action"]["reason"] = "repair: chosen action must be in affordances"

    if "contact_not_confirmed_but_close_gentle" in violations or "contact_unknown_but_close_gentle" in violations:
        # Replace chosen action only if currently unsafe.
        if fixed.get("next_action", {}).get("chosen") == "CLOSE_GENTLE":
            replacement = "HOLD_COMPLIANT"
            if "HOLD_COMPLIANT" in aff_names:
                fixed["next_action"]["chosen"] = replacement
            elif aff_names:
                fixed["next_action"]["chosen"] = aff_names[0]
            fixed["next_action"]["reason"] = "repair: avoid CLOSE_GENTLE without confirmed contact"

    if "occluded_but_non_conservative_action" in violations:
        # Ensure conservative choice is available and chosen.
        target = "VIEWPOINT_CHANGE"
        if target not in aff_names:
            affordances.append(
                {
                    "name": target,
                    "params": {"delta_deg": 20},
                    "preconditions": ["occluded=true"],
                    "expected_observation": "occlusion reduced",
                    "score": 0.9,
                }
            )
            aff_names.append(target)
        fixed["affordances"] = affordances
        fixed["next_action"]["chosen"] = target
        fixed["next_action"]["reason"] = "repair: conservative action under occlusion"

    # If checker gave allowed action set, snap chosen to allowed set.
    chosen = fixed.get("next_action", {}).get("chosen")
    if allowed and chosen not in allowed:
        fixed["next_action"]["chosen"] = allowed[0]
        fixed["next_action"]["reason"] = "repair: align with allowed action set"

    return fixed


def run_repair_loop(
    candidate: Dict,
    checker_fn: Callable[[Dict], Dict],
    max_retries: int = 2,
) -> Tuple[Dict, Dict, int]:
    current = deepcopy(candidate)
    report = checker_fn(current)
    retries = 0
    while (not report.get("valid", False)) and retries < max_retries:
        current = apply_minimal_repair(current, report)
        report = checker_fn(current)
        retries += 1
    return current, report, retries

