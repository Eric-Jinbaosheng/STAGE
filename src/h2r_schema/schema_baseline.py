from typing import Dict, List


PHASES = ["offer", "reach", "align", "contact", "transfer", "secure", "retract", "failure"]


def infer_phase(rel: Dict[str, float]) -> str:
    if rel.get("held_by_robot", 0.0) > 0.8 and rel.get("held_by_human", 0.0) < 0.2:
        return "secure"
    if rel.get("contact_confirmed", 0.0) > 0.7 and rel.get("held_by_human", 0.0) > 0.3:
        return "transfer"
    if rel.get("contact_confirmed", 0.0) > 0.6:
        return "contact"
    if rel.get("aligned_gripper_object", 0.0) > 0.7 and rel.get("approaching_gripper_object", 0.0) > 0.5:
        return "align"
    if rel.get("approaching_gripper_object", 0.0) > 0.45:
        return "reach"
    if rel.get("human_released", 0.0) > 0.6 and rel.get("held_by_robot", 0.0) > 0.4:
        return "retract"
    return "offer"


def build_uncertainty(rel: Dict[str, float]) -> Dict[str, float | bool]:
    occlusion_prob = rel.get("occluded_object", 0.0)
    conflict = max(0.0, rel.get("held_by_human", 0.0) + rel.get("held_by_robot", 0.0) - 1.0)
    phase_regress_prob = max(0.0, rel.get("approaching_gripper_object", 0.0) - rel.get("aligned_gripper_object", 0.0))
    unknown = (occlusion_prob > 0.6) or (conflict > 0.5)
    return {
        "release_prob": rel.get("human_released", 0.0),
        "secure_prob": rel.get("held_by_robot", 0.0),
        "conflict_score": conflict,
        "occlusion_prob": occlusion_prob,
        "phase_regress_prob": phase_regress_prob,
        "unknown": unknown,
    }


def run_checker(schema: Dict) -> Dict:
    violations: List[str] = []
    repairs: List[str] = []

    phase = schema.get("phase", "failure")
    uncertainty = schema.get("uncertainty", {})
    rel = schema.get("relations", {})

    if phase not in PHASES:
        violations.append("invalid_phase")
        schema["phase"] = "failure"
        repairs.append("phase->failure")

    if phase == "retract" and rel.get("held_by_robot", 0.0) < 0.6:
        violations.append("retract_before_secure")
        schema["phase"] = "transfer"
        repairs.append("phase:retract->transfer")

    if uncertainty.get("unknown", False):
        affordances = schema.get("affordances", [])
        safe_set = {"hold", "backoff", "viewpoint_change", "prompt", "abort"}
        filtered = [a for a in affordances if a.get("action") in safe_set]
        if filtered != affordances:
            schema["affordances"] = filtered
            repairs.append("affordance_filter_unknown")

    schema["checker"] = {
        "valid_schema": len(violations) == 0,
        "violations": violations,
        "repair_applied": repairs,
    }
    return schema


def baseline_affordances(rel: Dict[str, float], uncertainty: Dict[str, float | bool]) -> List[Dict]:
    actions = []
    if bool(uncertainty.get("unknown", False)):
        actions.extend(
            [
                {"action": "hold", "score": 0.9, "rationale": "high uncertainty"},
                {"action": "viewpoint_change", "score": 0.7, "rationale": "occlusion or conflict"},
            ]
        )
        return actions

    if rel.get("held_by_robot", 0.0) > 0.8:
        actions.append({"action": "retract", "score": 0.9, "rationale": "object secured"})
    elif rel.get("contact_confirmed", 0.0) > 0.7:
        actions.append({"action": "grasp", "score": 0.8, "rationale": "contact confirmed"})
        actions.append({"action": "pull", "score": 0.6, "rationale": "start transfer"})
    elif rel.get("aligned_gripper_object", 0.0) > 0.7:
        actions.append({"action": "realign", "score": 0.6, "rationale": "fine alignment"})
    else:
        actions.append({"action": "approach", "score": 0.6, "rationale": "continue reach"})
    return actions

