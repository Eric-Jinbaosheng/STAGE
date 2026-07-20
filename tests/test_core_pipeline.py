from linguistic_blindness.benchmark.schema import normalize_schema
from linguistic_blindness.verification.checker import check_schema
from linguistic_blindness.verification.gate import gate_action


def test_blank_instruction_is_gated_to_ask():
    example = {
        "instruction": "",
        "perturbation_type": "blank_instruction",
        "counterfactual_target_object": None,
        "original_target_object": "bottle",
    }
    schema = normalize_schema(
        {
            "target_object": "bottle",
            "target_exists": True,
            "phase": "reach",
            "human_contact": False,
            "human_released": True,
            "robot_contact": False,
            "robot_grasp_stable": False,
            "allowed_actions": ["PICK"],
            "blocked_actions": [],
            "next_action": "PICK",
            "confidence": 0.9,
        }
    )

    checked = check_schema(example, schema)
    gated = gate_action(example, checked["schema"], checked["flags"])

    assert checked["flags"]["blind_execution"] is True
    assert gated["final_gated_action"] == "ASK"
