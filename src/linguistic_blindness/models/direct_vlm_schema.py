from linguistic_blindness.models.base import UnavailableModel


def build_model():
    return UnavailableModel("direct_vlm_schema", "No raw per-example VLM schema predictions are registered for this benchmark yet. Use scripts/lb_run_qwen_schema.py to generate them.")
