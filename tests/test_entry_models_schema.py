import math

from engine.entry_models import ENTRY_MODELS

REQUIRED_RISK_KEYS = [
    "expected_R",
    "mae_bucket",
    "mfe_bucket",
    "time_horizon",
    "aggressiveness",
]


def test_entry_models_include_risk_profile_schema():
    for entry_id, model in ENTRY_MODELS.items():
        risk = getattr(model, "risk_profile", None)
        assert isinstance(risk, dict), f"{entry_id} missing risk_profile dict"
        for key in REQUIRED_RISK_KEYS:
            assert key in risk, f"{entry_id} missing risk_profile key {key}"
            assert risk[key] is not None, f"{entry_id} has null risk_profile.{key}"
        assert isinstance(risk["expected_R"], (int, float)) and not math.isnan(
            float(risk["expected_R"])
        ), f"{entry_id} risk_profile.expected_R must be numeric"
