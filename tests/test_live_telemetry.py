import json
from datetime import datetime

from engine.live_telemetry import LiveDecisionTelemetry, emit_live_telemetry


def test_live_decision_telemetry_json_safe():
    telemetry = LiveDecisionTelemetry(
        timestamp=datetime(2025, 1, 1),
        decision_frame={"symbol": "ES"},
        ranked_actions=[
            {"action": {"action_type": "NO_TRADE"}, "scores": {"unified_score": 0.0}}
        ],
        chosen_action={"action_type": "NO_TRADE"},
        safety_decision={"final_action": {"action_type": "NO_TRADE"}, "reason": "safe"},
        search_diagnostics={"unified_score": 0.0},
        mode="DRY_RUN",
        metadata={"policy_label": "ALLOWED"},
    )

    # Should not raise on emit and should be JSON-serializable
    emit_live_telemetry(telemetry)

    payload = json.dumps(telemetry.__dict__, default=str)
    assert "ES" in payload
