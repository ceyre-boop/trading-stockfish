from engine.anomaly_detector import AnomalyDetector
from engine.decision_actions import ActionType, DecisionAction
from engine.decision_frame import DecisionFrame
from engine.entry_brain import BrainPolicy
from engine.environment_health import EnvironmentHealthMonitor
from engine.realtime_decision_loop import LiveMode, RealtimeDecisionLoop
from engine.runtime_safety_envelope import RuntimeSafetyEnvelope, SafeMode


class StubSearchEngine:
    def __init__(self):
        self.calls = 0

    def rank_actions(self, frame, position_state, entry_models, risk_envelope):
        self.calls += 1
        action = DecisionAction(
            ActionType.OPEN_LONG, entry_model_id="E1", direction="LONG"
        )
        return [
            (
                action,
                {"unified_score": 2.0, "EV_brain": 1.0, "MCR": {"variance_EV": 0.1}},
            )
        ]


def test_realtime_attaches_environment_and_forces_no_trade_on_anomaly(monkeypatch):
    search = StubSearchEngine()
    env_monitor = EnvironmentHealthMonitor({"max_spread": 0.1, "min_volume": 100})
    anomaly_detector = AnomalyDetector({"max_anomaly_count": 1, "window_seconds": 10})
    safety = RuntimeSafetyEnvelope(
        safe_mode=SafeMode.LIVE_THROTTLED,
        global_risk_limits={},
        regime_rules={},
        brain_trust_thresholds={},
    )

    loop = RealtimeDecisionLoop(
        search_engine=search,  # type: ignore[arg-type]
        brain_policy=BrainPolicy(policy={"E1": "ALLOWED"}),
        mode=LiveMode.PAPER_TRADING,
        safety_envelope=safety,
        parity_checker=None,
        env_monitor=env_monitor,
        anomaly_detector=anomaly_detector,
    )

    # silence logging
    monkeypatch.setattr(
        "engine.realtime_decision_loop.log_live_decision", lambda t: None
    )
    monkeypatch.setattr(
        "engine.realtime_decision_loop.log_realtime_decision", lambda d: None
    )

    market_state = {
        "timestamp_utc": "2025-01-01T00:00:00Z",
        "best_bid": 100.0,
        "best_ask": 100.2,
        "volume": 50,  # low volume triggers anomaly
        "entry_models": [],
    }

    decision = loop.run_once(
        market_state=market_state,
        position_state={"is_open": False},
        clock_state={"timestamp_utc": "2025-01-01T00:00:01Z", "bar_index": 1},
        risk_envelope={},
    )

    assert "environment" in decision
    assert decision["environment"]["health"] is not None
    assert decision["environment"]["anomaly_decision"] is not None

    # Anomaly should force NO_TRADE via safety
    assert decision["chosen_action"]["action_type"] == ActionType.NO_TRADE
    assert decision["safety_decision"]["reason"] == "environment_anomaly"
