from engine.decision_actions import ActionType, DecisionAction
from engine.entry_brain import BrainPolicy
from engine.live_modes import LiveMode
from engine.mode_guard import ModeGuard
from engine.realtime_decision_loop import RealtimeDecisionLoop


class StubSearchEngine:
    def rank_actions(self, frame, position_state, entry_models, risk_envelope):
        action = DecisionAction(
            ActionType.OPEN_LONG, entry_model_id="E1", direction="LONG"
        )
        return [(action, {"unified_score": 1.0})]


def test_realtime_applies_mode_guard_and_attaches_mode_info(monkeypatch):
    guard = ModeGuard(LiveMode.SIM_LIVE_FEED)
    loop = RealtimeDecisionLoop(
        search_engine=StubSearchEngine(),  # type: ignore[arg-type]
        brain_policy=BrainPolicy(policy={"E1": "ALLOW"}),
        mode=LiveMode.SIM_LIVE_FEED,
        safety_envelope=None,
        parity_checker=None,
        env_monitor=None,
        anomaly_detector=None,
        mode_guard=guard,
    )

    captured = {}
    monkeypatch.setattr(
        "engine.realtime_decision_loop.log_live_decision",
        lambda t: captured.setdefault("telemetry", t),
    )
    monkeypatch.setattr(
        "engine.realtime_decision_loop.log_realtime_decision",
        lambda d: captured.setdefault("decision", d),
    )

    decision = loop.run_once(
        market_state={"timestamp_utc": "2025-01-01T00:00:00Z", "entry_models": []},
        position_state={"is_open": False},
        clock_state={"timestamp_utc": "2025-01-01T00:00:00Z", "bar_index": 1},
        risk_envelope={},
    )

    assert decision["mode_info"]["mode"] == LiveMode.SIM_LIVE_FEED.value
    assert decision["routing_allowed"] is False
    assert decision["position_updates_allowed"] is False
    assert decision["chosen_action"].action_type == ActionType.OPEN_LONG
    assert (
        captured["telemetry"].metadata["mode_info"]["mode"]
        == LiveMode.SIM_LIVE_FEED.value
    )


def test_realtime_mode_guard_live_throttled_keeps_routing(monkeypatch):
    guard = ModeGuard(LiveMode.LIVE_THROTTLED)
    loop = RealtimeDecisionLoop(
        search_engine=StubSearchEngine(),  # type: ignore[arg-type]
        brain_policy=BrainPolicy(policy={"E1": "ALLOW"}),
        mode=LiveMode.LIVE_THROTTLED,
        safety_envelope=None,
        parity_checker=None,
        env_monitor=None,
        anomaly_detector=None,
        mode_guard=guard,
    )

    monkeypatch.setattr(
        "engine.realtime_decision_loop.log_live_decision", lambda t: None
    )
    monkeypatch.setattr(
        "engine.realtime_decision_loop.log_realtime_decision", lambda d: None
    )

    decision = loop.run_once(
        market_state={"timestamp_utc": "2025-01-01T00:00:01Z", "entry_models": []},
        position_state={"is_open": False},
        clock_state={"timestamp_utc": "2025-01-01T00:00:01Z", "bar_index": 2},
        risk_envelope={},
    )

    assert decision["routing_allowed"] is True
    assert decision["position_updates_allowed"] is True
    assert decision["mode_info"]["mode"] == LiveMode.LIVE_THROTTLED.value
