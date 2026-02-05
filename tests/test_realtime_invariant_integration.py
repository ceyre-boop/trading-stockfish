import pytest

from engine.decision_actions import ActionType, DecisionAction
from engine.entry_brain import BrainPolicy
from engine.invariants import InvariantChecker, InvariantViolation
from engine.live_modes import LiveMode
from engine.mode_guard import ModeGuard
from engine.realtime_decision_loop import RealtimeDecisionLoop


class StubSearchEngineBad:
    def rank_actions(self, frame, position_state, entry_models, risk_envelope):
        action = DecisionAction(
            ActionType.OPEN_LONG, entry_model_id="E1", direction="LONG"
        )
        return [(action, {"unified_score": 1.0})]  # missing EV_brain and MCR


class StubSearchEngineGood:
    def rank_actions(self, frame, position_state, entry_models, risk_envelope):
        action = DecisionAction(
            ActionType.OPEN_LONG, entry_model_id="E1", direction="LONG"
        )
        return [
            (action, {"unified_score": 1.0, "EV_brain": 0.5, "MCR": {"mean_EV": 0.1}})
        ]


def test_realtime_loop_raises_on_invariant_breach(monkeypatch):
    loop = RealtimeDecisionLoop(
        search_engine=StubSearchEngineBad(),  # type: ignore[arg-type]
        brain_policy=BrainPolicy(policy={"E1": "ALLOW"}),
        mode=LiveMode.SIM_LIVE_FEED,
        safety_envelope=None,
        parity_checker=None,
        env_monitor=None,
        anomaly_detector=None,
        mode_guard=ModeGuard(LiveMode.SIM_LIVE_FEED),
        operator_report_builder=None,
        invariant_checker=InvariantChecker(),
    )

    monkeypatch.setattr(
        "engine.realtime_decision_loop.log_live_decision", lambda t: None
    )
    monkeypatch.setattr(
        "engine.realtime_decision_loop.log_realtime_decision", lambda d: None
    )

    with pytest.raises(InvariantViolation):
        loop.run_once(
            market_state={"timestamp_utc": "2025-01-01T00:00:00Z", "entry_models": []},
            position_state={"is_open": False},
            clock_state={"timestamp_utc": "2025-01-01T00:00:00Z", "bar_index": 1},
            risk_envelope={},
        )


def test_realtime_loop_passes_when_invariants_hold(monkeypatch):
    loop = RealtimeDecisionLoop(
        search_engine=StubSearchEngineGood(),  # type: ignore[arg-type]
        brain_policy=BrainPolicy(policy={"E1": "ALLOW"}),
        mode=LiveMode.SIM_LIVE_FEED,
        safety_envelope=None,
        parity_checker=None,
        env_monitor=None,
        anomaly_detector=None,
        mode_guard=ModeGuard(LiveMode.SIM_LIVE_FEED),
        operator_report_builder=None,
        invariant_checker=InvariantChecker(),
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

    assert decision["scores"]["EV_brain"] == 0.5
    assert decision["mode_info"]["mode"] == LiveMode.SIM_LIVE_FEED.value
