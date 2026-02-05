from engine.decision_actions import ActionType, DecisionAction
from engine.decision_frame import DecisionFrame
from engine.entry_brain import BrainPolicy
from engine.parity_checker import ParityChecker
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


def test_realtime_run_once_attaches_parity_and_safety(monkeypatch):
    search = StubSearchEngine()
    parity = ParityChecker(search_engine=search)  # type: ignore[arg-type]
    safety = RuntimeSafetyEnvelope(
        safe_mode=SafeMode.LIVE_THROTTLED,
        global_risk_limits={},
        regime_rules={},
        brain_trust_thresholds={"max_variance": 0.05},
    )
    loop = RealtimeDecisionLoop(
        search_engine=search,  # type: ignore[arg-type]
        brain_policy=BrainPolicy(policy={"E1": "ALLOWED"}),
        mode=LiveMode.PAPER_TRADING,
        safety_envelope=safety,
        parity_checker=parity,
    )

    # silence telemetry logging
    monkeypatch.setattr(
        "engine.realtime_decision_loop.log_live_decision", lambda t: None
    )
    monkeypatch.setattr(
        "engine.realtime_decision_loop.log_realtime_decision", lambda d: None
    )

    decision = loop.run_once(
        market_state={"timestamp_utc": "2025-01-01T00:00:00Z", "entry_models": []},
        position_state={"is_open": False},
        clock_state={"timestamp_utc": "2025-01-01T00:00:00Z", "bar_index": 1},
        risk_envelope={},
    )

    assert "safety_decision" in decision
    assert decision["safety_decision"]["reason"] == "brain_trust_high_variance"

    # parity metadata should exist in telemetry (via parity checker). Since logging is stubbed, we call parity directly
    report = parity.compare_live_vs_replay(
        DecisionFrame(),
        {},
        [],
        {},
        decision["ranked_actions"],
    )
    assert "match" in report
