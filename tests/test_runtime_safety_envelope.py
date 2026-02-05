from dataclasses import asdict

from engine.decision_actions import ActionType, DecisionAction
from engine.decision_frame import DecisionFrame
from engine.entry_brain import BrainPolicy
from engine.realtime_decision_loop import LiveMode, RealtimeDecisionLoop
from engine.runtime_safety_envelope import RuntimeSafetyEnvelope, SafeMode


class StubSearchEngine:
    def __init__(self, score_value=2.0, variance=0.0):
        self.score_value = score_value
        self.variance = variance
        self.calls = 0

    def rank_actions(self, frame, position_state, entry_models, risk_envelope):
        self.calls += 1
        act = DecisionAction(
            ActionType.OPEN_LONG, entry_model_id="E1", direction="LONG"
        )
        scores = {
            "unified_score": self.score_value,
            "EV_brain": 1.0,
            "MCR": {"variance_EV": self.variance},
        }
        return [(act, scores)]


def test_mode_guard_offline_forces_no_trade():
    env = RuntimeSafetyEnvelope(
        safe_mode=SafeMode.OFFLINE_ONLY,
        global_risk_limits={},
        regime_rules={},
        brain_trust_thresholds={},
    )
    frame = DecisionFrame()
    action = DecisionAction(ActionType.OPEN_LONG)
    intent = {"chosen_action": action, "scores": {"unified_score": 2.0}}
    decision = env.evaluate_intent(frame, {}, {}, intent)
    assert decision.final_action.action_type == ActionType.NO_TRADE
    assert decision.reason == "safe_mode_offline_only"


def test_mode_guard_paper_allows_but_marks():
    env = RuntimeSafetyEnvelope(
        safe_mode=SafeMode.PAPER_ONLY,
        global_risk_limits={},
        regime_rules={},
        brain_trust_thresholds={},
    )
    frame = DecisionFrame()
    action = DecisionAction(ActionType.OPEN_LONG)
    intent = {"chosen_action": action, "scores": {"unified_score": 2.0}}
    decision = env.evaluate_intent(frame, {}, {}, intent)
    assert decision.final_action.action_type == ActionType.OPEN_LONG
    assert decision.reason == "safe_mode_paper_only"


def test_global_risk_guard_downgrades():
    env = RuntimeSafetyEnvelope(
        safe_mode=SafeMode.LIVE_THROTTLED,
        global_risk_limits={"max_daily_R_loss": 5},
        regime_rules={},
        brain_trust_thresholds={},
    )
    frame = DecisionFrame()
    action = DecisionAction(ActionType.OPEN_LONG)
    intent = {"chosen_action": action, "scores": {"unified_score": 2.0}}
    risk_envelope = {"daily_R_loss": -6}
    decision = env.evaluate_intent(frame, {}, risk_envelope, intent)
    assert decision.final_action.action_type == ActionType.NO_TRADE
    assert decision.reason == "global_risk_violation"


def test_regime_guard_extreme_vol_blocks_entries():
    env = RuntimeSafetyEnvelope(
        safe_mode=SafeMode.LIVE_THROTTLED,
        global_risk_limits={},
        regime_rules={},
        brain_trust_thresholds={},
    )
    frame = DecisionFrame(vol_regime="extreme")
    action = DecisionAction(ActionType.OPEN_LONG, direction="LONG")
    intent = {"chosen_action": action, "scores": {"unified_score": 2.0}}
    decision = env.evaluate_intent(frame, {}, {}, intent)
    assert decision.final_action.action_type == ActionType.NO_TRADE
    assert decision.reason == "regime_vol_extreme"


def test_brain_trust_low_unified_blocks():
    env = RuntimeSafetyEnvelope(
        safe_mode=SafeMode.LIVE_THROTTLED,
        global_risk_limits={},
        regime_rules={},
        brain_trust_thresholds={"min_unified": 1.0},
    )
    frame = DecisionFrame()
    action = DecisionAction(ActionType.OPEN_LONG, direction="LONG")
    intent = {
        "chosen_action": action,
        "scores": {"unified_score": 0.0, "EV_brain": 0.0, "MCR": {"variance_EV": 0.0}},
    }
    decision = env.evaluate_intent(frame, {}, {}, intent)
    assert decision.final_action.action_type == ActionType.NO_TRADE
    assert decision.reason == "brain_trust_low_unified"


def test_brain_trust_policy_block():
    env = RuntimeSafetyEnvelope(
        safe_mode=SafeMode.LIVE_THROTTLED,
        global_risk_limits={},
        regime_rules={},
        brain_trust_thresholds={},
    )
    frame = DecisionFrame()
    action = DecisionAction(ActionType.OPEN_LONG, direction="LONG")
    intent = {
        "chosen_action": action,
        "scores": {"unified_score": 2.0, "policy_label": "DISCOURAGED"},
    }
    decision = env.evaluate_intent(frame, {}, {}, intent)
    assert decision.final_action.action_type == ActionType.NO_TRADE
    assert decision.reason == "brain_trust_policy_block"


def test_integration_with_realtime_loop_attaches_safety(monkeypatch):
    search = StubSearchEngine(score_value=2.0, variance=0.2)
    env = RuntimeSafetyEnvelope(
        safe_mode=SafeMode.LIVE_THROTTLED,
        global_risk_limits={},
        regime_rules={},
        brain_trust_thresholds={"max_variance": 0.1},
    )
    loop = RealtimeDecisionLoop(
        search_engine=search,
        brain_policy=BrainPolicy(policy={"E1": "ALLOWED"}),
        mode=LiveMode.SIM_REPLAY,
        safety_envelope=env,
    )

    market_state = {"timestamp_utc": "2025-01-01T00:00:00Z", "entry_models": []}
    position_state = {"is_open": False}
    clock_state = {"timestamp_utc": "2025-01-01T00:00:00Z", "bar_index": 1}

    # Silence logging
    monkeypatch.setattr(
        "engine.realtime_decision_loop.log_realtime_decision", lambda d: None
    )

    decision = loop.run_once(
        market_state, position_state, clock_state, risk_envelope={}
    )

    assert "safety_decision" in decision
    sd = decision["safety_decision"]
    assert sd["final_action"]["action_type"] == ActionType.NO_TRADE
    assert sd["reason"] == "brain_trust_high_variance"

    # Ensure deterministic safety response
    decision2 = loop.run_once(
        market_state, position_state, clock_state, risk_envelope={}
    )
    assert decision2["safety_decision"]["reason"] == sd["reason"]
