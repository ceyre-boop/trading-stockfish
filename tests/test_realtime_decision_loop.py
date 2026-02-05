from engine import realtime_decision_loop
from engine.decision_actions import ActionType, DecisionAction
from engine.decision_frame import DecisionFrame
from engine.entry_brain import BrainPolicy
from engine.realtime_decision_loop import (
    LiveMode,
    RealtimeDecisionLoop,
    run_realtime_dry_run,
)
from engine.search_engine_v1 import SearchEngineV1


class StubSearchEngine:
    def __init__(self):
        self.calls = 0
        self.rankings = []

    def rank_actions(self, frame, position_state, entry_models, risk_envelope):
        self.calls += 1
        # default deterministic ranking
        action_nt = DecisionAction(action_type=ActionType.NO_TRADE)
        action_long = DecisionAction(
            action_type=ActionType.OPEN_LONG, entry_model_id="E1", direction="LONG"
        )
        ranking = [
            (action_long, {"unified_score": 2.0}),
            (action_nt, {"unified_score": 0.5}),
        ]
        self.rankings.append(ranking)
        return ranking


def test_build_decision_frame_deterministic():
    search = StubSearchEngine()
    loop = RealtimeDecisionLoop(
        search_engine=search,
        brain_policy=BrainPolicy(policy={}),
        mode=LiveMode.SIM_REPLAY,
    )

    market_state = {
        "timestamp_utc": "2025-01-01T00:00:00Z",
        "symbol": "ES",
        "session_profile": "PROFILE_1A",
        "vol_regime": "NORMAL",
        "trend_regime": "UP",
        "liquidity_frame": {"state": "normal"},
        "market_profile_state": "BALANCED",
        "condition_vector": {"x": 1},
    }
    position_state = {"is_open": False}
    clock_state = {"timestamp_utc": "2025-01-01T00:00:00Z", "bar_index": 10}

    frame = loop.build_decision_frame(market_state, position_state, clock_state)

    assert isinstance(frame, DecisionFrame)
    assert frame.timestamp_utc == "2025-01-01T00:00:00Z"
    assert frame.session_profile == "PROFILE_1A"
    assert frame.vol_regime == "NORMAL"
    assert frame.trend_regime == "UP"
    assert frame.liquidity_frame == {"state": "normal"}


def test_run_once_calls_rank_and_returns_intent(monkeypatch):
    search = StubSearchEngine()
    loop = RealtimeDecisionLoop(
        search_engine=search,
        brain_policy=BrainPolicy(policy={"E1": "ALLOWED"}),
        mode=LiveMode.SIM_REPLAY,
    )

    market_state = {"timestamp_utc": "2025-01-01T00:00:00Z", "entry_models": []}
    position_state = {"is_open": False}
    clock_state = {"timestamp_utc": "2025-01-01T00:00:00Z", "bar_index": 1}

    called_log = {"count": 0}

    def fake_log(decision_dict):
        called_log["count"] += 1

    monkeypatch.setattr(realtime_decision_loop, "log_realtime_decision", fake_log)

    decision = loop.run_once(
        market_state, position_state, clock_state, risk_envelope={}
    )

    assert search.calls == 1
    assert decision["chosen_action"].action_type == ActionType.OPEN_LONG
    assert called_log["count"] == 1
    assert decision["ranked_actions"]


class StubPositionTracker:
    def __init__(self, states):
        self.states = list(states)
        self.idx = 0

    def get_state(self):
        state = self.states[min(self.idx, len(self.states) - 1)]
        self.idx += 1
        return state


def test_run_realtime_dry_run_deterministic(monkeypatch):
    search = StubSearchEngine()
    loop = RealtimeDecisionLoop(
        search_engine=search,
        brain_policy=BrainPolicy(policy={"E1": "ALLOWED"}),
        mode=LiveMode.SIM_REPLAY,
    )

    feed = [
        ({"timestamp_utc": "t0"}, {"timestamp_utc": "t0", "bar_index": 0}),
        ({"timestamp_utc": "t1"}, {"timestamp_utc": "t1", "bar_index": 1}),
    ]
    position_tracker = StubPositionTracker(
        [
            {"is_open": False},
            {"is_open": True},
        ]
    )

    log_calls = {"count": 0}

    def fake_log(decision_dict):
        log_calls["count"] += 1

    monkeypatch.setattr(realtime_decision_loop, "log_realtime_decision", fake_log)

    decisions = run_realtime_dry_run(
        loop,
        feed,
        position_tracker,
        risk_envelope_provider=lambda: {},
        max_ticks=None,
    )

    assert len(decisions) == 2
    assert log_calls["count"] == 2
    assert decisions[0]["timestamp"] == "t0"
    assert decisions[1]["timestamp"] == "t1"
    # deterministic ordering and chosen action
    assert decisions[0]["chosen_action"].action_type == ActionType.OPEN_LONG
    assert decisions[1]["chosen_action"].action_type == ActionType.OPEN_LONG

    # running again with same feed copy yields same result shapes
    position_tracker.idx = 0
    log_calls["count"] = 0
    decisions2 = run_realtime_dry_run(
        loop,
        list(feed),
        position_tracker,
        risk_envelope_provider=lambda: {},
        max_ticks=2,
    )
    assert [d["timestamp"] for d in decisions2] == ["t0", "t1"]
    assert log_calls["count"] == 2
