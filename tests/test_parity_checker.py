from engine.decision_actions import ActionType, DecisionAction
from engine.decision_frame import DecisionFrame
from engine.parity_checker import ParityChecker
from engine.search_engine_v1 import SearchEngineV1


class StubSearchEngine:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def rank_actions(self, frame, position_state, entry_models, risk_envelope):
        self.calls += 1
        return self.scores


def make_action(action_type, entry_id=None, direction=None):
    return DecisionAction(
        action_type=action_type, entry_model_id=entry_id, direction=direction
    )


def test_parity_match():
    scores = [
        (make_action(ActionType.OPEN_LONG, "E1", "LONG"), {"unified_score": 2.0}),
        (make_action(ActionType.NO_TRADE), {"unified_score": 0.5}),
    ]
    stub = StubSearchEngine(scores)
    checker = ParityChecker(search_engine=stub)  # type: ignore[arg-type]

    report = checker.compare_live_vs_replay(DecisionFrame(), {}, [], {}, scores)

    assert report["match"] is True
    assert report["differences"] == []
    assert stub.calls == 1


def test_parity_mismatch():
    live_scores = [
        (make_action(ActionType.OPEN_LONG, "E1", "LONG"), {"unified_score": 2.0}),
    ]
    replay_scores = [
        (make_action(ActionType.NO_TRADE), {"unified_score": 0.5}),
    ]
    stub = StubSearchEngine(replay_scores)
    checker = ParityChecker(search_engine=stub)  # type: ignore[arg-type]

    report = checker.compare_live_vs_replay(DecisionFrame(), {}, [], {}, live_scores)

    assert report["match"] is False
    assert "top_action_mismatch" in report["differences"]
