from engine import search_engine_v1, search_scoring
from engine.decision_actions import ActionType, DecisionAction
from engine.decision_frame import DecisionFrame
from engine.entry_brain import BrainPolicy
from engine.search_engine_v1 import SearchEngineV1


class StubOpeningBook:
    def __init__(self, boosts):
        self.boosts = boosts

    def lookup(self, frame, position_state, candidate_actions):
        return {
            self._aid(a): self.boosts.get(self._aid(a), 0.0) for a in candidate_actions
        }

    def _aid(self, action):
        aid = f"{action.action_type.value}:{action.entry_model_id or ''}:{(action.direction or '').upper()}:{(action.size_bucket or '').upper()}"
        return aid


class StubTablebases:
    def __init__(self, overrides):
        self.overrides = overrides

    def lookup(self, frame, position_state, candidate_actions):
        return {
            self._aid(a): self.overrides.get(self._aid(a), 0.0)
            for a in candidate_actions
        }

    def _aid(self, action):
        aid = f"{action.action_type.value}:{action.entry_model_id or ''}:{(action.direction or '').upper()}:{(action.size_bucket or '').upper()}"
        return aid


class StubEVBrain:
    def predict(self, X):
        import numpy as np

        return np.zeros(X.shape[0], dtype=float)


def make_action(action_type, entry_id=None, direction=None):
    return DecisionAction(
        action_type=action_type,
        entry_model_id=entry_id,
        direction=direction,
        size_bucket="SMALL",
    )


def test_rank_actions_deterministic_and_applies_layers(monkeypatch):
    # Stub candidate generation
    actions = [
        make_action(ActionType.NO_TRADE),
        make_action(ActionType.OPEN_LONG, "E1", "LONG"),
    ]

    def fake_generate(frame, position_state, entry_models, brain_policy, risk_envelope):
        return list(actions)

    # Base scores via cached search (will be mutated by opening/endgame)
    call_counter = {"calls": 0}

    def fake_score_actions_via_search(
        frame, position_state, candidate_actions, **kwargs
    ):
        call_counter["calls"] += 1
        results = []
        for a in candidate_actions:
            base = 2.0 if a.action_type != ActionType.NO_TRADE else 1.0
            results.append((a, {"unified_score": base}))
        return results

    monkeypatch.setattr(search_engine_v1, "generate_candidate_actions", fake_generate)
    monkeypatch.setattr(
        search_scoring, "score_actions_via_search", fake_score_actions_via_search
    )

    opening = StubOpeningBook({"OPEN_LONG:E1:LONG:SMALL": 0.5})
    tablebases = StubTablebases({"OPEN_LONG:E1:LONG:SMALL": -0.1})
    engine = SearchEngineV1(
        ev_brain=StubEVBrain(),
        brain_policy=BrainPolicy(policy={"E1": "ALLOWED"}),
        opening_book=opening,
        tablebases=tablebases,
        n_paths=2,
        horizon_bars=2,
        seed=7,
        cache_max_size=10,
    )

    frame = DecisionFrame()
    risk_envelope = {}

    ranked1 = engine.rank_actions(
        frame, position_state={}, entry_models=[], risk_envelope=risk_envelope
    )
    first_calls = call_counter["calls"]
    ranked2 = engine.rank_actions(
        frame, position_state={}, entry_models=[], risk_envelope=risk_envelope
    )

    # Ensure cache hit prevented additional compute
    assert call_counter["calls"] == first_calls

    # Ensure NO_TRADE present
    assert any(a.action_type == ActionType.NO_TRADE for a, _ in ranked1)

    # Opening and endgame adjustments reflected
    for action, scores in ranked1:
        if action.action_type == ActionType.OPEN_LONG:
            assert scores["opening_book_score"] == 0.5
            assert scores["endgame_score"] == -0.1
            assert scores["unified_score"] == 2.4

    # Deterministic ordering: OPEN_LONG should rank above NO_TRADE
    assert ranked1[0][0].action_type == ActionType.OPEN_LONG


def test_endgame_overrides_dominate(monkeypatch):
    actions = [
        make_action(ActionType.NO_TRADE),
        make_action(ActionType.OPEN_SHORT, "E2", "SHORT"),
    ]

    def fake_generate(frame, position_state, entry_models, brain_policy, risk_envelope):
        return list(actions)

    def fake_score_actions_via_search(
        frame, position_state, candidate_actions, **kwargs
    ):
        results = []
        for a in candidate_actions:
            base = 5.0 if a.action_type != ActionType.NO_TRADE else 0.5
            results.append((a, {"unified_score": base}))
        return results

    monkeypatch.setattr(search_engine_v1, "generate_candidate_actions", fake_generate)
    monkeypatch.setattr(
        search_scoring, "score_actions_via_search", fake_score_actions_via_search
    )

    # Endgame forbids new positions
    opening = StubOpeningBook({})
    tablebases = StubTablebases(
        {"OPEN_SHORT:E2:SHORT:SMALL": -1e6, "NO_TRADE::": 0.25, "NO_TRADE:::": 0.25}
    )

    engine = SearchEngineV1(
        ev_brain=StubEVBrain(),
        brain_policy=BrainPolicy(policy={"E2": "ALLOWED"}),
        opening_book=opening,
        tablebases=tablebases,
        n_paths=2,
        horizon_bars=2,
        seed=11,
        cache_max_size=10,
    )

    frame = DecisionFrame()

    ranked = engine.rank_actions(
        frame, position_state={}, entry_models=[], risk_envelope={}
    )

    # The endgame forbid should push NO_TRADE to top
    assert ranked[0][0].action_type == ActionType.NO_TRADE
    assert ranked[0][1]["unified_score"] > ranked[1][1]["unified_score"]
