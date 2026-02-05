from engine.decision_actions import ActionType, DecisionAction
from engine.decision_frame import DecisionFrame
from engine.search_scoring import score_actions_with_cache
from engine.transposition_table import TranspositionTable


def test_compute_state_hash_stability_and_sensitivity():
    table = TranspositionTable()
    frame = DecisionFrame(
        market_profile_state="ACCUMULATION",
        vol_regime="NORMAL",
        trend_regime="UP",
        liquidity_frame={"bias": "UP"},
    )
    pos = {"is_open": False}
    action = DecisionAction(
        action_type=ActionType.OPEN_LONG, entry_model_id="ENTRY_A", direction="LONG"
    )

    h1 = table.compute_state_hash(frame, pos, action)
    h2 = table.compute_state_hash(frame, pos, action)
    assert h1 == h2

    action2 = DecisionAction(
        action_type=ActionType.OPEN_LONG, entry_model_id="ENTRY_B", direction="LONG"
    )
    h3 = table.compute_state_hash(frame, pos, action2)
    assert h1 != h3

    pos2 = {"is_open": True}
    h4 = table.compute_state_hash(frame, pos2, action)
    assert h1 != h4


def test_lookup_store_and_eviction():
    table = TranspositionTable(max_size=2)
    table.store("k1", {"v": 1})
    table.store("k2", {"v": 2})
    assert table.lookup("k1") == {"v": 1}
    table.store("k3", {"v": 3})
    # k1 should be evicted (oldest)
    assert table.lookup("k1") is None
    assert table.lookup("k2") == {"v": 2}
    assert table.lookup("k3") == {"v": 3}


def test_score_actions_with_cache(monkeypatch):
    frame = DecisionFrame()
    actions = [
        DecisionAction(action_type=ActionType.NO_TRADE),
        DecisionAction(
            action_type=ActionType.OPEN_LONG, entry_model_id="ENTRY_A", direction="LONG"
        ),
    ]

    class StubBrain:
        def predict(self, X):
            import numpy as np

            return np.zeros(X.shape[0], dtype=float)

    class StubPolicy:
        def lookup(self, entry_id, frame):
            return "ALLOWED"

        def multiplier_for(self, label):
            return 0.0

    from engine import search_scoring

    monkeypatch.setattr(
        search_scoring,
        "score_actions_via_search",
        lambda *a, **k: [
            (actions[0], {"unified_score": 1.0}),
            (actions[1], {"unified_score": 2.0}),
        ],
    )

    table = TranspositionTable(max_size=10)

    first = score_actions_with_cache(
        frame,
        position_state={},
        candidate_actions=actions,
        ev_brain=StubBrain(),
        brain_policy=StubPolicy(),
        risk_envelope={},
        n_paths=2,
        horizon_bars=2,
        seed=1,
        table=table,
    )
    second = score_actions_with_cache(
        frame,
        position_state={},
        candidate_actions=actions,
        ev_brain=StubBrain(),
        brain_policy=StubPolicy(),
        risk_envelope={},
        n_paths=2,
        horizon_bars=2,
        seed=1,
        table=table,
    )

    assert first == second
    # Ensure cache populated
    key = table.compute_state_hash(frame, {}, actions[0])
    assert table.lookup(key) is not None
