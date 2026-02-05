import numpy as np

from engine.action_pairing import (
    build_action_feature_rows,
    generate_candidate_actions,
    score_candidate_actions,
)
from engine.decision_actions import ActionType
from engine.decision_frame import DecisionFrame
from engine.entry_brain import BrainPolicy
from engine.entry_models import ENTRY_MODELS
from engine.ev_brain_features import FEATURE_COLUMNS


class StubBrain:
    def predict(self, X):
        # deterministic ascending scores
        return np.arange(X.shape[0], dtype=np.float32)


def test_generate_candidate_actions_respects_policy_and_position(monkeypatch):
    frame = DecisionFrame(entry_signals_present={"sweep": True})
    allowed_id = next(iter(ENTRY_MODELS.keys()))
    disabled_id = "ENTRY_DISABLED_TEST"

    def fake_get_eligible(_frame):
        return [allowed_id, disabled_id]

    monkeypatch.setattr(
        "engine.action_pairing.get_eligible_entry_models", fake_get_eligible
    )

    policy_map = {allowed_id: "ALLOWED", disabled_id: "DISABLED"}
    brain_policy = BrainPolicy(policy=policy_map)

    actions_flat = generate_candidate_actions(
        frame,
        position_state={"is_open": False},
        entry_models=[{"id": allowed_id}, {"id": disabled_id}],
        brain_policy=brain_policy,
        risk_envelope={},
    )

    assert any(a.action_type == ActionType.NO_TRADE for a in actions_flat)
    open_actions = [
        a
        for a in actions_flat
        if a.action_type in (ActionType.OPEN_LONG, ActionType.OPEN_SHORT)
    ]
    assert open_actions  # eligible + enabled
    assert all(a.entry_model_id == allowed_id for a in open_actions)
    manage_actions = [
        a for a in actions_flat if a.action_type == ActionType.MANAGE_POSITION
    ]
    assert not manage_actions

    actions_open = generate_candidate_actions(
        frame,
        position_state={"is_open": True},
        entry_models=[{"id": allowed_id}],
        brain_policy=brain_policy,
        risk_envelope={},
    )
    manage_actions = [
        a for a in actions_open if a.action_type == ActionType.MANAGE_POSITION
    ]
    assert manage_actions


def test_build_action_feature_rows_covers_feature_columns(monkeypatch):
    frame = DecisionFrame(
        market_profile_state="ACCUMULATION",
        market_profile_confidence=0.5,
        session_profile="PROFILE_1A",
        session_profile_confidence=0.4,
        vol_regime="NORMAL",
        trend_regime="UP",
        liquidity_frame={"bias": "UP"},
        condition_vector={"session": "RTH"},
    )
    entry_id = next(iter(ENTRY_MODELS.keys()))

    def fake_get_eligible(_frame):
        return [entry_id]

    monkeypatch.setattr(
        "engine.action_pairing.get_eligible_entry_models", fake_get_eligible
    )

    brain_policy = BrainPolicy(policy={entry_id: "ALLOWED"})
    actions = generate_candidate_actions(
        frame,
        position_state={"is_open": False},
        entry_models=[{"id": entry_id}],
        brain_policy=brain_policy,
        risk_envelope={},
    )
    rows = build_action_feature_rows(frame, actions)
    assert rows
    for row in rows:
        for col in FEATURE_COLUMNS:
            assert col in row
        assert not any(k.startswith("label_") for k in row.keys())


def test_score_candidate_actions_deterministic(monkeypatch):
    frame = DecisionFrame()
    entry_id = next(iter(ENTRY_MODELS.keys()))

    def fake_get_eligible(_frame):
        return [entry_id]

    monkeypatch.setattr(
        "engine.action_pairing.get_eligible_entry_models", fake_get_eligible
    )
    brain_policy = BrainPolicy(policy={entry_id: "ALLOWED"})
    stub_brain = StubBrain()

    pairs = score_candidate_actions(
        stub_brain,
        frame,
        position_state={"is_open": False},
        entry_models=[{"id": entry_id}],
        brain_policy=brain_policy,
        risk_envelope={},
    )

    # Scores correspond to order of candidates
    actions, scores = zip(*pairs)
    assert list(scores) == list(range(len(actions)))
