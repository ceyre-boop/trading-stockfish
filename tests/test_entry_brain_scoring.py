import types

import pytest

from engine.decision_frame import DecisionFrame
from engine.entry_brain import BrainPolicy, score_entry_models
from engine.entry_models import ENTRY_MODELS


class FakePolicy(BrainPolicy):
    def __init__(self, labels=None, multipliers=None):
        super().__init__(selector_artifacts=None)
        self.labels = labels or {}
        self.multipliers = multipliers or {
            "PREFERRED": 1.5,
            "ALLOWED": 1.0,
            "DISABLED": 0.0,
        }

    def lookup(self, entry_id, frame):
        return self.labels.get(entry_id, "ALLOWED")

    def multiplier_for(self, label):
        return float(self.multipliers.get(label, 1.0))


@pytest.fixture
def frame():
    return DecisionFrame(
        market_profile_state="DISTRIBUTION",
        session_profile="PROFILE_1C",
        liquidity_frame={
            "bias": "UP",
            "sweep_state": "POST_SWEEP",
            "distance_bucket": "NEAR",
        },
        entry_signals_present={"sweep": True, "displacement": True, "fvg": True},
        condition_vector={"vol": "NORMAL", "trend": "UP"},
    )


def test_score_entry_models_respects_eligibility(monkeypatch, frame):
    calls = {}

    def fake_is_entry_eligible(model, decision_frame):
        calls[model.id] = True
        return model.id != "ENTRY_SWEEP_DISPLACEMENT_REVERSAL"

    monkeypatch.setattr("engine.entry_brain.is_entry_eligible", fake_is_entry_eligible)

    def fake_score_selector(decision_frame, eligible_ids, artifacts):
        assert eligible_ids == ["ENTRY_FVG_RESPECT_CONTINUATION"]
        return {
            "ENTRY_FVG_RESPECT_CONTINUATION": {"prob_select": 0.8, "expected_R": 2.0}
        }

    monkeypatch.setattr("engine.entry_brain.score_entry_selector", fake_score_selector)

    policy = FakePolicy(labels={"ENTRY_FVG_RESPECT_CONTINUATION": "PREFERRED"})

    models = [
        ENTRY_MODELS["ENTRY_FVG_RESPECT_CONTINUATION"],
        ENTRY_MODELS["ENTRY_SWEEP_DISPLACEMENT_REVERSAL"],
    ]

    scores = score_entry_models(frame, models, policy)

    assert "ENTRY_FVG_RESPECT_CONTINUATION" in scores
    assert "ENTRY_SWEEP_DISPLACEMENT_REVERSAL" not in scores
    s = scores["ENTRY_FVG_RESPECT_CONTINUATION"]
    assert s["raw_score"] == 2.0
    assert s["adjusted_score"] == pytest.approx(3.0)
    assert s["policy_label"] == "PREFERRED"


def test_score_entry_models_deterministic(monkeypatch, frame):
    def fake_is_entry_eligible(model, decision_frame):
        return True

    monkeypatch.setattr("engine.entry_brain.is_entry_eligible", fake_is_entry_eligible)

    def fake_score_selector(decision_frame, eligible_ids, artifacts):
        return {entry_id: {"expected_R": 1.1} for entry_id in eligible_ids}

    monkeypatch.setattr("engine.entry_brain.score_entry_selector", fake_score_selector)

    policy = FakePolicy(labels={"ENTRY_FVG_RESPECT_CONTINUATION": "ALLOWED"})
    models = [ENTRY_MODELS["ENTRY_FVG_RESPECT_CONTINUATION"]]

    scores_a = score_entry_models(frame, models, policy)
    scores_b = score_entry_models(frame, models, policy)

    assert scores_a == scores_b


def test_disabled_entries_zero_adjusted(monkeypatch, frame):
    def fake_is_entry_eligible(model, decision_frame):
        return True

    monkeypatch.setattr("engine.entry_brain.is_entry_eligible", fake_is_entry_eligible)

    def fake_score_selector(decision_frame, eligible_ids, artifacts):
        return {entry_id: {"expected_R": 5.0} for entry_id in eligible_ids}

    monkeypatch.setattr("engine.entry_brain.score_entry_selector", fake_score_selector)

    policy = FakePolicy(labels={"ENTRY_FVG_RESPECT_CONTINUATION": "DISABLED"})
    models = [ENTRY_MODELS["ENTRY_FVG_RESPECT_CONTINUATION"]]

    scores = score_entry_models(frame, models, policy)
    s = scores["ENTRY_FVG_RESPECT_CONTINUATION"]
    assert s["raw_score"] == 5.0
    assert s["adjusted_score"] == 0.0
    assert s["policy_label"] == "DISABLED"
