import json
from pathlib import Path

import pytest

from engine.decision_frame import DecisionFrame
from engine.entry_brain import BrainPolicy, score_entry_models
from engine.entry_models import ENTRY_MODELS


@pytest.fixture
def policy_file(tmp_path):
    content = {
        "policy": [
            {
                "entry_model_id": "ENTRY_SWEEP_DISPLACEMENT_REVERSAL",
                "label": "PREFERRED",
            },
            {"entry_model_id": "ENTRY_FVG_RESPECT_CONTINUATION", "label": "ALLOWED"},
            {"entry_model_id": "ENTRY_OB_CONTINUATION", "label": "ALLOWED"},
        ]
    }
    path = tmp_path / "brain_policy_entries.active.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_policy_loader_labels(policy_file):
    policy = BrainPolicy.from_file(policy_file)

    assert (
        policy.lookup("ENTRY_SWEEP_DISPLACEMENT_REVERSAL", DecisionFrame())
        == "PREFERRED"
    )
    assert policy.lookup("ENTRY_FVG_RESPECT_CONTINUATION", DecisionFrame()) == "ALLOWED"
    assert policy.lookup("ENTRY_OB_CONTINUATION", DecisionFrame()) == "ALLOWED"

    for entry_id in ENTRY_MODELS:
        if entry_id in {
            "ENTRY_SWEEP_DISPLACEMENT_REVERSAL",
            "ENTRY_FVG_RESPECT_CONTINUATION",
            "ENTRY_OB_CONTINUATION",
        }:
            continue
        assert policy.lookup(entry_id, DecisionFrame()) == "DISABLED"


def test_policy_multipliers_disable_zero(policy_file, monkeypatch):
    policy = BrainPolicy.from_file(policy_file)

    def fake_is_entry_eligible(model, decision_frame):
        return True

    monkeypatch.setattr("engine.entry_brain.is_entry_eligible", fake_is_entry_eligible)

    def fake_score_selector(frame, eligible_ids, artifacts):
        return {entry_id: {"expected_R": 2.0} for entry_id in eligible_ids}

    monkeypatch.setattr("engine.entry_brain.score_entry_selector", fake_score_selector)

    models = [ENTRY_MODELS[entry_id] for entry_id in ENTRY_MODELS.keys()]
    scores = score_entry_models(DecisionFrame(), models, policy)

    # Enabled entries keep multiplier > 0; disabled zero
    assert scores["ENTRY_SWEEP_DISPLACEMENT_REVERSAL"]["adjusted_score"] > 0
    assert scores["ENTRY_FVG_RESPECT_CONTINUATION"]["adjusted_score"] > 0
    assert scores["ENTRY_OB_CONTINUATION"]["adjusted_score"] > 0

    for entry_id, score in scores.items():
        if entry_id in {
            "ENTRY_SWEEP_DISPLACEMENT_REVERSAL",
            "ENTRY_FVG_RESPECT_CONTINUATION",
            "ENTRY_OB_CONTINUATION",
        }:
            continue
        assert score["policy_label"] == "DISABLED"
        assert score["adjusted_score"] == 0.0


def test_policy_loader_deterministic(policy_file):
    policy_a = BrainPolicy.from_file(policy_file)
    policy_b = BrainPolicy.from_file(policy_file)
    for entry_id in ENTRY_MODELS:
        assert policy_a.lookup(entry_id, DecisionFrame()) == policy_b.lookup(
            entry_id, DecisionFrame()
        )
        assert policy_a.multiplier_for("DISABLED") == policy_b.multiplier_for(
            "DISABLED"
        )
