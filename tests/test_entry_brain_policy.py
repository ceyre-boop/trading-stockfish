import pandas as pd
import pytest

import engine.evaluator as evaluator
from engine.brain_policy_builder import build_entry_brain_policy
from engine.decision_frame import DecisionFrame
from engine.ml_brain import BrainScore


class _Artifacts:
    def __init__(self):
        self.classifier = object()
        self.regressor = object()
        self.training_metadata = {}


@pytest.fixture
def artifacts():
    return _Artifacts()


def test_entry_policy_marks_low_sample_as_DISABLED(monkeypatch, artifacts, tmp_path):
    monkeypatch.setattr(
        evaluator, "_ENTRY_BRAIN_POLICY_PATH", tmp_path / "brain_policy_entries.json"
    )
    monkeypatch.setattr(evaluator, "_ENTRY_BRAIN_POLICY_CACHE", None, raising=False)

    def fake_score_entry_models(frame, eligible, brain_artifacts):
        return {
            eligible[0]: BrainScore(
                prob_good=0.9, expected_reward=1.0, sample_size=1, flags={}
            )
        }

    monkeypatch.setattr(
        "engine.brain_policy_builder.score_entry_models", fake_score_entry_models
    )

    df = pd.DataFrame(
        [
            {
                "entry_model_id": "E1",
                "market_profile_state": "DISTRIBUTION",
                "session_profile": "PROFILE_1C",
                "liquidity_bias_side": "UP",
                "sample_size": 1,
            }
        ]
    )

    policy = build_entry_brain_policy(df, artifacts)
    assert not policy.empty
    assert policy.loc[0, "label"] == "DISABLED"


def test_entry_policy_prefers_high_reward_high_confidence(
    monkeypatch, artifacts, tmp_path
):
    monkeypatch.setattr(
        evaluator, "_ENTRY_BRAIN_POLICY_PATH", tmp_path / "brain_policy_entries.json"
    )
    monkeypatch.setattr(evaluator, "_ENTRY_BRAIN_POLICY_CACHE", None, raising=False)

    def fake_score_entry_models(frame, eligible, brain_artifacts):
        return {
            eligible[0]: BrainScore(
                prob_good=0.95, expected_reward=2.5, sample_size=50, flags={}
            )
        }

    monkeypatch.setattr(
        "engine.brain_policy_builder.score_entry_models", fake_score_entry_models
    )

    df = pd.DataFrame(
        [
            {
                "entry_model_id": "E1",
                "market_profile_state": "DISTRIBUTION",
                "session_profile": "PROFILE_1C",
                "liquidity_bias_side": "UP",
                "sample_size": 10,
            }
        ]
    )

    policy = build_entry_brain_policy(df, artifacts)
    assert not policy.empty
    assert policy.loc[0, "label"] == "PREFERRED"


def test_shadow_tactical_brain_attaches_recommendations(monkeypatch):
    frame = DecisionFrame(
        market_profile_state="DISTRIBUTION",
        session_profile="PROFILE_1C",
        liquidity_frame={"bias": "UP"},
        eligible_entry_models=["E1"],
    )

    def fake_match(entry_id, decision_frame):
        return {
            "label": "PREFERRED",
            "prob_good": 0.8,
            "expected_reward": 1.2,
            "sample_size": 10,
        }

    monkeypatch.setattr(evaluator, "_match_entry_policy", fake_match)

    evaluator._attach_entry_brain_shadow(frame)

    assert frame.entry_brain_labels["E1"] == "PREFERRED"
    assert frame.entry_brain_scores["E1"]["expected_reward"] == 1.2


def test_shadow_tactical_brain_does_not_change_actions(monkeypatch):
    frame = DecisionFrame(
        eligible_entry_models=["E1"], market_profile_state="ACCUMULATION"
    )

    def fake_match(entry_id, decision_frame):
        return {
            "label": "DISCOURAGED",
            "prob_good": 0.3,
            "expected_reward": -0.1,
            "sample_size": 3,
        }

    monkeypatch.setattr(evaluator, "_match_entry_policy", fake_match)

    before = list(frame.eligible_entry_models)
    evaluator._attach_entry_brain_shadow(frame)
    after = list(frame.eligible_entry_models)

    assert before == after
    assert frame.entry_brain_labels["E1"] == "DISCOURAGED"


def test_entry_policy_artifact_is_deterministic(monkeypatch, artifacts, tmp_path):
    monkeypatch.setattr(
        evaluator, "_ENTRY_BRAIN_POLICY_PATH", tmp_path / "brain_policy_entries.json"
    )
    monkeypatch.setattr(evaluator, "_ENTRY_BRAIN_POLICY_CACHE", None, raising=False)

    def fake_score_entry_models(frame, eligible, brain_artifacts):
        return {
            eligible[0]: BrainScore(
                prob_good=0.7, expected_reward=0.5, sample_size=20, flags={}
            )
        }

    monkeypatch.setattr(
        "engine.brain_policy_builder.score_entry_models", fake_score_entry_models
    )

    df = pd.DataFrame(
        [
            {
                "entry_model_id": "E1",
                "market_profile_state": "DISTRIBUTION",
                "session_profile": "PROFILE_1C",
                "liquidity_bias_side": "UP",
                "sample_size": 20,
            }
        ]
    )

    policy_a = build_entry_brain_policy(df, artifacts)
    policy_b = build_entry_brain_policy(df, artifacts)

    pd.testing.assert_frame_equal(policy_a, policy_b)
