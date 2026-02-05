import numpy as np
import pandas as pd

from engine.adversarial_replay import adversarial_replay
from engine.decision_frame import DecisionFrame
from engine.entry_drift import detect_entry_drift
from engine.entry_performance import attribute_entry_performance
from engine.regime_stress_test import stress_test_entries_by_regime


class DummyEncoder:
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # One-hot encode the entry id into two columns
        mapping = {
            "ENTRY_FVG_RESPECT_CONTINUATION": np.array([1.0, 0.0]),
            "ENTRY_OB_CONTINUATION": np.array([0.0, 1.0]),
        }
        rows = []
        for _, row in df.iterrows():
            rows.append(mapping.get(row.get("entry_model_id"), np.array([0.0, 0.0])))
        return np.vstack(rows)


class DummyClassifier:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Bias towards ENTRY_A when its indicator is 1
        probs = []
        for row in X:
            if row[0] == 1.0:
                probs.append([0.8, 0.2])
            else:
                probs.append([0.3, 0.7])
        return np.array(probs)


class DummyRegressor:
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Higher expected R for ENTRY_A
        outputs = []
        for row in X:
            outputs.append(1.5 if row[0] == 1.0 else 0.2)
        return np.array(outputs)


class DummyArtifacts:
    def __init__(self):
        self.classifier = DummyClassifier()
        self.regressor = DummyRegressor()
        self.encoders = {"categorical": DummyEncoder()}
        self.metadata = {
            "classes": ["ENTRY_FVG_RESPECT_CONTINUATION", "ENTRY_OB_CONTINUATION"]
        }


def _decision_frame(ts: str) -> DecisionFrame:
    frame = DecisionFrame(
        timestamp_utc=ts,
        market_profile_state="ACCUMULATION",
        session_profile="PROFILE_1A",
        liquidity_frame={"bias": "UP", "sweep_state": "ANY", "distance_bucket": "NEAR"},
        vol_regime="NORMAL",
        trend_regime="UP",
        entry_signals_present={"fvg": True},
    )
    frame.entry_brain_scores = {
        "ENTRY_FVG_RESPECT_CONTINUATION": {"expected_reward": 1.0},
        "ENTRY_OB_CONTINUATION": {"expected_reward": 0.1},
    }
    return frame


def test_adversarial_replay_reconstructs_scores():
    artifacts = DummyArtifacts()
    frame = _decision_frame("2024-01-01T00:00:00Z")
    logs = pd.DataFrame(
        [
            {
                "timestamp_utc": frame.timestamp_utc,
                "entry_model_id": "ENTRY_FVG_RESPECT_CONTINUATION",
                "eligible_entry_models": [
                    "ENTRY_FVG_RESPECT_CONTINUATION",
                    "ENTRY_OB_CONTINUATION",
                ],
                "entry_outcome": 1.2,
                "decision_frame": frame.to_dict(),
            }
        ]
    )
    policy = pd.DataFrame(
        [
            {
                "entry_model_id": "ENTRY_FVG_RESPECT_CONTINUATION",
                "label": "PREFERRED",
            }
        ]
    )

    replay = adversarial_replay(logs, artifacts, policy)

    assert replay.iloc[0]["best_entry_model"] == "ENTRY_FVG_RESPECT_CONTINUATION"
    assert replay.iloc[0]["regret"] == 0.0
    assert replay.iloc[0]["score_drift"] == 0.5
    assert bool(replay.iloc[0]["policy_alignment"]) is True


def test_stress_test_groups_by_regime():
    replay_df = pd.DataFrame(
        [
            {
                "market_profile_state": "ACCUMULATION",
                "session_profile": "PROFILE_1A",
                "liquidity_bias_side": "UP",
                "entry_model_id": "ENTRY_A",
                "regret": 1.0,
                "expected_R": 0.5,
                "entry_success": 1,
                "eligibility_drift": False,
                "score_drift": 0.1,
            },
            {
                "market_profile_state": "ACCUMULATION",
                "session_profile": "PROFILE_1A",
                "liquidity_bias_side": "UP",
                "entry_model_id": "ENTRY_A",
                "regret": 2.0,
                "expected_R": -0.5,
                "entry_success": 0,
                "eligibility_drift": True,
                "score_drift": 0.3,
            },
        ]
    )

    stress = stress_test_entries_by_regime(replay_df)
    assert len(stress) == 1
    row = stress.iloc[0]
    assert row["mean_regret"] == 1.5
    assert row["mean_expected_R"] == 0.0
    assert row["winrate"] == 0.5
    assert row["eligibility_drift_rate"] == 0.5
    assert row["score_drift_mean"] == 0.2
    assert row["sample_size"] == 2


def test_entry_drift_detection():
    stress_df = pd.DataFrame(
        [
            {
                "entry_model_id": "ENTRY_A",
                "mean_regret": 1.0,
                "score_drift_mean": 0.6,
                "eligibility_drift_rate": 0.2,
                "winrate": 0.2,
                "sample_size": 3,
            }
        ]
    )
    thresholds = {
        "regret": 0.5,
        "score_drift": 0.5,
        "eligibility_drift": 0.1,
        "winrate_min": 0.4,
        "min_samples": 5,
    }

    drift = detect_entry_drift(stress_df, thresholds)
    flags = drift.iloc[0]["drift_flags"]
    assert flags["regret"] is True
    assert flags["score_drift"] is True
    assert flags["eligibility_drift"] is True
    assert flags["winrate"] is True
    assert flags["sample_size"] is True
    assert bool(drift.iloc[0]["ok"]) is False


def test_entry_performance_attribution():
    replay_df = pd.DataFrame(
        [
            {
                "entry_model_id": "ENTRY_A",
                "market_profile_state": "ACCUMULATION",
                "session_profile": "PROFILE_1A",
                "expected_R": 1.0,
                "entry_success": 1,
            },
            {
                "entry_model_id": "ENTRY_A",
                "market_profile_state": "DISTRIBUTION",
                "session_profile": "PROFILE_1C",
                "expected_R": -0.5,
                "entry_success": 0,
            },
        ]
    )

    perf = attribute_entry_performance(replay_df)
    row = perf.iloc[0]
    assert row["total_trades"] == 2
    assert row["total_R"] == 0.5
    assert row["avg_R"] == 0.25
    assert row["winrate"] == 0.5
    assert row["best_regime"] == "ACCUMULATION"
    assert row["worst_regime"] == "DISTRIBUTION"
    assert row["best_session_profile"] == "PROFILE_1A"
    assert row["worst_session_profile"] == "PROFILE_1C"
