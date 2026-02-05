import json
import os
from pathlib import Path

import pandas as pd

from engine.brain_policy_builder import build_brain_policy
from engine.ml_brain import train_brain_models


def _base_rows():
    return [
        {
            "strategy_id": "s-good",
            "entry_model_id": "e-good",
            "session": "RTH",
            "macro": "RISK_ON",
            "vol": "HIGH",
            "trend": "UP",
            "liquidity": "DEEP",
            "tod": "OPEN",
            "sample_size": 10,
            "win_rate": 0.9,
            "avg_reward": 1.5,
            "reward_variance": 0.1,
            "reward_std": 0.316,
            "stability_mean_5": 1.0,
            "stability_std_5": 0.1,
        },
        {
            "strategy_id": "s-bad",
            "entry_model_id": "e-bad",
            "session": "ETH",
            "macro": "RISK_OFF",
            "vol": "LOW",
            "trend": "DOWN",
            "liquidity": "THIN",
            "tod": "OVERNIGHT",
            "sample_size": 1,
            "win_rate": 0.2,
            "avg_reward": -0.5,
            "reward_variance": 0.2,
            "reward_std": 0.447,
            "stability_mean_5": -0.2,
            "stability_std_5": 0.2,
        },
    ]


def _train_artifacts():
    data = pd.DataFrame(_base_rows())
    return train_brain_models(data)


def test_brain_policy_marks_low_sample_as_DISABLED(tmp_path: Path):
    rows = _base_rows()
    df = pd.DataFrame([rows[1]])
    artifacts = train_brain_models(pd.DataFrame(rows))

    thresholds = {"min_samples": 5, "output_dir": str(tmp_path)}
    policy = build_brain_policy(df, artifacts, thresholds)

    assert len(policy) == 1
    assert policy.iloc[0]["label"] == "DISABLED"


def test_brain_policy_marks_high_reward_high_confidence_as_PREFERRED(tmp_path: Path):
    rows = _base_rows()
    df = pd.DataFrame([rows[0]])
    artifacts = train_brain_models(pd.DataFrame(rows))

    thresholds = {
        "min_samples": 1,
        "prob_good_min": 0.01,
        "expected_reward_min": -0.5,
        "output_dir": str(tmp_path),
    }
    policy = build_brain_policy(df, artifacts, thresholds)

    assert len(policy) == 1
    assert policy.iloc[0]["label"] == "PREFERRED"
    assert policy.iloc[0]["prob_good"] >= 0.0


def test_brain_policy_is_deterministic(tmp_path: Path):
    df = pd.DataFrame(_base_rows())
    artifacts = _train_artifacts()

    thresholds = {"output_dir": str(tmp_path)}

    policy_first = build_brain_policy(df, artifacts, thresholds)
    policy_second = build_brain_policy(df, artifacts, thresholds)

    pd.testing.assert_frame_equal(policy_first, policy_second)


def test_brain_policy_saves_artifact(tmp_path: Path):
    df = pd.DataFrame(_base_rows())
    artifacts = _train_artifacts()

    thresholds = {"output_dir": str(tmp_path)}
    policy = build_brain_policy(df, artifacts, thresholds)

    path = os.path.join(str(tmp_path), "brain_policy.json")
    assert os.path.exists(path)

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert "metadata" in payload
    assert "policy" in payload
    assert payload["metadata"].get("thresholds") is not None
    assert len(payload["policy"]) == len(policy)
