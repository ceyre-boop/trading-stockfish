import os

import pandas as pd
import pytest

from engine.ml_brain import (
    BrainModelArtifacts,
    BrainScore,
    load_brain_models,
    save_brain_models,
    score_combo,
    train_brain_models,
)


def _synthetic_dataset():
    return pd.DataFrame(
        [
            {
                "strategy_id": "good-strat",
                "entry_model_id": "model-a",
                "session": "RTH",
                "macro": "RISK_ON",
                "vol": "HIGH",
                "trend": "UP",
                "liquidity": "DEEP",
                "tod": "OPEN",
                "sample_size": 20,
                "win_rate": 0.8,
                "avg_reward": 2.0,
                "reward_variance": 0.1,
                "reward_std": 0.316,
                "stability_mean_5": 1.5,
                "stability_std_5": 0.2,
            },
            {
                "strategy_id": "bad-strat",
                "entry_model_id": "model-b",
                "session": "ETH",
                "macro": "RISK_OFF",
                "vol": "LOW",
                "trend": "DOWN",
                "liquidity": "THIN",
                "tod": "OVERNIGHT",
                "sample_size": 20,
                "win_rate": 0.2,
                "avg_reward": -1.0,
                "reward_variance": 0.5,
                "reward_std": 0.707,
                "stability_mean_5": -0.5,
                "stability_std_5": 0.3,
            },
            {
                "strategy_id": "neutral-strat",
                "entry_model_id": "model-c",
                "session": "RTH",
                "macro": "RISK_NEUTRAL",
                "vol": "MEDIUM",
                "trend": "FLAT",
                "liquidity": "NORMAL",
                "tod": "CLOSE",
                "sample_size": 15,
                "win_rate": 0.55,
                "avg_reward": 0.1,
                "reward_variance": 0.2,
                "reward_std": 0.447,
                "stability_mean_5": 0.05,
                "stability_std_5": 0.1,
            },
        ]
    )


def _score_inputs():
    good_cv = {
        "session": "RTH",
        "macro": "RISK_ON",
        "vol": "HIGH",
        "trend": "UP",
        "liquidity": "DEEP",
        "tod": "OPEN",
        "sample_size": 20,
        "win_rate": 0.8,
        "avg_reward": 2.0,
        "reward_variance": 0.1,
        "reward_std": 0.316,
        "stability_mean_5": 1.5,
        "stability_std_5": 0.2,
    }
    bad_cv = {
        "session": "ETH",
        "macro": "RISK_OFF",
        "vol": "LOW",
        "trend": "DOWN",
        "liquidity": "THIN",
        "tod": "OVERNIGHT",
        "sample_size": 20,
        "win_rate": 0.2,
        "avg_reward": -1.0,
        "reward_variance": 0.5,
        "reward_std": 0.707,
        "stability_mean_5": -0.5,
        "stability_std_5": 0.3,
    }
    return good_cv, bad_cv


def test_ml_brain_trains_on_synthetic_dataset():
    data = _synthetic_dataset()
    artifacts = train_brain_models(
        data, thresholds={"win_rate_min": 0.5, "avg_reward_min": 0.0}
    )

    assert isinstance(artifacts, BrainModelArtifacts)
    assert artifacts.classifier is not None
    assert artifacts.regressor is not None
    assert artifacts.training_metadata.get("metrics") is not None


def test_ml_brain_scores_known_good_vs_bad_combos_differently():
    data = _synthetic_dataset()
    artifacts = train_brain_models(data)

    good_cv, bad_cv = _score_inputs()

    good_score = score_combo("good-strat", "model-a", good_cv, artifacts)
    bad_score = score_combo("bad-strat", "model-b", bad_cv, artifacts)

    assert isinstance(good_score, BrainScore)
    assert isinstance(bad_score, BrainScore)
    assert good_score.prob_good > bad_score.prob_good
    assert good_score.expected_reward > bad_score.expected_reward


def test_ml_brain_saves_and_loads_artifacts_deterministically(
    tmp_path: os.PathLike[str],
):
    data = _synthetic_dataset()
    artifacts = train_brain_models(data)

    good_cv, bad_cv = _score_inputs()

    save_brain_models(artifacts, tmp_path)
    loaded = load_brain_models(tmp_path)

    good_score_before = score_combo("good-strat", "model-a", good_cv, artifacts)
    good_score_after = score_combo("good-strat", "model-a", good_cv, loaded)

    bad_score_before = score_combo("bad-strat", "model-b", bad_cv, artifacts)
    bad_score_after = score_combo("bad-strat", "model-b", bad_cv, loaded)

    assert good_score_after.prob_good == pytest.approx(good_score_before.prob_good)
    assert bad_score_after.prob_good == pytest.approx(bad_score_before.prob_good)
    assert good_score_after.expected_reward == pytest.approx(
        good_score_before.expected_reward
    )
    assert bad_score_after.expected_reward == pytest.approx(
        bad_score_before.expected_reward
    )
