import pandas as pd

from engine.ml_brain_dataset import build_brain_dataset


def _cv(session, macro, vol, trend, liq, tod):
    return {
        "session": session,
        "macro": macro,
        "vol": vol,
        "trend": trend,
        "liquidity": liq,
        "tod": tod,
    }


def test_ml_brain_dataset_groups_by_strategy_entry_condition():
    data = pd.DataFrame(
        [
            {
                "strategy_id": "s1",
                "entry_model_id": "e1",
                "outcome": 1.0,
                "condition_vector": _cv("RTH", "RISK_ON", "HIGH", "UP", "DEEP", "OPEN"),
            },
            {
                "strategy_id": "s1",
                "entry_model_id": "e1",
                "outcome": -1.0,
                "condition_vector": _cv("RTH", "RISK_ON", "HIGH", "UP", "DEEP", "OPEN"),
            },
            {
                "strategy_id": "s2",
                "entry_model_id": "e2",
                "outcome": 0.5,
                "condition_vector": _cv(
                    "ETH", "RISK_OFF", "LOW", "DOWN", "THIN", "OVERNIGHT"
                ),
            },
        ]
    )

    ds = build_brain_dataset(data)
    assert len(ds) == 2
    assert set(ds["strategy_id"]) == {"s1", "s2"}


def test_ml_brain_dataset_computes_win_rate_and_reward_correctly():
    data = pd.DataFrame(
        [
            {
                "strategy_id": "s1",
                "entry_model_id": "e1",
                "outcome": 1.0,
                "condition_vector": _cv("RTH", "RISK_ON", "HIGH", "UP", "DEEP", "OPEN"),
            },
            {
                "strategy_id": "s1",
                "entry_model_id": "e1",
                "outcome": -1.0,
                "condition_vector": _cv("RTH", "RISK_ON", "HIGH", "UP", "DEEP", "OPEN"),
            },
            {
                "strategy_id": "s1",
                "entry_model_id": "e1",
                "outcome": 3.0,
                "condition_vector": _cv("RTH", "RISK_ON", "HIGH", "UP", "DEEP", "OPEN"),
            },
        ]
    )

    ds = build_brain_dataset(data)
    row = ds.iloc[0]
    assert row["sample_size"] == 3
    assert abs(row["win_rate"] - (2 / 3)) < 1e-6
    assert abs(row["avg_reward"] - (1.0 - 1.0 + 3.0) / 3) < 1e-6
    assert abs(row["reward_variance"] - data["outcome"].var(ddof=0)) < 1e-6
    assert abs(row["reward_std"] - data["outcome"].std(ddof=0)) < 1e-6


def test_ml_brain_dataset_ignores_rows_without_strategy_ids():
    data = pd.DataFrame(
        [
            {
                "strategy_id": None,
                "entry_model_id": "e1",
                "outcome": 1.0,
                "condition_vector": _cv("RTH", "RISK_ON", "HIGH", "UP", "DEEP", "OPEN"),
            },
            {
                "strategy_id": "s1",
                "entry_model_id": None,
                "outcome": 2.0,
                "condition_vector": _cv("RTH", "RISK_ON", "HIGH", "UP", "DEEP", "OPEN"),
            },
            {
                "strategy_id": "s1",
                "entry_model_id": "e1",
                "outcome": None,
                "condition_vector": _cv("RTH", "RISK_ON", "HIGH", "UP", "DEEP", "OPEN"),
            },
            {
                "strategy_id": "s1",
                "entry_model_id": "e1",
                "outcome": 0.5,
                "condition_vector": _cv("RTH", "RISK_ON", "HIGH", "UP", "DEEP", "OPEN"),
            },
        ]
    )

    ds = build_brain_dataset(data)
    assert len(ds) == 1
    assert ds.iloc[0]["sample_size"] == 1


def test_ml_brain_dataset_expands_condition_vector():
    data = pd.DataFrame(
        [
            {
                "strategy_id": "s1",
                "entry_model_id": "e1",
                "outcome": 1.0,
                "condition_vector": _cv("RTH", "RISK_ON", "HIGH", "UP", "DEEP", "OPEN"),
            }
        ]
    )

    ds = build_brain_dataset(data)
    row = ds.iloc[0]
    assert row["session"] == "RTH"
    assert row["macro"] == "RISK_ON"
    assert row["vol"] == "HIGH"
    assert row["trend"] == "UP"
    assert row["liquidity"] == "DEEP"
    assert row["tod"] == "OPEN"


def test_dataset_includes_cognitive_features():
    data = pd.DataFrame(
        [
            {
                "strategy_id": "s1",
                "entry_model_id": "e1",
                "outcome": 1.0,
                "timestamp_utc": "2026-02-04T00:00:00Z",
                "condition_vector": _cv("RTH", "RISK_ON", "HIGH", "UP", "DEEP", "OPEN"),
                "decision_frame": {
                    "market_profile_state": "MANIPULATION",
                    "market_profile_confidence": 0.8,
                    "session_profile": "PROFILE_1B",
                    "session_profile_confidence": 0.7,
                    "liquidity_frame": {
                        "bias": "UP",
                        "primary_target": "PDH",
                        "distances": {"PDH": 5.0},
                        "swept": {"PDH": False},
                    },
                    "market_profile_evidence": {"displacement_score": 0.6},
                },
            }
        ]
    )

    ds = build_brain_dataset(data)
    row = ds.iloc[0]
    assert row["market_profile_state"] == "MANIPULATION"
    assert row["market_profile_confidence"] == 0.8
    assert row["session_profile"] == "PROFILE_1B"
    assert row["liquidity_bias_side"] == "UP"
    assert row["nearest_target_type"] == "PDH"
    assert row["nearest_target_distance"] == 5.0
    assert row["displacement_score"] == 0.6
