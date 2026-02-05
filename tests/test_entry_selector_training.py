import numpy as np
import pandas as pd

from engine.decision_frame import DecisionFrame
from engine.entry_selector_dataset import build_entry_selector_dataset
from engine.entry_selector_model import train_entry_selector_model
from engine.entry_selector_scoring import score_entry_selector


def _make_log(ts: str, entry_id: str, eligible: list[str], outcome: float | None):
    frame = DecisionFrame(
        market_profile_state="DISTRIBUTION",
        session_profile="PROFILE_1C",
        liquidity_frame={"bias": "UP", "distance_bucket": "NEAR", "sweep_state": "ANY"},
        vol_regime="NORMAL",
        trend_regime="UP",
        entry_signals_present={"fvg": True},
        timestamp_utc=ts,
        symbol="EURUSD",
    )
    return {
        "timestamp_utc": ts,
        "symbol": "EURUSD",
        "entry_model_id": entry_id,
        "eligible_entry_models": eligible,
        "entry_outcome": outcome,
        "decision_frame": frame.to_dict(),
    }


def test_entry_selector_dataset_shapes_correctly():
    logs = pd.DataFrame(
        [
            _make_log(
                "2024-01-01T00:00:00Z",
                "ENTRY_FVG_RESPECT_CONTINUATION",
                ["ENTRY_FVG_RESPECT_CONTINUATION", "ENTRY_OB_CONTINUATION"],
                1.5,
            ),
            _make_log(
                "2024-01-01T00:05:00Z",
                "ENTRY_OB_CONTINUATION",
                ["ENTRY_FVG_RESPECT_CONTINUATION", "ENTRY_OB_CONTINUATION"],
                -0.5,
            ),
        ]
    )

    ds = build_entry_selector_dataset(logs)

    assert len(ds) == 4  # 2 decisions * 2 eligible entries
    assert set(ds["entry_model_id"]) == {
        "ENTRY_FVG_RESPECT_CONTINUATION",
        "ENTRY_OB_CONTINUATION",
    }
    chosen_rows = ds[ds["chosen_flag"] == 1]
    assert len(chosen_rows) == 2
    assert chosen_rows["entry_outcome_R"].notna().all()


def test_entry_selector_model_trains_on_synthetic_data():
    logs = pd.DataFrame(
        [
            _make_log(
                "2024-01-01T00:00:00Z",
                "ENTRY_FVG_RESPECT_CONTINUATION",
                ["ENTRY_FVG_RESPECT_CONTINUATION"],
                1.0,
            ),
            _make_log(
                "2024-01-01T00:01:00Z",
                "ENTRY_OB_CONTINUATION",
                ["ENTRY_OB_CONTINUATION"],
                -0.2,
            ),
        ]
    )
    ds = build_entry_selector_dataset(logs)
    artifacts = train_entry_selector_model(ds, random_state=123)
    assert artifacts.classifier is not None
    assert artifacts.regressor is not None
    assert artifacts.encoders.get("categorical") is not None


def test_entry_selector_scoring_produces_probabilities():
    logs = pd.DataFrame(
        [
            _make_log(
                "2024-01-01T00:00:00Z",
                "ENTRY_FVG_RESPECT_CONTINUATION",
                ["ENTRY_FVG_RESPECT_CONTINUATION"],
                1.0,
            ),
            _make_log(
                "2024-01-01T00:01:00Z",
                "ENTRY_OB_CONTINUATION",
                ["ENTRY_OB_CONTINUATION"],
                0.2,
            ),
        ]
    )
    ds = build_entry_selector_dataset(logs)
    artifacts = train_entry_selector_model(ds, random_state=99)

    frame = DecisionFrame(
        market_profile_state="DISTRIBUTION",
        session_profile="PROFILE_1C",
        liquidity_frame={"bias": "UP", "distance_bucket": "NEAR", "sweep_state": "ANY"},
        vol_regime="NORMAL",
        trend_regime="UP",
        entry_signals_present={"fvg": True},
    )

    scores = score_entry_selector(frame, ["ENTRY_FVG_RESPECT_CONTINUATION"], artifacts)
    assert "ENTRY_FVG_RESPECT_CONTINUATION" in scores
    assert "prob_select" in scores["ENTRY_FVG_RESPECT_CONTINUATION"]
    assert "expected_R" in scores["ENTRY_FVG_RESPECT_CONTINUATION"]


def test_entry_selector_determinism():
    logs = pd.DataFrame(
        [
            _make_log(
                "2024-01-01T00:00:00Z",
                "ENTRY_FVG_RESPECT_CONTINUATION",
                ["ENTRY_FVG_RESPECT_CONTINUATION", "ENTRY_OB_CONTINUATION"],
                1.0,
            ),
            _make_log(
                "2024-01-01T00:02:00Z",
                "ENTRY_OB_CONTINUATION",
                ["ENTRY_FVG_RESPECT_CONTINUATION", "ENTRY_OB_CONTINUATION"],
                -0.5,
            ),
        ]
    )
    ds = build_entry_selector_dataset(logs)

    artifacts_a = train_entry_selector_model(ds, random_state=7)
    artifacts_b = train_entry_selector_model(ds, random_state=7)

    frame = DecisionFrame(
        market_profile_state="DISTRIBUTION",
        session_profile="PROFILE_1C",
        liquidity_frame={"bias": "UP", "distance_bucket": "NEAR", "sweep_state": "ANY"},
        vol_regime="NORMAL",
        trend_regime="UP",
        entry_signals_present={"fvg": True},
    )

    scores_a = score_entry_selector(
        frame, ["ENTRY_FVG_RESPECT_CONTINUATION", "ENTRY_OB_CONTINUATION"], artifacts_a
    )
    scores_b = score_entry_selector(
        frame, ["ENTRY_FVG_RESPECT_CONTINUATION", "ENTRY_OB_CONTINUATION"], artifacts_b
    )

    assert scores_a == scores_b
