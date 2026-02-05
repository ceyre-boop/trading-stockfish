import pandas as pd

from engine.decision_actions import (
    ActionType,
    DecisionAction,
    DecisionOutcome,
    DecisionRecord,
)
from engine.ev_dataset_builder import build_ev_dataset


def _decision_record():
    action = DecisionAction(
        action_type=ActionType.OPEN_LONG,
        entry_model_id="ENTRY_SWEEP_DISPLACEMENT_REVERSAL",
        direction="LONG",
        size_bucket="SMALL",
        stop_structure={"type": "ATR", "mult": 2.0},
        tp_structure={"rr": 2.5},
    )
    outcome = DecisionOutcome(
        realized_R=1.1,
        max_adverse_excursion=-0.3,
        max_favorable_excursion=2.2,
        time_in_trade_bars=7,
        drawdown_impact=-0.15,
    )
    return DecisionRecord(
        decision_id="dec_001",
        timestamp_utc="2026-02-04T12:00:00Z",
        bar_index=15,
        state_ref="runA-15",
        action=action,
        outcome=outcome,
        metadata={"symbol": "ES"},
    )


def _replay_logs():
    return pd.DataFrame(
        [
            {
                "decision_id": "dec_001",
                "timestamp_utc": "2026-02-04T12:00:00Z",
                "bar_index": 15,
                "decision_frame": {
                    "market_profile_state": "ACCUMULATION",
                    "market_profile_confidence": 0.6,
                    "session_profile": "PROFILE_1A",
                    "session_profile_confidence": 0.5,
                    "vol_regime": "NORMAL",
                    "trend_regime": "UP",
                    "liquidity_frame": {"bias": "UP"},
                    "condition_vector": {"session": "RTH", "vol": "NORMAL"},
                },
                "structural_context": {
                    "market_profile_state": "ACCUMULATION",
                    "liquidity_bias_side": "UP",
                    "vol_regime": "NORMAL",
                    "trend_regime": "UP",
                },
            }
        ]
    )


def test_ev_dataset_builder_columns_and_rows():
    rec = _decision_record()
    replay_logs = _replay_logs()

    df = build_ev_dataset(replay_logs, [rec], version="v1")

    required_columns = [
        "decision_id",
        "timestamp_utc",
        "bar_index",
        "state_ref",
        "dataset_version",
        "state_market_profile_state",
        "state_session_profile",
        "state_vol_regime",
        "state_trend_regime",
        "state_liquidity_bias",
        "action_type_id",
        "entry_model_id_idx",
        "direction_id",
        "size_bucket_id",
        "label_realized_R",
        "label_max_adverse_excursion",
        "label_max_favorable_excursion",
        "label_time_in_trade_bars",
        "label_drawdown_impact",
    ]

    for col in required_columns:
        assert col in df.columns

    assert len(df) == 1
    row = df.iloc[0]
    assert row["action_type_id"] == 1  # OPEN_LONG
    assert row["direction_id"] == 1  # LONG
    assert row["size_bucket_id"] == 1  # SMALL
    assert row["label_realized_R"] == 1.1
    assert row["state_market_profile_state"] == "ACCUMULATION"


def test_ev_dataset_encodings_are_deterministic():
    rec = _decision_record()
    replay_logs = _replay_logs()

    df1 = build_ev_dataset(replay_logs, [rec], version="v1")
    df2 = build_ev_dataset(replay_logs, [rec], version="v1")

    assert df1.equals(df2)


def test_no_future_leakage_columns():
    rec = _decision_record()
    replay_logs = _replay_logs()

    df = build_ev_dataset(replay_logs, [rec], version="v1")

    feature_cols = [c for c in df.columns if not c.startswith("label_")]
    label_cols = [c for c in df.columns if c.startswith("label_")]

    # Ensure label columns are segregated
    for col in label_cols:
        assert col not in feature_cols

    # Check features do not accidentally include outcome
    for col in feature_cols:
        assert "label" not in col
