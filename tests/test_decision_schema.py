from engine.decision_actions import (
    ActionType,
    DecisionAction,
    DecisionOutcome,
    DecisionRecord,
)
from engine.decision_logger import build_decision_record


def test_action_type_enum_members():
    assert [
        ActionType.NO_TRADE.value,
        ActionType.OPEN_LONG.value,
        ActionType.OPEN_SHORT.value,
        ActionType.MANAGE_POSITION.value,
    ] == ["NO_TRADE", "OPEN_LONG", "OPEN_SHORT", "MANAGE_POSITION"]


def test_decision_action_and_outcome_instantiation():
    action = DecisionAction(
        action_type=ActionType.OPEN_LONG,
        entry_model_id="ENTRY_X",
        direction="LONG",
        size_bucket="MEDIUM",
    )
    outcome = DecisionOutcome(
        realized_R=1.25,
        max_adverse_excursion=-0.2,
        max_favorable_excursion=2.4,
        time_in_trade_bars=5,
        drawdown_impact=-0.1,
    )
    record = DecisionRecord(
        decision_id="dec1",
        timestamp_utc="2026-02-04T12:00:00Z",
        bar_index=10,
        state_ref="runA-10",
        action=action,
        outcome=outcome,
        metadata={"symbol": "ES", "timeframe": "M5"},
    )

    assert record.action.action_type is ActionType.OPEN_LONG
    assert record.action.entry_model_id == "ENTRY_X"
    assert record.outcome.realized_R == 1.25
    assert record.metadata["symbol"] == "ES"


def test_build_decision_record_from_log_entry():
    log_entry = {
        "decision_id": "dec_123",
        "timestamp_utc": "2026-02-04T13:00:00Z",
        "bar_index": 42,
        "run_id": "run_1",
        "symbol": "ES",
        "timeframe": "M5",
        "session_regime": "RTH",
        "macro_regimes": ["MACRO_ON"],
        "action": "LONG",
        "entry_model_id": "ENTRY_SWEEP_DISPLACEMENT_REVERSAL",
        "size_bucket": "SMALL",
        "outcome": {
            "pnl": 1.2,
            "max_drawdown": -0.3,
            "max_favorable_excursion": 2.4,
            "holding_period_bars": 6,
        },
    }

    record = build_decision_record(log_entry)

    assert record.decision_id == "dec_123"
    assert record.state_ref == "run_1"
    assert record.action.action_type is ActionType.OPEN_LONG
    assert record.action.direction == "LONG"
    assert record.action.entry_model_id == "ENTRY_SWEEP_DISPLACEMENT_REVERSAL"
    assert record.action.size_bucket == "SMALL"
    assert record.outcome is not None
    assert record.outcome.max_favorable_excursion == 2.4
    assert record.outcome.time_in_trade_bars == 6
    assert record.metadata["symbol"] == "ES"


def test_decision_record_fields_stable():
    expected_fields = {
        "decision_id",
        "timestamp_utc",
        "bar_index",
        "state_ref",
        "action",
        "outcome",
        "metadata",
    }
    assert set(DecisionRecord.__dataclass_fields__.keys()) == expected_fields
