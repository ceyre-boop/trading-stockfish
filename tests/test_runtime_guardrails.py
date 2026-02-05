from __future__ import annotations

from pathlib import Path
from typing import List

from pytest import MonkeyPatch

import engine.guardrails as guardrails
from engine.guardrails import GuardrailDecision  # noqa: F401
from engine.guardrails import apply_guardrail_decision, check_runtime_limits


class DummyAdapter:
    def __init__(self):
        self.disabled = False
        self.name = "dummy"
        self.calls: List[str] = []

    def disable_orders(self):
        self.calls.append("disable_orders")
        self.disabled = True


def _patch_paths(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    guard_dir = tmp_path / "logs" / "guardrails"
    guard_dir.mkdir(parents=True, exist_ok=True)
    safe_state = tmp_path / "logs" / "safe_mode_state.txt"
    monkeypatch.setattr(
        guardrails,
        "GUARDRAIL_LOG_DIR",
        guard_dir,
        raising=False,
    )
    monkeypatch.setattr(
        guardrails,
        "SAFE_MODE_STATE",
        safe_state,
        raising=False,
    )


def _assert_decision(decision: GuardrailDecision) -> GuardrailDecision:
    assert isinstance(decision, GuardrailDecision)
    return decision


def test_guardrail_triggers_on_daily_loss(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
):
    _patch_paths(monkeypatch, tmp_path)
    decision = _assert_decision(
        check_runtime_limits(
            state={"max_daily_loss": 500},
            metrics={"realized_loss": 400, "unrealized_loss": 200},
        )
    )

    assert decision.triggered is True
    assert decision.guardrail_type == "max_daily_loss"
    assert decision.safe_mode_required is True


def test_guardrail_triggers_on_connector_failure(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
):
    _patch_paths(monkeypatch, tmp_path)
    decision = _assert_decision(
        check_runtime_limits(
            state={"max_connector_failures": 1},
            metrics={"connector_failures": 2},
        )
    )

    assert decision.triggered is True
    assert decision.guardrail_type == "connector_failure"
    assert decision.safe_mode_required is True


def test_guardrail_triggers_on_heartbeat_timeout(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
):
    _patch_paths(monkeypatch, tmp_path)
    decision = _assert_decision(
        check_runtime_limits(
            state={"max_heartbeat_age_sec": 5},
            metrics={"heartbeat_age_sec": 15},
        )
    )

    assert decision.triggered is True
    assert decision.guardrail_type == "heartbeat_timeout"
    assert decision.safe_mode_required is True


def test_guardrail_blocks_orders_in_safe_mode(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
):
    _patch_paths(monkeypatch, tmp_path)
    adapter = DummyAdapter()
    decision = _assert_decision(
        check_runtime_limits(
            state={},
            metrics={"safe_mode_active": True},
        )
    )

    result = apply_guardrail_decision(decision, adapter=adapter)

    assert decision.triggered is True
    assert decision.guardrail_type == "safe_mode_active"
    assert adapter.disabled is True
    assert "disable_orders" in adapter.calls
    assert result["applied"] is True


def test_guardrail_triggers_on_max_leverage(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
):
    _patch_paths(monkeypatch, tmp_path)
    decision = _assert_decision(
        check_runtime_limits(
            state={"max_leverage": 2.0},
            metrics={"leverage": 3.0},
        )
    )

    assert decision.triggered is True
    assert decision.guardrail_type == "max_leverage"
    assert decision.safe_mode_required is True


def test_guardrail_triggers_on_order_frequency(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
):
    _patch_paths(monkeypatch, tmp_path)
    decision = _assert_decision(
        check_runtime_limits(
            state={"max_orders_per_min": 10},
            metrics={"orders_per_min": 12},
        )
    )

    assert decision.triggered is True
    assert decision.guardrail_type == "max_order_frequency"
    assert decision.safe_mode_required is False


def test_guardrail_triggers_on_slippage(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
):
    _patch_paths(monkeypatch, tmp_path)
    decision = _assert_decision(
        check_runtime_limits(
            state={"max_slippage": 1.5},
            metrics={"estimated_slippage": 2.0},
        )
    )

    assert decision.triggered is True
    assert decision.guardrail_type == "max_slippage"
    assert decision.safe_mode_required is False


def test_guardrail_no_violation_passes(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
):
    _patch_paths(monkeypatch, tmp_path)
    adapter = DummyAdapter()
    decision = _assert_decision(
        check_runtime_limits(
            state={
                "max_daily_loss": 500,
                "max_position_size": 10,
                "max_leverage": 2.0,
                "max_orders_per_min": 10,
                "max_slippage": 1.5,
                "max_heartbeat_age_sec": 10,
                "max_connector_failures": 3,
            },
            metrics={
                "realized_loss": 10,
                "unrealized_loss": 5,
                "position_size": 1,
                "leverage": 1.0,
                "orders_per_min": 5,
                "estimated_slippage": 1.0,
                "heartbeat_age_sec": 1,
                "connector_failures": 0,
                "safe_mode_active": False,
                "kill_switch_active": False,
            },
        )
    )

    result = apply_guardrail_decision(decision, adapter=adapter)

    assert decision.triggered is False
    assert result["applied"] is False
    assert adapter.disabled is False
