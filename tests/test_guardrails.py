from pathlib import Path

import pytest

import engine.guardrails as guardrails
from engine.guardrails import (
    GuardrailDecision,
    apply_guardrail_decision,
    check_runtime_limits,
    kill_switch,
)
from engine.modes import LiveAdapter, Mode


def _patch_paths(monkeypatch, tmp_path: Path) -> None:
    guard_dir = tmp_path / "logs" / "guardrails"
    guard_dir.mkdir(parents=True, exist_ok=True)
    safe_state = tmp_path / "logs" / "safe_mode_state.txt"
    monkeypatch.setattr(guardrails, "GUARDRAIL_LOG_DIR", guard_dir, raising=False)
    monkeypatch.setattr(guardrails, "SAFE_MODE_STATE", safe_state, raising=False)


def test_kill_switch_stops_order_flow_in_LIVE_MODE():
    adapter = LiveAdapter()
    assert adapter.disabled is False
    kill_info = kill_switch(Mode.LIVE, adapter=adapter)
    assert kill_info["disabled"] is True
    with pytest.raises(RuntimeError):
        adapter.place_order({})


def test_guardrail_triggers_and_activates_safe_mode(monkeypatch, tmp_path: Path):
    _patch_paths(monkeypatch, tmp_path)
    adapter = LiveAdapter()

    decision: GuardrailDecision = check_runtime_limits(
        state={"max_daily_loss": 100.0},
        metrics={"realized_loss": 120.0},
    )

    assert decision.triggered is True
    assert decision.guardrail_type == "max_daily_loss"
    result = apply_guardrail_decision(decision, adapter=adapter)

    assert result["applied"] is True
    assert result["safe_mode_state"] == "ON"
    assert adapter.disabled is True


def test_guardrail_passes_when_within_limits(monkeypatch, tmp_path: Path):
    _patch_paths(monkeypatch, tmp_path)

    decision: GuardrailDecision = check_runtime_limits(
        state={"max_daily_loss": 100.0, "max_position_size": 10},
        metrics={"realized_loss": 10.0, "position_size": 1},
    )

    assert decision.triggered is False
    assert decision.guardrail_type == "none"
