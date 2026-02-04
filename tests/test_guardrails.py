from pathlib import Path

import pytest

from engine.guardrails import kill_switch, preflight_check, runtime_limits
from engine.modes import LiveAdapter, Mode


def test_preflight_fails_when_tests_not_green_or_policy_missing(tmp_path: Path):
    ok, issues = preflight_check(
        tests_green=False,
        policy_path=tmp_path / "missing_policy.json",
        safe_mode_state=None,
        connectors_healthy=False,
    )
    assert not ok
    assert "test_suite_not_green" in issues
    assert "policy_missing" in issues
    assert "safe_mode_unknown" in issues
    assert "connectors_unhealthy" in issues


def test_kill_switch_stops_order_flow_in_LIVE_MODE():
    adapter = LiveAdapter()
    assert adapter.disabled is False
    kill_info = kill_switch(Mode.LIVE, adapter=adapter)
    assert kill_info["disabled"] is True
    with pytest.raises(RuntimeError):
        adapter.place_order({})


def test_runtime_limits_enforced():
    ok, issues = runtime_limits(
        pnl_today=-200.0, max_daily_loss=100.0, position=5, max_position=2
    )
    assert not ok
    assert "max_daily_loss_exceeded" in issues
    assert "max_position_exceeded" in issues
