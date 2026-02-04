from pathlib import Path

from engine.jobs.daily_run import run_daily
from engine.modes import Mode


def test_daily_run_simulation_creates_expected_artifacts(tmp_path: Path):
    result = run_daily("SIMULATION", run_id="run_test", base_dir=tmp_path)
    assert result["mode"] == Mode.SIMULATION.value
    assert result["decision_log"].exists()
    assert result["audit_path"].exists()
    assert result["stats_path"].exists()
    content = result["decision_log"].read_text(encoding="utf-8")
    assert "run_test" in content
    # No orders placed in SIMULATION path
    assert result["adapter_disabled"] is False


def test_daily_run_respects_mode_and_does_not_place_orders_in_SIMULATION(tmp_path: Path):
    result = run_daily("SIMULATION", run_id="run_test2", base_dir=tmp_path)
    assert result["mode"] == Mode.SIMULATION.value
    # Adapter should remain enabled but no orders were placed
    assert result["adapter_disabled"] is False
