from pathlib import Path

from engine.jobs.weekly_cycle import run_weekly_cycle
from engine.modes import Mode


def test_weekly_cycle_runs_without_side_effects_in_SIMULATION(tmp_path: Path):
    result = run_weekly_cycle(mode_str="SIMULATION", days=3, report_dir=tmp_path)
    assert result["mode"] == Mode.SIMULATION.value
    assert result["report_path"].exists()
    assert result["promoted"] is False


def test_weekly_cycle_generates_report_and_does_not_promote_on_FAIL(tmp_path: Path):
    result = run_weekly_cycle(
        mode_str="SIMULATION", days=2, report_dir=tmp_path, allow_promote=False
    )
    assert result["report_path"].exists()
    assert result["promo