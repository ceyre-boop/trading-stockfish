import json
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

from pytest import MonkeyPatch

import engine.preflight as preflight
from engine.modes import Mode
from engine.preflight import PreflightResult, run_preflight


def _seed_health(tmp_path: Path) -> None:
    tests_dir = tmp_path / "logs" / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "latest.json").write_text(
        json.dumps({"status": "PASS"}),
        encoding="utf-8",
    )

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "check_clock_sync.ps1").write_text(
        "exit 0",
        encoding="utf-8",
    )
    (scripts_dir / "validate_connectors.py").write_text(
        "print('ok')",
        encoding="utf-8",
    )

    policy_payload = {"version": "1.0", "hash": "abc"}
    (tmp_path / "policy_config.json").write_text(
        json.dumps(policy_payload),
        encoding="utf-8",
    )

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "safe_mode_state.txt").write_text(
        "OFF",
        encoding="utf-8",
    )

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "weekly_report_20250101.md").write_text(
        "ok",
        encoding="utf-8",
    )

    storage_dir = tmp_path / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    (storage_dir / "sample.parquet").write_bytes(b"data")


def _configure_preflight(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    run_cmd_results: Dict[str, Tuple[int, str, str]] | None = None,
) -> Dict[str, Tuple[int, str, str]]:
    run_cmd_results = run_cmd_results or {}
    _seed_health(tmp_path)

    monkeypatch.setattr(
        preflight,
        "PROJECT_ROOT",
        tmp_path,
        raising=False,
    )
    log_dir = tmp_path / "logs" / "preflight"
    log_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(preflight, "LOG_DIR", log_dir, raising=False)

    class DiskUsage(NamedTuple):
        total: int
        used: int
        free: int

    def fake_disk_usage(_: Path) -> DiskUsage:
        return DiskUsage(10**12, 1, 5 * 10**11)

    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        fake_disk_usage,
    )

    def fake_run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
        target = str(cmd[-1])
        if "check_clock_sync.ps1" in target:
            return run_cmd_results.get("clock", (0, "", ""))
        if "validate_connectors.py" in target:
            return run_cmd_results.get("connectors", (0, "", ""))
        return (0, "", "")

    monkeypatch.setattr(preflight, "run_cmd", fake_run_cmd)

    def _ignore_env(_: Path) -> None:
        return None

    monkeypatch.setattr(
        preflight,
        "_check_env_and_python",
        _ignore_env,
        raising=False,
    )
    monkeypatch.setattr(
        preflight,
        "_check_kill_switch",
        lambda: None,
        raising=False,
    )
    return run_cmd_results


def test_preflight_fails_on_clock_desync(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
):
    _configure_preflight(
        monkeypatch,
        tmp_path,
        run_cmd_results={"clock": (1, "", "skew")},
    )

    result: PreflightResult = run_preflight(Mode.LIVE)

    assert result.passed is False
    assert "clock_sync_failed" in result.failures


def test_preflight_fails_on_invalid_policy(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
):
    _configure_preflight(monkeypatch, tmp_path)
    (tmp_path / "policy_config.json").write_text("not-json", encoding="utf-8")

    result: PreflightResult = run_preflight(Mode.LIVE)

    assert result.passed is False
    assert "policy_validation_error" in result.failures


def test_preflight_fails_on_connector_failure(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
):
    _configure_preflight(
        monkeypatch,
        tmp_path,
        run_cmd_results={"connectors": (2, "", "fail")},
    )

    result: PreflightResult = run_preflight(Mode.LIVE)

    assert result.passed is False
    assert "connectors_unhealthy" in result.failures


def test_preflight_passes_when_all_checks_valid(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
):
    _configure_preflight(monkeypatch, tmp_path)

    result: PreflightResult = run_preflight(Mode.PAPER)

    assert result.passed is True
    assert result.failures == []
    assert preflight.LOG_DIR.exists()
