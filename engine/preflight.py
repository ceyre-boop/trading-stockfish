"""Phase 11 pre-flight checker (LIVE mode gatekeeper).

Deterministic, idempotent checks that gate entry into LIVE mode.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from engine.modes import Mode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "preflight"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MIN_FREE_GB = 1.0
REQUIRED_ENV_VARS: List[str] = []  # extend as needed
EXPECTED_PY_VERSION = os.environ.get("PREFLIGHT_PY_VERSION")  # e.g., "3.12"


@dataclass
class PreflightResult:
    passed: bool
    failures: List[str]
    timestamp: dt.datetime

    def to_json(self) -> dict:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "timestamp": self.timestamp.isoformat(),
        }


def run_cmd(cmd: List[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _check_test_suite(root: Path) -> Optional[str]:
    try:
        marker = root / "logs" / "tests" / "latest.json"
        if not marker.exists():
            return "test_suite_status_missing"
        data = json.loads(marker.read_text(encoding="utf-8"))
        status = str(data.get("status", "")).upper()
        if status != "PASS":
            return "test_suite_not_green"
        return None
    except Exception:
        return "test_suite_check_error"


def _check_clock_sync(root: Path) -> Optional[str]:
    try:
        script = root / "scripts" / "check_clock_sync.ps1"
        if not script.exists():
            return "clock_sync_script_missing"
        code, _, _ = run_cmd(["powershell", "-File", str(script)])
        if code != 0:
            return "clock_sync_failed"
        return None
    except Exception:
        return "clock_sync_error"


def _check_active_policy(root: Path) -> Optional[str]:
    try:
        policy_path = root / "policy_config.json"
        if not policy_path.exists():
            return "policy_missing"
        data = json.loads(policy_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return "policy_invalid_format"
        if "version" not in data or "hash" not in data:
            return "policy_version_or_hash_missing"
        return None
    except Exception:
        return "policy_validation_error"


def _check_safe_mode(root: Path) -> Optional[str]:
    try:
        marker = root / "logs" / "safe_mode_state.txt"
        if not marker.exists():
            return "safe_mode_unknown"
        state = marker.read_text(encoding="utf-8").strip().upper()
        if state not in {"ON", "OFF"}:
            return "safe_mode_invalid"
        if state == "ON":
            return "safe_mode_active"
        return None
    except Exception:
        return "safe_mode_error"


def _check_connectors(root: Path) -> Optional[str]:
    try:
        script = root / "scripts" / "validate_connectors.py"
        if not script.exists():
            return "connector_validator_missing"
        code, _, _ = run_cmd([sys.executable, str(script)])
        if code != 0:
            return "connectors_unhealthy"
        return None
    except Exception:
        return "connectors_check_error"


def _check_disk_space_and_dirs(root: Path) -> Optional[str]:
    try:
        usage = shutil.disk_usage(root)
        free_gb = usage.free / (1024**3)
        if free_gb < MIN_FREE_GB:
            return "disk_space_low"
        for rel in ["storage", "logs"]:
            path = root / rel
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".preflight_probe"
            try:
                probe.write_text("ok", encoding="utf-8")
                probe.unlink(missing_ok=True)
            except Exception:
                return f"{rel}_not_writable"
        return None
    except Exception:
        return "disk_or_dir_check_error"


def _check_env_and_python(root: Path) -> Optional[str]:
    try:
        missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
        if missing:
            return "env_vars_missing:" + ",".join(missing)
        if root.name not in Path(sys.prefix).parts:
            return "venv_not_active"
        if EXPECTED_PY_VERSION:
            current = f"{sys.version_info.major}.{sys.version_info.minor}"
            if current != EXPECTED_PY_VERSION:
                return "python_version_mismatch"
        return None
    except Exception:
        return "env_check_error"


def _check_kill_switch() -> Optional[str]:
    try:
        from engine.guardrails import kill_switch  # noqa: WPS433

        if not callable(kill_switch):
            return "kill_switch_not_callable"
        return None
    except Exception:
        return "kill_switch_unavailable"


def _check_weekly_cycle(root: Path) -> Optional[str]:
    try:
        reports_dir = root / "reports"
        if not reports_dir.exists():
            return "weekly_cycle_report_missing"
        reports = sorted(reports_dir.glob("weekly_report_*.md"))
        if not reports:
            return "weekly_cycle_report_missing"
        latest = reports[-1]
        if latest.stat().st_size == 0:
            return "weekly_cycle_report_empty"
        return None
    except Exception:
        return "weekly_cycle_check_error"


def _check_storage_integrity(root: Path) -> Optional[str]:
    try:
        storage = root / "storage"
        if not storage.exists():
            return "storage_missing"
        zero_files = []
        for f in storage.rglob("*.parquet"):
            try:
                if f.stat().st_size == 0:
                    zero_files.append(str(f))
            except FileNotFoundError:
                continue
        if zero_files:
            return "corrupted_parquet:" + ",".join(zero_files)
        return None
    except Exception:
        return "storage_integrity_error"


def run_preflight(mode: Mode) -> PreflightResult:
    timestamp = dt.datetime.now(dt.UTC)
    failures: List[str] = []

    checks = [
        _check_test_suite,
        _check_clock_sync,
        _check_active_policy,
        _check_safe_mode,
        _check_connectors,
        _check_disk_space_and_dirs,
        _check_env_and_python,
        lambda root: _check_kill_switch(),
        _check_weekly_cycle,
        _check_storage_integrity,
    ]

    for check in checks:
        message = check(PROJECT_ROOT)
        if message:
            failures.append(message)

    result = PreflightResult(
        passed=len(failures) == 0, failures=failures, timestamp=timestamp
    )

    log_payload = result.to_json() | {"mode": mode.value}
    log_path = LOG_DIR / f"preflight_{timestamp:%Y%m%d_%H%M%S}.json"
    try:
        log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
    except Exception:
        pass

    return result
