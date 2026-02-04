"""Run PAPER Day 2 and perform post-run checks.

- Executes daily_run for PAPER mode with run-id day2.
- Logs stdout/stderr to logs/scheduled/daily_day2_manual.log.
- Verifies decisions/audits/stats for run_id=day2 in storage.
- Ensures no SAFE_MODE mention and no errors.
- Prints PASS/FAIL summary; exits non-zero on failure.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "scheduled"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "daily_day2_manual.log"


def _run_daily() -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        "-m",
        "engine.jobs.daily_run",
        "--mode",
        "PAPER",
        "--run-id",
        "day2",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc


def _write_log(proc: subprocess.CompletedProcess) -> None:
    payload = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    LOG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_storage_rows(path: Path, run_id: str) -> int:
    if not path.exists():
        return 0
    files = sorted(path.glob("*.parquet"))
    if not files:
        return 0
    frames: List[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if "run_id" not in df.columns:
            continue
        frames.append(df[df["run_id"] == run_id])
    if not frames:
        return 0
    return int(pd.concat(frames, ignore_index=True).shape[0])


def _has_safe_mode_indicator(proc: subprocess.CompletedProcess) -> bool:
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return "SAFE_MODE" in text.upper()


def main() -> None:
    proc = _run_daily()
    _write_log(proc)

    errors: List[str] = []

    if proc.returncode != 0:
        errors.append(f"daily_run exit code {proc.returncode}")

    output = (proc.stdout or "") + (proc.stderr or "")
    if "TRACEBACK" in output.upper() or "ERROR" in output.upper():
        errors.append("stdout/stderr contains ERROR/Traceback")

    if _has_safe_mode_indicator(proc):
        errors.append("SAFE_MODE indicator found in output")

    decisions_count = _load_storage_rows(PROJECT_ROOT / "storage" / "decisions", "day2")
    audits_count = _load_storage_rows(PROJECT_ROOT / "storage" / "audits", "day2")
    stats_count = _load_storage_rows(PROJECT_ROOT / "storage" / "stats", "day2")

    if decisions_count < 1:
        errors.append("decisions not found for run_id=day2")
    if audits_count < 1:
        errors.append("audits not found for run_id=day2 (expect >=1 row)")
    if stats_count < 1:
        errors.append("stats not found for run_id=day2")

    if errors:
        print("FAIL: PAPER Day 2 validation failed")
        for e in errors:
            print(f" - {e}")
        sys.exit(1)

    print("PASS: PAPER Day 2 completed")
    print(f"decisions={decisions_count} audits={audits_count} stats={stats_count}")
    sys.exit(0)


if __name__ == "__main__":
    main()
