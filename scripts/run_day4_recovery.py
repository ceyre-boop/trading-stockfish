"""Manual recovery run for PAPER day4.

Executes daily_run for day4 and verifies storage artifacts without overwriting.
Logs to logs/scheduled/daily_day4_recovery.log and prints PASS/FAIL summary.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "scheduled"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "daily_day4_recovery.log"
RUN_ID = "day4"


def _run_daily() -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        "-m",
        "engine.jobs.daily_run",
        "--mode",
        "PAPER",
        "--run-id",
        RUN_ID,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def _write_log(proc: subprocess.CompletedProcess) -> None:
    payload = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    LOG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_count(path: Path, run_id: str) -> int:
    if not path.exists():
        return 0
    frames = []
    for f in sorted(path.glob("*.parquet")):
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


def _has_safe_mode(proc: subprocess.CompletedProcess) -> bool:
    text = (proc.stdout or "") + (proc.stderr or "")
    return "SAFE_MODE" in text.upper()


def main() -> None:
    proc = _run_daily()
    _write_log(proc)

    errors: List[str] = []
    if proc.returncode != 0:
        errors.append(f"daily_run exit code {proc.returncode}")

    if _has_safe_mode(proc):
        errors.append("SAFE_MODE indicator found in output")

    decisions = _load_count(PROJECT_ROOT / "storage" / "decisions", RUN_ID)
    audits = _load_count(PROJECT_ROOT / "storage" / "audits", RUN_ID)
    stats = _load_count(PROJECT_ROOT / "storage" / "stats", RUN_ID)

    if decisions < 1:
        errors.append("decisions missing for day4")
    if audits < 1:
        errors.append("audits missing for day4 (expect >=1)")
    if stats < 1:
        errors.append("stats missing for day4")

    if errors:
        print("FAIL: Day4 recovery")
        for e in errors:
            print(f" - {e}")
        sys.exit(1)

    print("PASS: Day4 recovery")
    print(f"decisions={decisions} audits={audits} stats={stats}")
    sys.exit(0)


if __name__ == "__main__":
    main()
