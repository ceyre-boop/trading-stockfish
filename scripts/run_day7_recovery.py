"""Manual recovery run for PAPER day7.

Executes daily_run for day7 and verifies storage artifacts without overwriting existing completed data.
Logs to logs/scheduled/daily_day7_recovery.log and prints PASS/FAIL summary.
Idempotent: skips execution if all day7 artifacts already exist; fails fast if partial data is present.
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
LOG_PATH = LOG_DIR / "daily_day7_recovery.log"
RUN_ID = "day7"


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


def _write_log(payload: Dict) -> None:
    LOG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    pre_counts = {
        "decisions": _load_count(PROJECT_ROOT / "storage" / "decisions", RUN_ID),
        "audits": _load_count(PROJECT_ROOT / "storage" / "audits", RUN_ID),
        "stats": _load_count(PROJECT_ROOT / "storage" / "stats", RUN_ID),
    }

    if all(v >= 1 for v in pre_counts.values()):
        payload = {
            "status": "skipped_existing_data",
            "executed": False,
            "pre_counts": pre_counts,
        }
        _write_log(payload)
        print("SKIP: Day7 artifacts already present; no recovery run executed")
        sys.exit(0)

    if any(v > 0 for v in pre_counts.values()):
        payload = {
            "status": "partial_data_present",
            "executed": False,
            "pre_counts": pre_counts,
            "error": "partial day7 data present; manual cleanup required before recovery",
        }
        _write_log(payload)
        print("FAIL: Partial day7 data present; manual cleanup required before recovery")
        for k, v in pre_counts.items():
            print(f" - {k}: {v}")
        sys.exit(1)

    proc = _run_daily()

    post_counts = {
        "decisions": _load_count(PROJECT_ROOT / "storage" / "decisions", RUN_ID),
        "audits": _load_count(PROJECT_ROOT / "storage" / "audits", RUN_ID),
        "stats": _load_count(PROJECT_ROOT / "storage" / "stats", RUN_ID),
    }

    errors: List[str] = []
    if proc.returncode != 0:
        errors.append(f"daily_run exit code {proc.returncode}")
    if _has_safe_mode(proc):
        errors.append("SAFE_MODE indicator found in output")
    if post_counts["decisions"] < 1:
        errors.append("decisions missing for day7")
    if post_counts["audits"] < 1:
        errors.append("audits missing for day7 (expect >=1)")
    if post_counts["stats"] < 1:
        errors.append("stats missing for day7")

    payload = {
        "status": "executed",
        "executed": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "pre_counts": pre_counts,
        "post_counts": post_counts,
        "errors": errors,
    }
    _write_log(payload)

    if errors:
        print("FAIL: Day7 recovery")
        for e in errors:
            print(f" - {e}")
        sys.exit(1)

    print("PASS: Day7 recovery")
    print(f"decisions={post_counts['decisions']} audits={post_counts['audits']} stats={post_counts['stats']}")
    sys.exit(0)


if __name__ == "__main__":
    main()
