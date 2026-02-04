"""Pre-run checklist for PAPER Day 5.

Runs:
- Clock sync check (PowerShell)
- Policy validation
- Connector validation
- Scheduler debug (non-mutating)

Prints consolidated PASS/FAIL; exits non-zero on any failure.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd, cwd=None):
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return proc.returncode, proc.stdout, proc.stderr


def check_clock() -> list[str]:
    code, out, err = run_cmd([
        "powershell",
        "-File",
        str(PROJECT_ROOT / "scripts" / "check_clock_sync.ps1"),
    ])
    errors = []
    if code != 0:
        errors.append("clock check script failed")
    if "not synchronized" in (out + err).lower():
        errors.append("clock not synchronized")
    return errors


def check_policy() -> list[str]:
    code, out, err = run_cmd([sys.executable, str(PROJECT_ROOT / "scripts" / "validate_policy_config.py")])
    return [] if code == 0 else ["policy validation failed"]


def check_connectors() -> list[str]:
    code, out, err = run_cmd([sys.executable, str(PROJECT_ROOT / "scripts" / "validate_connectors.py")])
    return [] if code == 0 else ["connector validation failed"]


def check_scheduler() -> list[str]:
    code, out, err = run_cmd([sys.executable, str(PROJECT_ROOT / "scripts" / "debug_scheduler.py")])
    if code != 0:
        return ["scheduler debug failed"]
    return []


def main() -> None:
    failures: list[str] = []
    failures += check_clock()
    failures += check_policy()
    failures += check_connectors()
    failures += check_scheduler()

    if failures:
        print("FAIL: Day 5 pre-run checks")
        for f in failures:
            print(f" - {f}")
        sys.exit(1)

    print("PASS: Day 5 pre-run checks (clock, policy, connectors, scheduler)")
    sys.exit(0)


if __name__ == "__main__":
    main()
