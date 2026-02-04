"""Pre-run checklist for PAPER Day 3.

Runs three checks:
- Clock sync via scripts/check_clock_sync.ps1
- Policy config validation via scripts/validate_policy_config.py
- Connector validation via scripts/validate_connectors.py

Prints PASS/FAIL summary and exits non-zero on any failure.
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
    code, out, err = run_cmd(["powershell", "-File", str(PROJECT_ROOT / "scripts" / "check_clock_sync.ps1")])
    errors = []
    if code != 0:
        errors.append("clock check script failed")
    if "not synchronized" in (out + err).lower():
        errors.append("clock not synchronized")
    return errors


def check_policy() -> list[str]:
    code, out, err = run_cmd([sys.executable, str(PROJECT_ROOT / "scripts" / "validate_policy_config.py")])
    errors = []
    if code != 0:
        errors.append("policy validation failed")
    return errors


def check_connectors() -> list[str]:
    code, out, err = run_cmd([sys.executable, str(PROJECT_ROOT / "scripts" / "validate_connectors.py")])
    errors = []
    if code != 0:
        errors.append("connector validation failed")
    return errors


def main() -> None:
    failures: list[str] = []

    failures += check_clock()
    failures += check_policy()
    failures += check_connectors()

    if failures:
        print("FAIL: Day 3 pre-run checks")
        for f in failures:
            print(f" - {f}")
        sys.exit(1)

    print("PASS: Day 3 pre-run checks (clock, policy, connectors)")
    sys.exit(0)


if __name__ == "__main__":
    main()
