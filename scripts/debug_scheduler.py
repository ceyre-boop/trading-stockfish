"""Scheduler diagnostics.

Checks Task Scheduler daily task status (assumes task names contain 'daily').
Logs to logs/system/scheduler_debug_YYYYMMDD.log.
"""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "system"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"scheduler_debug_{dt.datetime.utcnow():%Y%m%d}.log"


def run_ps(command: str) -> subprocess.CompletedProcess:
    return subprocess.run([
        "powershell",
        "-NoProfile",
        "-Command",
        command,
    ], capture_output=True, text=True)


def collect_tasks() -> List[str]:
    # Query all tasks, filter to those with 'daily' in the path/name.
    cmd = "Get-ScheduledTask | Where-Object { $_.TaskName -like '*daily*' -or $_.TaskPath -like '*daily*' } | Select-Object TaskName, TaskPath"
    proc = run_ps(cmd)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip() and not line.startswith('TaskName')]


def query_task_details(task_name: str) -> str:
    cmd = (
        "(Get-ScheduledTask -TaskName '{0}' | Get-ScheduledTaskInfo) | "
        "Select-Object LastRunTime, LastTaskResult, NextRunTime, TaskName | Format-List"
    ).format(task_name)
    proc = run_ps(cmd)
    if proc.returncode != 0:
        return f"Error querying {task_name}: {proc.stderr}"
    return proc.stdout


def main() -> None:
    lines: List[str] = []
    tasks = collect_tasks()
    if not tasks:
        print("FAIL: No daily tasks found")
        LOG_PATH.write_text("No daily tasks found", encoding="utf-8")
        sys.exit(1)

    failures: List[str] = []
    for task in tasks:
        detail = query_task_details(task)
        lines.append(f"=== {task} ===\n{detail}\n")
        if "LastTaskResult" in detail and "0" not in detail:
            failures.append(f"Task {task} last result not 0")

    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")

    if failures:
        print("FAIL: Scheduler issues detected")
        for f in failures:
            print(f" - {f}")
        sys.exit(1)

    print("PASS: Scheduler tasks query completed (see log)")
    sys.exit(0)


if __name__ == "__main__":
    main()
