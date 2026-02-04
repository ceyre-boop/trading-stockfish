"""Pre-run checks for the Phase 10 Weekly Cycle.

Performs:
- Clock sync check (PowerShell)
- Policy validation
- Connector validation
- Storage continuity for run_id day1-day7 (decisions, audits, stats)
- SAFE_MODE scan across scheduled logs

Prints consolidated PASS/FAIL and exits non-zero on any failure.
Deterministic and non-mutating.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def check_clock() -> List[str]:
    code, out, err = run_cmd(
        [
            "powershell",
            "-File",
            str(PROJECT_ROOT / "scripts" / "check_clock_sync.ps1"),
        ]
    )
    errs: List[str] = []
    if code != 0:
        errs.append("clock check script failed")
    if "not synchronized" in (out + err).lower():
        errs.append("clock not synchronized")
    return errs


def check_policy() -> List[str]:
    code, _, _ = run_cmd(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "validate_policy_config.py")]
    )
    return [] if code == 0 else ["policy validation failed"]


def check_connectors() -> List[str]:
    code, _, _ = run_cmd(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "validate_connectors.py")]
    )
    return [] if code == 0 else ["connector validation failed"]


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


def check_storage_continuity() -> tuple[List[str], Dict[str, Dict[str, int]]]:
    errors: List[str] = []
    counts: Dict[str, Dict[str, int]] = {}
    decisions_path = PROJECT_ROOT / "storage" / "decisions"
    audits_path = PROJECT_ROOT / "storage" / "audits"
    stats_path = PROJECT_ROOT / "storage" / "stats"
    for day in range(1, 8):
        run_id = f"day{day}"
        dec = _load_count(decisions_path, run_id)
        aud = _load_count(audits_path, run_id)
        sta = _load_count(stats_path, run_id)
        counts[run_id] = {"decisions": dec, "audits": aud, "stats": sta}
        if dec < 1:
            errors.append(f"decisions missing for {run_id}")
        if aud < 1:
            errors.append(f"audits missing for {run_id} (expect >=1)")
        if sta < 1:
            errors.append(f"stats missing for {run_id}")
    return errors, counts


def check_safe_mode() -> List[str]:
    log_dir = PROJECT_ROOT / "logs" / "scheduled"
    if not log_dir.exists():
        return []
    hits: List[str] = []
    for f in sorted(log_dir.glob("*.log")):
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if "SAFE_MODE" in text.upper():
            hits.append(f.name)
    return [] if not hits else [f"SAFE_MODE detected in logs: {', '.join(hits)}"]


def main() -> None:
    failures: List[str] = []
    storage_counts: Dict[str, Dict[str, int]] = {}

    failures += check_clock()
    failures += check_policy()
    failures += check_connectors()
    storage_errors, storage_counts = check_storage_continuity()
    failures += storage_errors
    failures += check_safe_mode()

    if failures:
        print("FAIL: Weekly cycle pre-checks")
        for f in failures:
            print(f" - {f}")
        if storage_counts:
            print("Storage counts (day1-day7):")
            for run_id, counts in storage_counts.items():
                print(
                    f"  {run_id}: decisions={counts['decisions']} audits={counts['audits']} stats={counts['stats']}"
                )
        sys.exit(1)

    print(
        "PASS: Weekly cycle pre-checks (clock, policy, connectors, storage continuity, SAFE_MODE clear)"
    )
    for run_id, counts in storage_counts.items():
        print(
            f"  {run_id}: decisions={counts['decisions']} audits={counts['audits']} stats={counts['stats']}"
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
