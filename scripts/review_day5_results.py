"""Post-run review for PAPER Day 5.

Checks:
- storage/decisions has run_id=day5 rows
- storage/audits has run_id=day5 rows (>=1)
- storage/stats has run_id=day5 rows
- scheduler log for day5 present
- SAFE_MODE not indicated in scheduler log

Writes summary to logs/scheduled/daily_day5_review.log and exits non-zero on failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "scheduled"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "daily_day5_review.log"
RUN_ID = "day5"


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


def _find_scheduler_log() -> Path | None:
    if not LOG_DIR.exists():
        return None
    for f in sorted(LOG_DIR.glob("daily*.log"), reverse=True):
        try:
            content = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if RUN_ID in content:
            return f
    return None


def _safe_mode_indicator(text: str) -> bool:
    return "SAFE_MODE" in text.upper()


def main() -> None:
    report: dict = {}
    errors: List[str] = []

    decisions_count = _load_count(PROJECT_ROOT / "storage" / "decisions", RUN_ID)
    audits_count = _load_count(PROJECT_ROOT / "storage" / "audits", RUN_ID)
    stats_count = _load_count(PROJECT_ROOT / "storage" / "stats", RUN_ID)

    report["decisions_count"] = decisions_count
    report["audits_count"] = audits_count
    report["stats_count"] = stats_count

    if decisions_count < 1:
        errors.append("decisions missing for day5")
    if audits_count < 1:
        errors.append("audits missing for day5 (expect >=1 row)")
    if stats_count < 1:
        errors.append("stats missing for day5")

    sched_log = _find_scheduler_log()
    report["scheduler_log"] = str(sched_log) if sched_log else None
    if not sched_log:
        errors.append("scheduler log for day5 not found")
    else:
        text = sched_log.read_text(encoding="utf-8")
        if _safe_mode_indicator(text):
            errors.append("SAFE_MODE indicator found in scheduler log")

    payload = {"report": report, "errors": errors}
    LOG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if errors:
        print("FAIL: Day 5 review")
        for e in errors:
            print(f" - {e}")
        sys.exit(1)

    print("PASS: Day 5 review")
    print(json.dumps(report, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
