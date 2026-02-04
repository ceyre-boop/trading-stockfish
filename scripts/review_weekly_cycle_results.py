"""Weekly cycle results validation for Phase 10.

Validates:
- Weekly report exists (prefers storage/reports, falls back to reports/)
- Report contains regime performance, feature drift, and policy version sections
- No unintended policy promotions detected (scans report text for 'promotion')
- SAFE_MODE absent in scheduled logs and report text
- Storage continuity intact for run_id day1-day7 across decisions/audits/stats

Writes JSON summary to logs/scheduled/weekly_cycle_review.log and prints PASS/FAIL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "scheduled"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "weekly_cycle_review.log"
RUN_IDS = [f"day{i}" for i in range(1, 8)]


def _find_reports_dir() -> Optional[Path]:
    for candidate in [PROJECT_ROOT / "storage" / "reports", PROJECT_ROOT / "reports"]:
        if candidate.exists():
            return candidate
    return None


def _latest_report(path: Path) -> Optional[Path]:
    files = sorted(
        [p for p in path.glob("**/*") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


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


def _check_storage_continuity() -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    errors: List[str] = []
    counts: Dict[str, Dict[str, int]] = {}
    decisions = PROJECT_ROOT / "storage" / "decisions"
    audits = PROJECT_ROOT / "storage" / "audits"
    stats = PROJECT_ROOT / "storage" / "stats"
    for run_id in RUN_IDS:
        dec = _load_count(decisions, run_id)
        aud = _load_count(audits, run_id)
        sta = _load_count(stats, run_id)
        counts[run_id] = {"decisions": dec, "audits": aud, "stats": sta}
        if dec < 1:
            errors.append(f"decisions missing for {run_id}")
        if aud < 1:
            errors.append(f"audits missing for {run_id} (expect >=1)")
        if sta < 1:
            errors.append(f"stats missing for {run_id}")
    return errors, counts


def _check_safe_mode(report_text: str) -> List[str]:
    hits: List[str] = []
    sched_dir = PROJECT_ROOT / "logs" / "scheduled"
    if sched_dir.exists():
        for f in sorted(sched_dir.glob("*.log")):
            try:
                text = f.read_text(encoding="utf-8")
            except Exception:
                continue
            if "SAFE_MODE" in text.upper():
                hits.append(f.name)
    if "SAFE_MODE" in report_text.upper():
        hits.append("weekly_report_text")
    return [] if not hits else ["SAFE_MODE detected in: " + ", ".join(hits)]


def _check_promotions(report_text: str) -> List[str]:
    text = report_text.lower()
    if "promotion" in text or "promoted" in text:
        return ["policy promotion indicator found in report text"]
    return []


def _check_report_sections(report_text: str) -> List[str]:
    text = report_text.lower()
    missing: List[str] = []
    required = {
        "regime performance": "regime performance section missing",
        "feature drift": "feature drift section missing",
        "policy version": "policy version comparison section missing",
    }
    for key, msg in required.items():
        if key not in text:
            missing.append(msg)
    return missing


def main() -> None:
    errors: List[str] = []
    report_info: Dict[str, str | None] = {"path": None}

    reports_dir = _find_reports_dir()
    if not reports_dir:
        errors.append("reports directory not found")
        report_text = ""
    else:
        report_path = _latest_report(reports_dir)
        if not report_path:
            errors.append("no weekly report files found")
            report_text = ""
        else:
            report_info["path"] = str(report_path)
            try:
                report_text = report_path.read_text(encoding="utf-8")
            except Exception:
                report_text = ""
                errors.append("failed to read weekly report text")

    # Section and promotion checks
    if report_text:
        errors += _check_report_sections(report_text)
        errors += _check_promotions(report_text)
    else:
        errors.append("weekly report content unavailable")

    # SAFE_MODE checks
    errors += _check_safe_mode(report_text)

    # Storage continuity
    storage_errors, storage_counts = _check_storage_continuity()
    errors += storage_errors

    payload = {
        "report": report_info,
        "storage_counts": storage_counts,
        "errors": errors,
    }
    LOG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if errors:
        print("FAIL: Weekly cycle review")
        for e in errors:
            print(f" - {e}")
        sys.exit(1)

    print("PASS: Weekly cycle review")
    print(json.dumps(payload, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
