"""
Calendar maintenance entrypoint.

Runs scrape -> normalize -> retention with logging. Engine remains isolated and
ingests only the downstream snapshot; this script only maintains the calendar
SQLite store.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict

from calendar_logger import (
    log_error,
    log_parse_stats,
    log_retention_stats,
    log_run_end,
    log_run_start,
    log_scrape_stats,
)
from calendar_parser import normalize_and_store
from calendar_scraper import RAW_DB, collect_calendar_snapshot, persist_raw_events
from retention import apply_retention


def run_cycle(retention_days: int, max_future_days: int, log_level: str) -> Dict:
    # 1) scrape
    snapshot = collect_calendar_snapshot()
    fetched = len(snapshot.get("events", []))
    persist_stats = persist_raw_events(snapshot.get("events", []), RAW_DB)
    log_scrape_stats(
        fetched=fetched,
        inserted=persist_stats.get("inserted", 0),
        deduped=persist_stats.get("deduped", 0),
        level=log_level,
    )

    # 2) parse/normalize
    normalize_stats = normalize_and_store(RAW_DB)
    log_parse_stats(
        normalized=normalize_stats.get("normalized_count", 0), level=log_level
    )

    # 3) retention (past + future)
    retention_stats = apply_retention(
        db_path=RAW_DB, days=retention_days, max_days_ahead=max_future_days
    )
    log_retention_stats(
        purged_past=retention_stats.get("purged_past", 0),
        purged_future=retention_stats.get("purged_future", 0),
        total_remaining=retention_stats.get("normalized_events", 0),
        level=log_level,
    )

    return {
        "fetched": fetched,
        **persist_stats,
        **normalize_stats,
        **retention_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calendar maintenance runner")
    parser.add_argument(
        "--retention-days", type=int, default=7, help="Days to keep past events"
    )
    parser.add_argument(
        "--max-future-days", type=int, default=60, help="Max days ahead to retain"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
    )
    args = parser.parse_args()

    log_run_start(level=args.log_level)
    try:
        stats = run_cycle(args.retention_days, args.max_future_days, args.log_level)
        print(json.dumps(stats, indent=2))
    except Exception as exc:
        log_error(f"run_failed: {exc}")
        print(json.dumps({"error": str(exc)}, indent=2))
        log_run_end(level=args.log_level)
        sys.exit(1)
    log_run_end(level=args.log_level)


if __name__ == "__main__":
    main()
