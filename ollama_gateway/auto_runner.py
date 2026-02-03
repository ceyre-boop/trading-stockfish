"""
End-to-end pull → parse → filter → Ollama → validate → route loop for ForexFactory.
Raw RSS/Twitter are not used here. Everything flows through the gateway snapshot
before reaching Stockfish-Trade.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from economic_calendar.calendar_parser import normalize_and_store
from economic_calendar.calendar_scraper import (
    collect_calendar_snapshot,
    persist_raw_events,
)
from ollama_gateway import router

LOGGER = logging.getLogger("ollama_gateway.auto_runner")
if not LOGGER.hasHandlers():
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(handler)


def _refresh_calendar(db_path: Path) -> None:
    snapshot = collect_calendar_snapshot()
    stats = persist_raw_events(snapshot.get("events", []), db_path)
    norm_stats = normalize_and_store(db_path)
    LOGGER.info(
        "Calendar refreshed raw=%s normalized=%s",
        stats.get("inserted", 0),
        norm_stats.get("normalized", 0),
    )


def run_once(outbox_post: bool = False) -> None:
    base_dir = Path(__file__).resolve().parent.parent
    raw_db = base_dir / "economic_calendar" / "raw_events.db"

    _refresh_calendar(raw_db)
    # asset_filter is called inside aggregator/router via config assets
    router.run(write=True, post=outbox_post)
    LOGGER.info("Gateway snapshot updated and routed")


def run_loop(interval: int, outbox_post: bool = False) -> None:
    while True:
        run_once(outbox_post=outbox_post)
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Ollama Gateway runner")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument(
        "--interval", type=int, default=180, help="Seconds between runs when looping"
    )
    parser.add_argument(
        "--post", action="store_true", help="POST snapshot to engine endpoint"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Override OLLAMA_MAX_ITEMS for this run (send only N due events)",
    )
    args = parser.parse_args()

    # Optional per-run cap for fast tests.
    if args.max_items is not None:
        import os

        os.environ["OLLAMA_MAX_ITEMS"] = str(args.max_items)

    if args.loop:
        run_loop(interval=args.interval, outbox_post=args.post)
    else:
        run_once(outbox_post=args.post)
