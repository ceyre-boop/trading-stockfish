"""
Event scheduler: scans calendar_data/events.json, determines which events are due,
fetches raw text, sends to LLM parser, and writes structured outputs into
news_events/<impact>/ folders.

This module is deterministic and can be invoked periodically (cron/task) rather
than running a long-lived loop.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from economic_calendar.llm_parser import parse_event
from economic_calendar.news_fetcher import fetch_event_text
from economic_calendar.router import route_event

BASE_DIR = Path(__file__).resolve().parent.parent
EVENTS_PATH = BASE_DIR / "economic_calendar" / "calendar_data" / "events.json"
NEWS_BASE = BASE_DIR / "news_events"


def _load_events(path: Path = EVENTS_PATH) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("events", [])


def _impact_folder(impact: str) -> Path:
    impact_lower = (impact or "medium").lower()
    if impact_lower not in {"low", "medium", "high"}:
        impact_lower = "medium"
    return NEWS_BASE / impact_lower


def _slugify(text: str) -> str:
    return (
        "".join(ch.lower() if ch.isalnum() else "-" for ch in text)[:80].strip("-")
        or "event"
    )


def _is_due(ts_iso: str, tolerance_seconds: int = 5) -> bool:
    try:
        ts = datetime.fromisoformat(ts_iso)
    except Exception:
        return False
    now = datetime.now(tz=timezone.utc)
    return ts <= now and (now - ts).total_seconds() <= tolerance_seconds


def process_due_events() -> List[Dict]:
    events = _load_events()
    processed: List[Dict] = []
    for evt in events:
        if not _is_due(evt.get("timestamp", "")):
            continue
        raw_text = fetch_event_text(evt)
        parsed = parse_event(raw_text, evt)
        route_event(parsed)
        processed.append(parsed)
    return processed


def sleep_until_next_event() -> float:
    events = _load_events()
    now = datetime.now(tz=timezone.utc)
    future_events = []
    for evt in events:
        try:
            ts = datetime.fromisoformat(evt.get("timestamp", ""))
            delta = (ts - now).total_seconds()
            if delta > 0:
                future_events.append(delta)
        except Exception:
            continue
    if not future_events:
        return 0.0
    wait_seconds = min(future_events)
    time.sleep(wait_seconds)
    return wait_seconds


if __name__ == "__main__":
    processed = process_due_events()
    print(json.dumps({"processed": len(processed)}, indent=2))
