"""
Router for structured news payloads.

Currently writes JSON snapshots into news_events/<impact>/ for ingestion by the
engine. The write is synchronous but lightweight; callers can wrap with their
own async executor to avoid blocking.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent.parent
NEWS_BASE = BASE_DIR / "news_events"
MAX_RETRIES = 3


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


def route_event(event: Dict) -> Path:
    impact_folder = _impact_folder(event.get("impact_level", "medium"))
    impact_folder.mkdir(parents=True, exist_ok=True)
    ts = event.get("timestamp") or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    slug = _slugify(event.get("event_type", "event"))
    target = impact_folder / f"{ts}_{slug}.json"

    payload = {
        "routed_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "event": event,
    }

    for attempt in range(MAX_RETRIES):
        try:
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return target
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
    return target
