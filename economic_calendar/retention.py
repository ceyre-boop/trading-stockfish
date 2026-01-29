"""
Retention policies for the economic calendar store.

Rules:
- Keep past 7 days (configurable)
- Keep future up to 60 days (configurable)
- Purge anything older than the past window
- Purge anything beyond the future window
- Return stats for logging
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent
RAW_DB = BASE_DIR / "raw_events.db"


def _safe_connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def count_events(db_path: Path = RAW_DB) -> Dict:
    conn = _safe_connect(db_path)
    try:
        raw = conn.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0]
    except Exception:
        raw = 0
    try:
        norm = conn.execute("SELECT COUNT(*) FROM normalized_events").fetchone()[0]
    except Exception:
        norm = 0
    conn.close()
    return {"raw_events": int(raw), "normalized_events": int(norm)}


def purge_past_events(db_path: Path = RAW_DB, days: int = 7) -> Dict:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    conn = _safe_connect(db_path)
    conn.row_factory = sqlite3.Row
    past_ids = []
    try:
        cur = conn.execute("SELECT event_id, timestamp_iso FROM normalized_events")
        for row in cur.fetchall():
            ts = row["timestamp_iso"] or ""
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            if dt < cutoff:
                past_ids.append(row["event_id"])
    except Exception:
        conn.close()
        return {"purged_past": 0}

    purged_norm = purged_raw = 0
    if past_ids:
        conn.executemany(
            "DELETE FROM normalized_events WHERE event_id = ?",
            [(eid,) for eid in past_ids],
        )
        purged_norm = conn.total_changes
        conn.executemany(
            "DELETE FROM raw_events WHERE event_id = ?", [(eid,) for eid in past_ids]
        )
        purged_raw = conn.total_changes - purged_norm
    conn.commit()
    conn.close()
    return {"purged_past": purged_norm}


def purge_future_events(db_path: Path = RAW_DB, max_days_ahead: int = 60) -> Dict:
    horizon = datetime.now(tz=timezone.utc) + timedelta(days=max_days_ahead)
    conn = _safe_connect(db_path)
    conn.row_factory = sqlite3.Row
    future_ids = []
    try:
        cur = conn.execute("SELECT event_id, timestamp_iso FROM normalized_events")
        for row in cur.fetchall():
            ts = row["timestamp_iso"] or ""
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            if dt > horizon:
                future_ids.append(row["event_id"])
    except Exception:
        conn.close()
        return {"purged_future": 0}

    purged_norm = purged_raw = 0
    if future_ids:
        conn.executemany(
            "DELETE FROM normalized_events WHERE event_id = ?",
            [(eid,) for eid in future_ids],
        )
        purged_norm = conn.total_changes
        conn.executemany(
            "DELETE FROM raw_events WHERE event_id = ?", [(eid,) for eid in future_ids]
        )
        purged_raw = conn.total_changes - purged_norm
    conn.commit()
    conn.close()
    return {"purged_future": purged_norm}


def apply_retention(
    db_path: Path = RAW_DB, days: int = 7, max_days_ahead: int = 60
) -> Dict:
    past_stats = purge_past_events(db_path, days=days)
    future_stats = purge_future_events(db_path, max_days_ahead=max_days_ahead)
    counts = count_events(db_path)
    return {**past_stats, **future_stats, **counts}
