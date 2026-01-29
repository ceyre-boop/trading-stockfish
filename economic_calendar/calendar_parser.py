"""
Calendar parser: normalize raw_events.db into normalized_events table with
ISO timestamps, impact buckets, and asset relevance mapping. Engine never reads
raw rows directly.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

IMPACT_MAP = {
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
    "holiday": "LOW",
}

CURRENCY_ASSETS = {
    "USD": ["ES", "NQ", "SPX", "DXY", "BTC"],
    "EUR": ["EURUSD", "DAX"],
    "GBP": ["GBPUSD", "FTSE"],
    "JPY": ["USDJPY", "NIKKEI"],
}

BASE_DIR = Path(__file__).resolve().parent
RAW_DB = BASE_DIR / "raw_events.db"


def _maybe_add_full_text_column(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE normalized_events ADD COLUMN full_text TEXT")
        conn.commit()
    except Exception:
        # Column already exists or migration not needed.
        pass


def _normalize_timestamp(date_str: str, time_str: str, year: int) -> str:
    # Supports ISO datetime (JSON feed) and legacy "Jan 22" + "8:30am" format
    try:
        # JSON feed already provides a full datetime string with offset
        if "T" in date_str:
            dt = datetime.fromisoformat(date_str)
            return dt.astimezone(timezone.utc).isoformat()

        if time_str.lower() == "all day":
            dt = datetime.strptime(f"{date_str} {year} 12:00pm", "%b %d %Y %I:%M%p")
        else:
            dt = datetime.strptime(f"{date_str} {year} {time_str}", "%b %d %Y %I:%M%p")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        # Fallback to current day to avoid crashing pipeline
        return datetime.now(tz=timezone.utc).isoformat()


def _connect(db_path: Path = RAW_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS normalized_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            event_name TEXT,
            timestamp_iso TEXT,
            impact_level TEXT,
            currency TEXT,
            forecast TEXT,
            previous TEXT,
            detail_link TEXT,
            asset_scope TEXT,
            event_type TEXT,
            full_text TEXT
        );
        """
    )
    _maybe_add_full_text_column(conn)
    conn.commit()
    return conn


def _event_type_from_name(name: str) -> str:
    name_u = name.upper()
    for key, evt_type in [
        ("CPI", "CPI"),
        ("PCE", "CPI"),
        ("FOMC", "FOMC"),
        ("NFP", "NFP"),
        ("PAYROLL", "NFP"),
        ("PMI", "PMI"),
        ("ISM", "PMI"),
        ("GDP", "GDP"),
    ]:
        if key in name_u:
            return evt_type
    return "OTHER"


def normalize_and_store(db_path: Path = RAW_DB) -> Dict:
    conn = _connect(db_path)
    cur = conn.execute(
        """
        SELECT event_id, event_name, currency, impact, date_text, time_text, forecast, previous, detail_link, full_text
        FROM raw_events
        ORDER BY date_text, time_text
        """
    )
    rows = cur.fetchall()
    now = datetime.now(tz=timezone.utc)
    current_year = now.year
    insert_sql = """
        INSERT OR REPLACE INTO normalized_events
        (event_id, event_name, timestamp_iso, impact_level, currency, forecast, previous, detail_link, asset_scope, event_type, full_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    inserted = 0
    for (
        event_id,
        name,
        currency,
        impact_raw,
        date_text,
        time_text,
        forecast,
        previous,
        detail_link,
        full_text,
    ) in rows:
        ts_iso = _normalize_timestamp(date_text or "", time_text or "", current_year)
        impact = IMPACT_MAP.get((impact_raw or "").lower(), "MEDIUM")
        assets = CURRENCY_ASSETS.get((currency or "").upper(), [])
        event_type = _event_type_from_name(name or "")
        conn.execute(
            insert_sql,
            (
                event_id,
                name or "",
                ts_iso,
                impact,
                currency or "",
                forecast or "",
                previous or "",
                detail_link or "",
                json.dumps(assets),
                event_type,
                full_text or "",
            ),
        )
        inserted += 1
    conn.commit()
    conn.close()
    return {"normalized_count": inserted, "db": str(db_path)}


if __name__ == "__main__":
    stats = normalize_and_store(RAW_DB)
    print(json.dumps(stats, indent=2))
