"""
ForexFactory calendar scraper (deterministic, no browser).

Fetches the public XML feed and persists raw rows into raw_events.db with
deduplication by event_id. Engine must never read this DB directly.
"""

from __future__ import annotations

import re
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Dict, List, Tuple

import requests

# Primary feed and fallbacks (ForexFactory/CDN variants). Order matters: XML first to capture detail links.
FEED_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.xml",  # XML with detail links
    "http://nfs.faireconomy.media/ff_calendar_thisweek.xml",  # XML http fallback
    "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.xml",  # legacy CDN
    "http://cdn-nfs.faireconomy.media/ff_calendar_thisweek.xml",  # legacy CDN http
    "https://cdn-nfs.forexfactory.net/ff_calendar_thisweek.xml",  # legacy
    "http://cdn-nfs.forexfactory.net/ff_calendar_thisweek.xml",  # legacy http
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",  # JSON fallback
]
BASE_DIR = Path(__file__).resolve().parent
RAW_DB = BASE_DIR / "raw_events.db"


def _maybe_add_full_text_column(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE raw_events ADD COLUMN full_text TEXT")
        conn.commit()
    except Exception:
        # Column already exists or migration not needed.
        pass


def _connect(db_path: Path = RAW_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            event_name TEXT,
            currency TEXT,
            impact TEXT,
            date_text TEXT,
            time_text TEXT,
            forecast TEXT,
            previous TEXT,
            detail_link TEXT,
            fetched_at TEXT,
            full_text TEXT
        );
        """
    )
    _maybe_add_full_text_column(conn)
    conn.commit()
    return conn


def _make_event_id(
    title: str, currency: str, date_text: str, time_text: str, detail_link: str
) -> str:
    if detail_link:
        return detail_link.strip()
    key = f"{date_text}-{time_text}-{currency}-{title}"
    return key.strip()


def _fetch_first_available(urls: List[str]) -> tuple[str, bytes]:
    last_err: Exception | None = None
    for url in urls:
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            return url, resp.content
        except Exception as exc:  # log and continue to next URL
            last_err = exc
            continue
    raise last_err if last_err else RuntimeError("No feed URLs attempted")


def _strip_html(body: str) -> str:
    # Remove scripts/styles and strip tags for a clean text body.
    body = re.sub(r"<script[\s\S]*?</script>", " ", body, flags=re.IGNORECASE)
    body = re.sub(r"<style[\s\S]*?</style>", " ", body, flags=re.IGNORECASE)
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"\s+", " ", body)
    return unescape(body).strip()


def _fetch_full_text(detail_link: str, timeout: float = 3.0) -> Tuple[str, bool]:
    if not detail_link:
        return "", False
    candidates = [detail_link]
    if detail_link.startswith("http"):
        candidates.append(f"https://r.jina.ai/{detail_link}")
    for url in candidates:
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"},
                verify=False,
            )
            resp.raise_for_status()
            return _strip_html(resp.text), True
        except Exception:
            continue
    return "", False


def _fallback_text(evt: Dict) -> str:
    parts = [evt.get("title", "").strip()]
    impact = evt.get("impact", "").strip()
    if impact:
        parts.append(f"Impact: {impact}.")
    forecast = evt.get("forecast", "").strip()
    previous = evt.get("previous", "").strip()
    actual = evt.get("actual", "").strip()
    if forecast:
        parts.append(f"Forecast: {forecast}.")
    if previous:
        parts.append(f"Previous: {previous}.")
    if actual:
        parts.append(f"Actual: {actual}.")
    return " ".join([p for p in parts if p]).strip()


def fetch_raw_events(urls: List[str] = FEED_URLS) -> List[Dict]:
    url, payload = _fetch_first_available(urls)
    events: List[Dict] = []

    if url.endswith(".json"):
        import json as _json

        rows = _json.loads(payload)
        for item in rows:
            events.append(
                {
                    "title": (item.get("title") or "").strip(),
                    "currency": (item.get("country") or "").strip(),
                    "impact": (item.get("impact") or "").strip(),
                    "forecast": (item.get("forecast") or "").strip(),
                    "previous": (item.get("previous") or "").strip(),
                    "actual": (item.get("actual") or "").strip(),
                    # JSON feed already uses ISO-ish date; keep as-is
                    "date": (item.get("date") or "").strip(),
                    # JSON feed uses datetime; keep time empty to avoid double parsing
                    "time": "",
                    "detail_link": (item.get("url") or "").strip(),
                    "full_text": "",
                }
            )
    else:
        root = ET.fromstring(payload)
        for item in root.findall("event"):
            events.append(
                {
                    "title": (item.findtext("title") or "").strip(),
                    # feed exposes <country> for the currency code
                    "currency": (item.findtext("country") or "").strip(),
                    "impact": (item.findtext("impact") or "").strip(),
                    "forecast": (item.findtext("forecast") or "").strip(),
                    "previous": (item.findtext("previous") or "").strip(),
                    "actual": (item.findtext("actual") or "").strip(),
                    "date": (item.findtext("date") or "").strip(),
                    "time": (item.findtext("time") or "").strip(),
                    # current feed uses <url> for detail link
                    "detail_link": (item.findtext("url") or "").strip(),
                    "full_text": "",
                }
            )

    return events


def collect_calendar_snapshot(
    urls: List[str] = FEED_URLS,
    skip_full_text: bool = False,
    full_text_timeout: float = 3.0,
) -> Dict:
    raw_events = fetch_raw_events(urls)
    # Enrich with full_text when detail_link is present.
    for evt in raw_events:
        if skip_full_text:
            if not evt.get("full_text"):
                evt["full_text"] = _fallback_text(evt)
            continue
        detail_link = evt.get("detail_link", "")
        full_text, ok = _fetch_full_text(detail_link, timeout=full_text_timeout)
        if ok and full_text:
            evt["full_text"] = full_text
        elif not evt.get("full_text"):
            evt["full_text"] = _fallback_text(evt)
    return {
        "source": "forexfactory",
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        "events": raw_events,
    }


def persist_raw_events(events: List[Dict], db_path: Path = RAW_DB) -> Dict:
    conn = _connect(db_path)
    inserted = 0
    skipped = 0
    sql = """
        INSERT OR IGNORE INTO raw_events
        (event_id, event_name, currency, impact, date_text, time_text, forecast, previous, detail_link, fetched_at, full_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    for evt in events:
        event_id = _make_event_id(
            evt.get("title", ""),
            evt.get("currency", ""),
            evt.get("date", ""),
            evt.get("time", ""),
            evt.get("detail_link", ""),
        )
        cur = conn.execute(
            sql,
            (
                event_id,
                evt.get("title", ""),
                evt.get("currency", ""),
                evt.get("impact", ""),
                evt.get("date", ""),
                evt.get("time", ""),
                evt.get("forecast", ""),
                evt.get("previous", ""),
                evt.get("detail_link", ""),
                now_iso,
                evt.get("full_text", ""),
            ),
        )
        if cur.rowcount:
            inserted += 1
        else:
            skipped += 1
    conn.commit()
    conn.close()
    deduped = skipped  # skipped rows are duplicates due to UNIQUE constraint
    return {
        "inserted": inserted,
        "deduped": deduped,
        "skipped": skipped,
        "db": str(db_path),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="ForexFactory calendar scraper")
    parser.add_argument(
        "--skip-full-text",
        action="store_true",
        help="Skip fetching detail page full_text (use fallback summary instead)",
    )
    parser.add_argument(
        "--full-text-timeout",
        type=float,
        default=3.0,
        help="Timeout in seconds for full_text fetch (default: 3)",
    )
    args = parser.parse_args()

    snapshot = collect_calendar_snapshot(
        skip_full_text=args.skip_full_text, full_text_timeout=args.full_text_timeout
    )
    stats = persist_raw_events(snapshot.get("events", []), RAW_DB)
    print(
        json.dumps(
            {
                "fetched": len(snapshot.get("events", [])),
                **stats,
                "skip_full_text": args.skip_full_text,
                "full_text_timeout": args.full_text_timeout,
            },
            indent=2,
        )
    )
