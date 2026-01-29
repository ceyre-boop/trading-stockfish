"""
Ollama Gateway aggregator.
Reads filtered ForexFactory events (already normalized and asset-filtered),
parses them via Ollama, validates, and returns a single engine-safe snapshot.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from economic_calendar.future_events import get_future_events
from ollama_gateway import config, ollama_client, validator

LOGGER = logging.getLogger("ollama_gateway.aggregator")
if not LOGGER.hasHandlers():
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(handler)


def _ensure_processed_table(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_events (
            event_id TEXT PRIMARY KEY,
            processed_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _fetch_processed_ids(db_path: Path) -> set[str]:
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT event_id FROM processed_events").fetchall()
        conn.close()
        return {r[0] for r in rows}
    except Exception:
        return set()


def _mark_processed(db_path: Path, event_ids: List[str]) -> None:
    if not event_ids:
        return
    conn = sqlite3.connect(db_path)
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    conn.executemany(
        "INSERT OR REPLACE INTO processed_events (event_id, processed_at) VALUES (?, ?)",
        [(eid, now_iso) for eid in event_ids],
    )
    conn.commit()
    conn.close()


def collect_due_items(assets: List[str]) -> List[Dict[str, Any]]:
    _ensure_processed_table(Path(config.RAW_EVENTS_DB))
    processed = _fetch_processed_ids(Path(config.RAW_EVENTS_DB))
    now = datetime.now(tz=timezone.utc)
    # grab events in horizon and select those due now or in past 30m
    future_events = get_future_events(
        assets, db_path=Path(config.RAW_EVENTS_DB), horizon_days=1
    )
    due_events = [e for e in future_events if _parse_time(e.get("timestamp")) <= now]
    items: List[Dict[str, Any]] = []
    for evt in due_events:
        if evt.get("origin_id") in processed:
            continue
        summary = f"{evt.get('event_name','')}. Impact: {evt.get('impact_level','')}. Event type: {evt.get('event_type','OTHER')}.".strip()
        detail_link = evt.get("detail_link", "")
        full_text = (evt.get("full_text", "") or "").strip()
        raw_text = full_text if full_text else summary
        items.append(
            {
                "raw_text": raw_text,
                "metadata": {
                    "source": "forex",
                    "origin_id": evt.get("origin_id", ""),
                    "timestamp": evt.get("timestamp", ""),
                    "asset_scope": evt.get("asset_scope", []),
                    "event_type": evt.get("event_type", "OTHER"),
                    "impact_level": str(evt.get("impact_level", "HIGH")).upper(),
                    "directional_bias": "NEUTRAL",
                    "keywords": [evt.get("event_name", "")],
                    "numeric_extractions": {},
                    "detail_link": detail_link,
                    "full_text_included": bool(full_text),
                },
            }
        )
    return items


def _parse_time(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(tz=timezone.utc)


def parse_and_validate(raw_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    parsed: List[Dict[str, Any]] = []
    sent_to_ollama = len(raw_items)
    for item in raw_items:
        raw_text = item.get("raw_text", "")
        metadata = item.get("metadata", {})
        try:
            model_record = ollama_client.parse_with_ollama(raw_text, metadata)
        except Exception as exc:  # pragma: no cover - network boundary
            LOGGER.error(
                "Ollama parse failed origin_id=%s error=%s",
                metadata.get("origin_id", ""),
                exc,
            )
            continue
        impact_val = model_record.get("impact") or model_record.get("impact_level") or metadata.get("impact_level")
        combined = {
            "source": metadata.get("source", "forex"),
            "origin_id": metadata.get("origin_id", "unknown"),
            "timestamp": metadata.get("timestamp", ""),
            "asset_scope": metadata.get("asset_scope", []),
            "event_type": metadata.get("event_type", "OTHER"),
            "impact_level": str(impact_val or "MEDIUM").upper(),
            "directional_bias": str(model_record.get("directional_bias", "NEUTRAL")).upper(),
            "confidence": model_record.get("confidence", 0.0),
            "sentiment_score": 0.0,
            "sentiment_volatility": 0.0,
            "summary": model_record.get("summary", ""),
            "keywords": model_record.get("keywords", []),
            "numeric_extractions": model_record.get("numeric_extractions", {}),
        }
        clean = validator.validate_record(combined)
        if clean:
            parsed.append(clean)
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "parsed_events": parsed,
        "record_count": len(parsed),
        "aggregated_scores": validator.aggregate_features(parsed),
        "sent_to_ollama": sent_to_ollama,
        "ollama_unreachable": False,
    }


def run(write: bool = False, assets: List[str] | None = None) -> Dict[str, Any]:
    health_ok, health_error = ollama_client.check_ollama_health()
    if not health_ok:
        LOGGER.error("Ollama unreachable; skipping parse: %s", health_error)
        return {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "parsed_events": [],
            "record_count": 0,
            "aggregated_scores": validator.aggregate_features([]),
            "sent_to_ollama": 0,
            "ollama_unreachable": True,
        }

    raw_items = collect_due_items(assets or config.TRADED_ASSETS)
    LOGGER.info(
        "Sending %s due events to Ollama host=%s", len(raw_items), config.OLLAMA_HOST
    )
    snapshot = parse_and_validate(raw_items)
    processed_ids = [
        rec.get("origin_id", "") for rec in snapshot.get("parsed_events", [])
    ]
    _mark_processed(
        Path(config.RAW_EVENTS_DB),
        [
            pid
            for pid in processed_ids
            if pid and not snapshot.get("ollama_unreachable", False)
        ],
    )
    LOGGER.info(
        "Gateway sent_to_ollama=%s parsed_events=%s",
        snapshot.get("sent_to_ollama", 0),
        snapshot.get("record_count", 0),
    )
    if write:
        from ollama_gateway import router

        router.write_snapshot(snapshot)
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ollama Gateway end-to-end")
    parser.add_argument("--write", action="store_true", help="Write snapshot to outbox")
    parser.add_argument("--assets", nargs="*", default=config.TRADED_ASSETS)
    args = parser.parse_args()
    result = run(write=args.write, assets=args.assets)
    print(json.dumps(result, indent=2))
