"""
Historical backfill runner for Ollama Gateway.
Fetches normalized economic events in a date range and parses each full_text
through Ollama, writing NDJSON lines for offline analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

from ollama_gateway import config, ollama_client, validator

LOGGER = logging.getLogger("ollama_gateway.historical_runner")
if not LOGGER.hasHandlers():
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(handler)


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def _available_range(db_path: Path) -> Tuple[str | None, str | None]:
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT MIN(timestamp_iso), MAX(timestamp_iso) FROM normalized_events"
        ).fetchone()
        conn.close()
        return row[0], row[1]
    except Exception:
        return None, None


def _load_events_between(
    start_dt: datetime, end_dt: datetime, db_path: Path, max_items: int | None = None
) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT n.event_id, n.event_name, n.timestamp_iso, n.impact_level, n.currency,
               n.asset_scope, n.event_type, n.detail_link, COALESCE(n.full_text, r.full_text)
        FROM normalized_events n
        LEFT JOIN raw_events r ON n.event_id = r.event_id
        ORDER BY n.timestamp_iso ASC
        """
    ).fetchall()
    conn.close()

    events: List[Dict[str, Any]] = []
    for row in rows:
        ts = _parse_ts(row[2])
        if not ts:
            continue
        if ts < start_dt or ts >= end_dt:
            continue
        try:
            asset_scope = json.loads(row[5]) if row[5] else []
        except Exception:
            asset_scope = []
        events.append(
            {
                "event_id": row[0],
                "event_name": row[1],
                "timestamp": row[2],
                "timestamp_dt": ts,
                "impact_level": str(row[3] or "MEDIUM").upper(),
                "currency": row[4] or "",
                "asset_scope": asset_scope,
                "event_type": row[6] or "OTHER",
                "detail_link": row[7] or "",
                "full_text": row[8] or "",
            }
        )
    if max_items:
        events = events[:max_items]
    return events


def _normalize_conf(val: Any) -> float:
    try:
        v = float(val)
    except Exception:
        return 0.0
    return v / 100.0 if v > 1.0 else v


def _build_metadata(
    event: Dict[str, Any], raw_text: str, truncated: bool
) -> Dict[str, Any]:
    return {
        "source": "forex",
        "origin_id": event["event_id"],
        "timestamp": event["timestamp"],
        "asset_scope": event.get("asset_scope", []),
        "event_type": event.get("event_type", "OTHER"),
        "impact_level": event.get("impact_level", "MEDIUM"),
        "directional_bias": "NEUTRAL",
        "keywords": [event.get("event_name", "")],
        "numeric_extractions": {},
        "detail_link": event.get("detail_link", ""),
        "full_text_included": bool(raw_text),
        "truncated": truncated,
        "currency": event.get("currency", ""),
    }


def _write_ndjson_line(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def check_connectivity(
    url: str = "http://localhost:11434/api/tags", timeout: float = 3.0
) -> bool:
    """Check if Ollama endpoint is reachable within timeout seconds."""

    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            LOGGER.info("Ollama reachable at %s (status=%s)", url, resp.status_code)
            return True
        LOGGER.warning("Ollama reachable but returned status %s", resp.status_code)
        return True
    except requests.RequestException as exc:
        LOGGER.error("Ollama connectivity check failed: %s", exc)
        return False


def run(
    start_date: str,
    end_date: str,
    output_path: Path | None = None,
    max_items: int | None = None,
    timeout: float | None = None,
    connectivity_check_only: bool = False,
) -> Dict[str, Any]:
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date) + timedelta(days=1)
    out_path = output_path or (config.OUTBOX_DIR / "parsed_history.ndjson")

    if timeout:
        cli_timeout = float(timeout)
        if getattr(config, "OLLAMA_TIMEOUT", cli_timeout) < cli_timeout:
            config.OLLAMA_TIMEOUT = cli_timeout

    health_ok = check_connectivity()
    if connectivity_check_only:
        return {
            "connectivity_ok": bool(health_ok),
            "start_date": start_date,
            "end_date": end_date,
            "events_loaded": 0,
            "processed": 0,
            "written": 0,
            "failures": 0,
            "output": str(out_path),
        }
    if not health_ok:
        raise RuntimeError("Ollama unreachable on connectivity check")

    health_ok, health_err = ollama_client.check_ollama_health()
    if not health_ok:
        raise RuntimeError(f"Ollama unreachable: {health_err}")

    db_path = Path(config.RAW_EVENTS_DB)
    events = _load_events_between(start_dt, end_dt, db_path, max_items=max_items)
    LOGGER.info(
        "Loaded %s normalized events between %s and %s",
        len(events),
        start_date,
        end_date,
    )

    if not events:
        min_ts, max_ts = _available_range(db_path)
        LOGGER.warning(
            "No parsed events found between %s and %s. Available range: %s -> %s.",
            start_date,
            end_date,
            min_ts or "none",
            max_ts or "none",
        )
        return {
            "start_date": start_date,
            "end_date": end_date,
            "events_loaded": 0,
            "processed": 0,
            "written": 0,
            "failures": 0,
            "available_min_ts": min_ts,
            "available_max_ts": max_ts,
            "output": str(out_path),
        }

    processed = 0
    written = 0
    failures = 0

    for evt in events:
        raw_text = evt.get("full_text") or evt.get("event_name") or ""
        truncated = False
        if config.OLLAMA_MAX_CHARS and len(raw_text) > config.OLLAMA_MAX_CHARS:
            raw_text = raw_text[: config.OLLAMA_MAX_CHARS]
            truncated = True
        if not raw_text:
            LOGGER.warning("Skipping event_id=%s due to empty text", evt["event_id"])
            failures += 1
            continue

        metadata = _build_metadata(evt, raw_text, truncated)
        try:
            parsed = ollama_client.parse_with_ollama(raw_text, metadata)
        except Exception as exc:  # pragma: no cover - network boundary
            LOGGER.error("Parse failed event_id=%s error=%s", evt["event_id"], exc)
            failures += 1
            continue

        combined = {
            "source": "forex",
            "origin_id": evt["event_id"],
            "timestamp": evt["timestamp"],
            "asset_scope": evt.get("asset_scope", []),
            "event_type": evt.get("event_type", "OTHER"),
            "impact_level": parsed.get("impact")
            or parsed.get("impact_level")
            or evt.get("impact_level", "MEDIUM"),
            "directional_bias": parsed.get("directional_bias", "NEUTRAL"),
            "confidence": _normalize_conf(parsed.get("confidence", 0.0)),
            "sentiment_score": 0.0,
            "sentiment_volatility": 0.0,
            "summary": parsed.get("summary", ""),
            "keywords": parsed.get("keywords", []),
            "numeric_extractions": parsed.get("numeric_extractions", {}),
        }

        clean = validator.validate_record(combined)
        if not clean:
            LOGGER.warning("Validation failed event_id=%s", evt["event_id"])
            failures += 1
            continue

        out_record = {
            "event_id": evt["event_id"],
            "timestamp": evt["timestamp"],
            "currency": evt.get("currency", ""),
            "impact": evt.get("impact_level", "MEDIUM"),
            "asset_scope": evt.get("asset_scope", []),
            "event_type": evt.get("event_type", "OTHER"),
            "detail_link": evt.get("detail_link", ""),
            "raw_text_truncated": truncated,
        }
        out_record.update(clean)
        out_record["parsed_confidence_raw"] = parsed.get("confidence", 0.0)
        _write_ndjson_line(out_path, out_record)

        processed += 1
        written += 1

    return {
        "start_date": start_date,
        "end_date": end_date,
        "events_loaded": len(events),
        "processed": processed,
        "written": written,
        "failures": failures,
        "output": str(out_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Historical Ollama backfill runner")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD (inclusive)")
    parser.add_argument(
        "--output",
        type=Path,
        default=config.OUTBOX_DIR / "parsed_history.ndjson",
        help="NDJSON output path (default: outbox/parsed_history.ndjson)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit number of events parsed (optional)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override Ollama timeout in seconds for this run (optional)",
    )
    parser.add_argument(
        "--connectivity-check-only",
        action="store_true",
        help="Run only the Ollama connectivity check and exit",
    )
    args = parser.parse_args()

    stats = run(
        args.start_date,
        args.end_date,
        args.output,
        max_items=args.max_items,
        timeout=args.timeout,
        connectivity_check_only=args.connectivity_check_only,
    )
    print(json.dumps(stats, indent=2))
