"""
Future event visibility helpers for engine-safe consumption.
- Query normalized_events
- Compute time deltas and risk windows
- Compute macro pressure score from impact, proximity, and asset overlap
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
RAW_DB = BASE_DIR / "raw_events.db"

IMPACT_WEIGHT = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 1.0}


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _proximity_weight(minutes: float) -> float:
    if minutes <= 30:
        return 1.0
    if minutes <= 120:
        return 0.7
    if minutes <= 720:
        return 0.4
    return 0.2


def _asset_overlap(asset_scope: List[str], assets: List[str]) -> float:
    if not assets:
        return 1.0
    aset = {a.upper() for a in assets}
    overlap = len([a for a in asset_scope if a.upper() in aset])
    if overlap == 0:
        return 0.1
    return min(1.0, overlap / max(1, len(assets)))


def get_future_events(
    assets: List[str],
    db_path: Path = RAW_DB,
    horizon_days: int = 60,
    now: datetime | None = None,
) -> List[Dict]:
    now_dt = now or datetime.now(tz=timezone.utc)
    conn = _connect(db_path)
    rows = conn.execute(
        """
        SELECT n.event_id, n.event_name, n.timestamp_iso, n.impact_level, n.currency, n.asset_scope, n.event_type, n.detail_link, COALESCE(n.full_text, r.full_text)
        FROM normalized_events n
        LEFT JOIN raw_events r ON n.event_id = r.event_id
        ORDER BY n.timestamp_iso ASC
        """
    ).fetchall()
    conn.close()

    events: List[Dict] = []
    for row in rows:
        ts = _parse_ts(row[2])
        if not ts:
            continue
        delta = (ts - now_dt).total_seconds() / 60.0
        if delta < -10080:  # older than 7 days past
            continue
        if delta > horizon_days * 1440:
            continue
        asset_scope = []
        try:
            import json as _json

            asset_scope = _json.loads(row[5]) if row[5] else []
        except Exception:
            asset_scope = []
        if assets and not any(
            a.upper() in {s.upper() for s in assets} for a in asset_scope
        ):
            continue
        impact = str(row[3] or "MEDIUM").upper()
        impact_w = IMPACT_WEIGHT.get(impact, 0.6)
        prox_w = _proximity_weight(delta)
        asset_w = _asset_overlap(asset_scope, assets)
        macro_pressure = round(impact_w * prox_w * asset_w, 4)
        risk_window = delta <= 30 and delta >= -30
        events.append(
            {
                "origin_id": row[0],
                "event_name": row[1],
                "timestamp": ts.isoformat(),
                "time_delta_minutes": delta,
                "impact_level": impact,
                "event_type": row[6] or "OTHER",
                "asset_scope": asset_scope,
                "risk_window": risk_window,
                "macro_pressure_score": macro_pressure,
                "detail_link": row[7] or "",
                "full_text": row[8] or "",
            }
        )
    return sorted(events, key=lambda e: e["time_delta_minutes"])


def next_event_summary(events: List[Dict]) -> Tuple[str, float, str, float]:
    if not events:
        return ("NONE", 0.0, "NONE", 0.0)
    # Events are sorted by time delta
    first = events[0]
    return (
        first.get("event_type", "NONE"),
        float(first.get("time_delta_minutes", 0.0)),
        first.get("impact_level", "MEDIUM"),
        float(first.get("macro_pressure_score", 0.0)),
    )
