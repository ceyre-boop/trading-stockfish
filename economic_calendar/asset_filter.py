"""
Filter normalized economic events by traded assets before sending to Ollama.
Engine must only see events that match configured assets.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List

BASE_DIR = Path(__file__).resolve().parent
RAW_DB = BASE_DIR / "raw_events.db"


def _connect(db_path: Path = RAW_DB) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def _overlaps(asset_scope: Iterable[str], assets: List[str]) -> bool:
    aset = {a.upper() for a in assets}
    for item in asset_scope:
        if item.upper() in aset:
            return True
    return False


def filter_events(
    assets: List[str], db_path: Path = RAW_DB, limit: int = 200
) -> List[Dict]:
    assets = assets or []
    conn = _connect(db_path)
    rows = conn.execute(
        """
        SELECT event_id, event_name, timestamp_iso, impact_level, currency, forecast,
               previous, detail_link, asset_scope, event_type
        FROM normalized_events
        ORDER BY timestamp_iso DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()

    events: List[Dict] = []
    for row in rows:
        asset_scope = []
        try:
            import json as _json

            asset_scope = _json.loads(row[8]) if row[8] else []
        except Exception:
            asset_scope = []
        if assets and not _overlaps(asset_scope, assets):
            continue
        events.append(
            {
                "origin_id": row[0],
                "event_name": row[1],
                "timestamp": row[2],
                "impact_level": row[3],
                "currency": row[4],
                "forecast": row[5],
                "previous": row[6],
                "detail_link": row[7],
                "asset_scope": asset_scope,
                "event_type": row[9] or "OTHER",
            }
        )
    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter normalized events by assets")
    parser.add_argument("--assets", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    filtered = filter_events(args.assets, RAW_DB, limit=args.limit)
    print(json.dumps({"filtered": len(filtered), "events": filtered}, indent=2))
