"""
Router for Twitter microservice.
- Reads latest processed tweets snapshot from DB
- Writes payload to outbox file for engine ingestion
- Optional HTTP post to STOCKFISH_NEWS_ENDPOINT if configured
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import requests

from twitter_microservice import config, twitter_parser

LOGGER = logging.getLogger("twitter_router")
if not LOGGER.hasHandlers():
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(handler)


def _write_outbox(payload: Dict, outbox: Path) -> Path:
    outbox.mkdir(parents=True, exist_ok=True)
    fname = (
        "twitter_"
        + datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
        + ".json"
    )
    target = outbox / fname
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _post_if_configured(payload: Dict, endpoint: str) -> None:
    if not endpoint:
        return
    try:
        resp = requests.post(endpoint, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        LOGGER.warning("Failed to POST to engine: %s", exc)


def run(db_path: str, snapshot_limit: int, outbox: Path, endpoint: str) -> Dict:
    snapshot = twitter_parser.process(
        db_path, limit=snapshot_limit, snapshot_limit=snapshot_limit
    )
    path = _write_outbox(snapshot, outbox)
    _post_if_configured(snapshot, endpoint)
    return {"written": str(path), "count": snapshot.get("count", 0)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Route Twitter snapshot to engine")
    parser.add_argument("--db", default=config.DATABASE_FILE)
    parser.add_argument("--snapshot", type=int, default=100)
    parser.add_argument("--outbox", default=str(config.ROUTER_OUTBOX))
    parser.add_argument("--endpoint", default=config.ENGINE_ENDPOINT)
    args = parser.parse_args()

    result = run(
        db_path=args.db,
        snapshot_limit=args.snapshot,
        outbox=Path(args.outbox),
        endpoint=args.endpoint,
    )
    print(json.dumps(result, indent=2))
