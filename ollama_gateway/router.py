"""
Router for Ollama Gateway outputs.
Writes a single engine-facing snapshot and (optionally) POSTs it.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import requests

from economic_calendar.future_events import get_future_events
from ollama_gateway import config, validator

LOGGER = logging.getLogger("ollama_gateway.router")
if not LOGGER.hasHandlers():
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(handler)


def write_snapshot(snapshot: Dict, target: Path = config.OUTBOX_SNAPSHOT) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    tmp.replace(target)
    LOGGER.info("Wrote validated news snapshot to %s", target)
    return target


def post_snapshot(snapshot: Dict, endpoint: str) -> None:
    if not endpoint:
        return
    try:
        resp = requests.post(endpoint, json=snapshot, timeout=10)
        resp.raise_for_status()
        LOGGER.info("Posted snapshot to engine endpoint=%s", endpoint)
    except Exception as exc:  # pragma: no cover - network boundary
        LOGGER.warning("Engine POST failed: %s", exc)


def _merge_snapshot(future_events: List[Dict], parsed_snapshot: Dict) -> Dict:
    parsed_events = (
        parsed_snapshot.get("parsed_events", [])
        if isinstance(parsed_snapshot, dict)
        else []
    )
    aggregated = parsed_snapshot.get(
        "aggregated_scores", validator.aggregate_features(parsed_events)
    )
    ollama_unreachable = bool(parsed_snapshot.get("ollama_unreachable", False))
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "future_events": future_events,
        "parsed_events": parsed_events,
        "aggregated_scores": aggregated,
        "record_count": len(parsed_events),
        "future_count": len(future_events),
        "ollama_unreachable": ollama_unreachable,
    }


def run(
    write: bool = True, post: bool = False, assets: list[str] | None = None
) -> Dict:
    from ollama_gateway import aggregator

    future_events = get_future_events(
        assets or config.TRADED_ASSETS, db_path=Path(config.RAW_EVENTS_DB)
    )
    parsed_snapshot = aggregator.run(write=False, assets=assets or config.TRADED_ASSETS)
    merged = _merge_snapshot(future_events, parsed_snapshot)
    if write:
        write_snapshot(merged)
    if post:
        post_snapshot(merged, config.ENGINE_ENDPOINT)
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Route Ollama Gateway snapshot")
    parser.add_argument("--post", action="store_true", help="POST to engine endpoint")
    parser.add_argument("--assets", nargs="*", default=config.TRADED_ASSETS)
    args = parser.parse_args()
    result = run(write=True, post=args.post, assets=args.assets)
    print(json.dumps({"record_count": result.get("record_count", 0)}, indent=2))
