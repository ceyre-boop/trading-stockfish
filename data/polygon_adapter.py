"""
Polygon adapter for historical and live data.
Normalizes outputs to canonical tick format used by Trading Stockfish.
Adds strict tick validation to prevent malformed or synthetic data from entering the engine.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, Generator, List, Optional

import requests

from config.env_loader import load_env

BASE_URL = "https://api.polygon.io"
DEFAULT_TIMEOUT = 10
MAX_RETRIES = 3
POLL_INTERVAL_SECONDS = 1.0

logger = logging.getLogger(__name__)
_last_ts_seen: Optional[float] = None


class PolygonAPIError(Exception):
    pass


def _api_key() -> str:
    key = load_env().get("POLYGON_API_KEY")
    if not key:
        raise PolygonAPIError("POLYGON_API_KEY not set in environment or .env")
    return key


def _request_json(url: str, params: Optional[Dict[str, str]] = None) -> Dict:
    params = params or {}
    params["apiKey"] = _api_key()
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 429:
                time.sleep(min(2 * attempt, 5))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # requests exceptions
            last_exc = exc
            time.sleep(min(2 * attempt, 5))
    raise PolygonAPIError(f"Request failed after {MAX_RETRIES} attempts: {last_exc}")


def _canonical_tick_from_bar(bar: Dict, symbol: str) -> Dict[str, float]:
    ts_raw = bar.get("t") or bar.get("timestamp")
    bid = bar.get("bid")
    ask = bar.get("ask")
    volume = bar.get("v") or bar.get("volume")
    if ts_raw is None or bid is None or ask is None or volume is None:
        raise PolygonAPIError("Malformed bar: missing bid/ask/timestamp/volume")
    ts_val = float(ts_raw)
    ts = ts_val / 1000.0 if ts_val > 1e12 else ts_val
    mid = (float(bid) + float(ask)) / 2.0
    return {
        "timestamp": ts,
        "bid": float(bid),
        "ask": float(ask),
        "mid": mid,
        "price": mid,
        "volume": float(volume),
        "buy_volume": float(volume) / 2.0,
        "sell_volume": float(volume) / 2.0,
        "symbol": symbol,
        "raw": bar,
    }


def get_historical_bars(symbol: str, date: str, timespan: str = "minute") -> List[Dict]:
    """
    Fetch historical bars for a specific date and normalize to canonical ticks.
    date: YYYY-MM-DD
    timespan: e.g., "minute", "second"
    """
    url = f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/1/{timespan}/{date}/{date}"
    data = _request_json(
        url, params={"adjusted": "true", "sort": "asc", "limit": 50000}
    )
    results = data.get("results") or []
    ticks: List[Dict] = []
    last_ts: Optional[float] = None
    for bar in results:
        try:
            candidate = _canonical_tick_from_bar(bar, symbol)
            if _validate_tick(candidate, last_ts):
                last_ts = candidate["timestamp"]
                ticks.append(candidate)
        except PolygonAPIError as exc:
            logger.critical(f"Skipping malformed bar: {exc}")
            continue
    return ticks


def _canonical_tick_from_quote(payload: Dict, symbol: str) -> Optional[Dict]:
    quote = payload.get("results") or payload.get("result") or payload
    if not quote:
        return None
    bid = quote.get("bP") or quote.get("bid")
    ask = quote.get("aP") or quote.get("ask")
    ts_raw = quote.get("t") or quote.get("timestamp")
    if bid is None or ask is None or ts_raw is None:
        return None
    mid = (float(bid) + float(ask)) / 2.0
    volume = quote.get("s") or quote.get("volume") or 0.0
    ts_val_f = float(ts_raw)
    ts_val = ts_val_f / 1000.0 if ts_val_f > 1e12 else ts_val_f
    return {
        "timestamp": ts_val,
        "bid": float(bid),
        "ask": float(ask),
        "mid": mid,
        "price": mid,
        "volume": float(volume),
        "buy_volume": float(volume) / 2.0,
        "sell_volume": float(volume) / 2.0,
        "symbol": symbol,
        "raw": quote,
    }


def stream_live(
    symbol: str, poll_interval: float = POLL_INTERVAL_SECONDS
) -> Generator[Dict, None, None]:
    """
    Stream live data using REST polling (WebSocket optional). Yields canonical ticks.
    """
    url = f"{BASE_URL}/v2/last/nbbo/{symbol}"
    global _last_ts_seen
    while True:
        try:
            payload = _request_json(url)
            tick = _canonical_tick_from_quote(payload, symbol)
            if tick and _validate_tick(tick, _last_ts_seen):
                _last_ts_seen = tick["timestamp"]
                yield tick
        except Exception as exc:
            logger.critical(f"Polygon live stream error: {exc}")
            time.sleep(2.0)
        time.sleep(poll_interval)


def _validate_tick(tick: Dict, last_ts: Optional[float]) -> bool:
    required = [
        "timestamp",
        "bid",
        "ask",
        "mid",
        "volume",
        "buy_volume",
        "sell_volume",
        "raw",
    ]
    for field in required:
        if field not in tick or tick[field] is None:
            logger.critical(f"Polygon tick missing required field {field}; skipping")
            return False

    bid = float(tick["bid"])
    ask = float(tick["ask"])
    mid = float(tick["mid"])
    volume = float(tick["volume"])
    ts = float(tick["timestamp"])

    if any(math.isnan(val) or math.isinf(val) for val in [bid, ask, mid, volume, ts]):
        logger.critical("Polygon tick has NaN/inf values; skipping")
        return False
    if bid <= 0 or ask <= 0:
        logger.critical("Polygon tick has non-positive price; skipping")
        return False
    if ask <= bid:
        logger.critical("Polygon tick has non-positive spread (ask <= bid); skipping")
        return False
    spread = ask - bid
    if spread <= 0:
        logger.critical("Polygon tick spread must be positive; skipping")
        return False

    expected_mid = (bid + ask) / 2.0
    if not math.isclose(mid, expected_mid, rel_tol=0.0, abs_tol=1e-9):
        logger.critical("Polygon tick mid does not match (bid+ask)/2; skipping")
        return False

    if last_ts is not None and ts <= last_ts:
        logger.critical("Polygon tick timestamp is not strictly increasing; skipping")
        return False

    return True
