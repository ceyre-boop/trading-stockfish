"""
MT5 live adapter for Trading Stockfish.
Provides initialization and tick streaming in canonical format.
Adds strict validation so no malformed or synthetic ticks reach the engine.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, Generator, Optional

import MetaTrader5 as mt5

from config.env_loader import load_env

logger = logging.getLogger(__name__)
_last_ts_seen: Optional[float] = None


class MT5AdapterError(Exception):
    pass


def initialize_mt5(
    login: Optional[int] = None,
    password: Optional[str] = None,
    server: Optional[str] = None,
) -> None:
    creds = load_env()
    login = login if login is not None else creds.get("MT5_LOGIN")
    password = password if password is not None else creds.get("MT5_PASSWORD")
    server = server if server is not None else creds.get("MT5_SERVER")

    if login is None or password is None or server is None:
        raise MT5AdapterError("MT5 credentials not set in environment or .env")

    if not mt5.initialize(server=server):
        raise MT5AdapterError(f"MT5 initialize failed: {mt5.last_error()}")
    if not mt5.login(int(login), password=password, server=server):
        raise MT5AdapterError(f"MT5 login failed: {mt5.last_error()}")


def shutdown_mt5() -> None:
    mt5.shutdown()


def _canonical_tick(tick, symbol: str) -> Dict:
    mid = (tick.bid + tick.ask) / 2.0
    return {
        "timestamp": float(tick.time),
        "bid": float(tick.bid),
        "ask": float(tick.ask),
        "mid": mid,
        "price": mid,
        "volume": float(getattr(tick, "volume", 0.0)),
        "buy_volume": float(getattr(tick, "volume", 0.0)) / 2.0,
        "sell_volume": float(getattr(tick, "volume", 0.0)) / 2.0,
        "symbol": symbol,
        "raw": tick._asdict(),
    }


def stream_live(symbol: str, poll_interval: float = 0.5) -> Generator[Dict, None, None]:
    if not mt5.symbol_select(symbol, True):
        raise MT5AdapterError(f"Unable to select symbol {symbol}: {mt5.last_error()}")
    if not mt5.initialized() and not mt5.initialize():
        raise MT5AdapterError(
            f"MT5 not initialized while starting stream: {mt5.last_error()}"
        )
    global _last_ts_seen
    while True:
        if not mt5.initialized():
            raise MT5AdapterError("MT5 connection lost during stream")
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            candidate = _canonical_tick(tick, symbol)
            if _validate_tick(candidate, _last_ts_seen):
                _last_ts_seen = candidate["timestamp"]
                yield candidate
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
            logger.error(f"MT5 tick missing required field {field}; skipping")
            return False

    bid = float(tick["bid"])
    ask = float(tick["ask"])
    mid = float(tick["mid"])
    volume = float(tick["volume"])
    ts = float(tick["timestamp"])

    if any(math.isnan(val) or math.isinf(val) for val in [bid, ask, mid, volume, ts]):
        logger.error("MT5 tick has NaN/inf values; skipping")
        return False
    if bid <= 0 or ask <= 0:
        logger.error("MT5 tick has non-positive price; skipping")
        return False
    if ask <= bid:
        logger.error("MT5 tick has non-positive spread (ask <= bid); skipping")
        return False
    spread = ask - bid
    if spread <= 0:
        logger.error("MT5 tick spread must be positive; skipping")
        return False

    expected_mid = (bid + ask) / 2.0
    if not math.isclose(mid, expected_mid, rel_tol=0.0, abs_tol=1e-9):
        logger.error("MT5 tick mid does not match (bid+ask)/2; skipping")
        return False

    if last_ts is not None and ts <= last_ts:
        logger.error("MT5 tick timestamp is not strictly increasing; skipping")
        return False

    return True
