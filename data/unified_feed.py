"""
Unified market feed interface for Trading Stockfish.
Supports polygon historical/live, mt5 live, and scenario replay sources.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from typing import Any, Dict, Generator, Iterable, List, Optional

from data import mt5_adapter, polygon_adapter

CanonicalEvent = Dict[str, Any]


def _canonicalize_tick(tick: Dict, symbol: str) -> Dict:
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
        if tick.get(field) is None:
            raise RuntimeError(
                f"Missing required tick field {field}; synthetic fallback is forbidden"
            )

    bid = float(tick["bid"])
    ask = float(tick["ask"])
    mid_raw = float(tick["mid"])
    volume = float(tick["volume"])
    buy_volume = float(tick["buy_volume"])
    sell_volume = float(tick["sell_volume"])
    ts = float(tick["timestamp"])

    values = [bid, ask, mid_raw, volume, buy_volume, sell_volume, ts]
    if any(math.isnan(v) or math.isinf(v) for v in values):
        raise RuntimeError("NaN/inf detected in tick; reject event")
    if bid <= 0 or ask <= 0 or mid_raw <= 0:
        raise RuntimeError("Tick contains zero/negative prices; reject event")
    if ask <= bid:
        raise RuntimeError("Tick spread must be positive; reject event")

    mid_expected = (bid + ask) / 2.0
    if not math.isclose(mid_raw, mid_expected, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("Tick mid must equal (bid+ask)/2; reject event")

    if "raw" not in tick or tick.get("raw") is None:
        raise RuntimeError("Tick missing raw payload; reject event")

    return {
        **tick,
        "bid": bid,
        "ask": ask,
        "mid": mid_expected,
        "price": float(tick.get("price", mid_expected)),
        "volume": volume,
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "symbol": symbol,
    }


def _load_scenario(path: str) -> Dict:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class UnifiedFeed:
    def __init__(self, source: str, symbol: str, scenario_path: Optional[str] = None):
        self.source = source
        self.symbol = symbol
        self.scenario_path = scenario_path

    def load_historical(self, params: Optional[Dict] = None) -> List[CanonicalEvent]:
        params = params or {}
        if self.source == "polygon":
            date = params.get("date")
            timespan = params.get("timespan", "minute")
            ticks = polygon_adapter.get_historical_bars(self.symbol, date, timespan)
            events: List[CanonicalEvent] = []
            for tick in ticks:
                canon = _canonicalize_tick(tick, self.symbol)
                events.append({"tick": canon, "book": {}, "market": self.symbol})
            return events
        if self.source == "scenario":
            if not self.scenario_path:
                raise ValueError("scenario_path required for scenario source")
            scenario = _load_scenario(self.scenario_path)
            books = scenario.get("order_books", [])
            if len(books) != len(scenario.get("ticks", [])):
                raise RuntimeError(
                    "Order books must align 1:1 with ticks; synthetic padding is forbidden"
                )
            events: List[CanonicalEvent] = []
            for idx, tick in enumerate(scenario.get("ticks", [])):
                book = books[idx]
                canon = _canonicalize_tick(
                    tick, scenario.get("instrument", self.symbol)
                )
                if not book.get("bids") or not book.get("asks"):
                    raise RuntimeError(
                        "Order book must contain bids and asks; reject event"
                    )
                events.append(
                    {
                        "tick": canon,
                        "book": book,
                        "market": scenario.get("instrument", self.symbol),
                    }
                )
            return events
        raise ValueError(f"Unsupported source for historical load: {self.source}")

    def stream(
        self, params: Optional[Dict] = None
    ) -> Generator[CanonicalEvent, None, None]:
        params = params or {}
        if self.source == "polygon":
            for tick in polygon_adapter.stream_live(self.symbol):
                canon = _canonicalize_tick(tick, self.symbol)
                yield {"tick": canon, "book": {}, "market": self.symbol}
        elif self.source == "mt5":
            mt5_adapter.initialize_mt5()
            try:
                for tick in mt5_adapter.stream_live(
                    self.symbol, poll_interval=params.get("poll_interval", 0.5)
                ):
                    canon = _canonicalize_tick(tick, self.symbol)
                    yield {"tick": canon, "book": {}, "market": self.symbol}
            finally:
                mt5_adapter.shutdown_mt5()
        elif self.source == "scenario":
            for event in self.load_historical(params):
                yield event
        else:
            raise ValueError(f"Unsupported source: {self.source}")
