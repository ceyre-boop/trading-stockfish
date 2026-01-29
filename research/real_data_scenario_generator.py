"""
Real-data scenario generator for deterministic replay.

Converts historical tick + order book data into canonical scenario files under
research/scenarios/YYYY-MM-DD.json.gz with derived microstructure features and
regime tags aligned with ScenarioLibrary requirements.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Sequence

VOL_THRESHOLDS = (0.05, 0.15)
LIQ_THRESHOLDS = (0.02, 0.005)
TREND_THRESHOLD = 0.0


def _safe_float(value: Any, name: str) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numeric for {name}: {value}")
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"Non-finite value for {name}: {value}")
    return f


def _parse_timestamp(value: Any, name: str) -> float:
    if isinstance(value, (int, float)):
        return _safe_float(value, name)
    if isinstance(value, str):
        try:
            return _safe_float(float(value), name)
        except ValueError:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError as exc:
                raise ValueError(f"Invalid timestamp for {name}: {value}") from exc
    raise ValueError(f"Unsupported timestamp type for {name}: {value}")


def _rolling_std(series: Sequence[float], window: int) -> float:
    if len(series) < 2:
        return 0.0
    windowed = series[-window:]
    if len(windowed) < 2:
        return 0.0
    return pstdev(windowed)


def _rolling_mean(series: Sequence[float], window: int) -> float:
    if not series:
        return 0.0
    return mean(series[-window:])


def _slope(series: Sequence[float]) -> float:
    n = len(series)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = mean(series)
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(series))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def _load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    if raw.lstrip().startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _load_csv_records(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def load_ticks(path: str) -> List[Dict[str, Any]]:
    if path.endswith((".json", ".jsonl")):
        ticks = _load_json_records(path)
    else:
        ticks = _load_csv_records(path)
    if not ticks:
        raise ValueError("No ticks loaded")
    parsed: List[Dict[str, Any]] = []
    for t in ticks:
        price = _safe_float(t.get("price"), "price")
        ts = _parse_timestamp(t.get("timestamp"), "timestamp")
        volume = _safe_float(t.get("volume", 0.0), "volume")
        buy_volume_raw = t.get("buy_volume")
        sell_volume_raw = t.get("sell_volume")
        if buy_volume_raw is None or sell_volume_raw is None:
            raise ValueError(
                "buy_volume and sell_volume are required; synthetic balancing is forbidden"
            )
        parsed.append(
            {
                "timestamp": ts,
                "price": price,
                "volume": volume,
                "buy_volume": _safe_float(buy_volume_raw, "buy_volume"),
                "sell_volume": _safe_float(sell_volume_raw, "sell_volume"),
            }
        )
    parsed.sort(key=lambda x: x["timestamp"])
    _validate_monotonic(parsed)
    return parsed


def load_order_books(path: str | None, expected_len: int) -> List[Dict[str, Any]]:
    if not path:
        raise ValueError("Order book path is required; synthetic books are forbidden")
    books = _load_json_records(path)
    if not books:
        raise ValueError("Empty order book data; synthetic books are forbidden")
    normalized: List[Dict[str, Any]] = []
    for book in books:
        ts = _parse_timestamp(book.get("timestamp", 0.0), "order_book.timestamp")
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        if not bids or not asks:
            raise ValueError(
                "Order book entries must include bids and asks; synthetic depth is forbidden"
            )
        normalized.append({"timestamp": ts, "bids": bids, "asks": asks})
    normalized.sort(key=lambda x: x["timestamp"])
    if len(normalized) != expected_len:
        raise ValueError(
            "Order book length must match tick length; padding/cropping is forbidden"
        )
    return normalized


def _validate_monotonic(events: Iterable[Dict[str, Any]]) -> None:
    prev_ts = None
    for event in events:
        ts = event.get("timestamp")
        if ts is None:
            raise ValueError("Missing timestamp")
        if prev_ts is not None and ts < prev_ts:
            raise ValueError("Timestamps must be non-decreasing for determinism")
        prev_ts = ts


def compute_features(
    ticks: List[Dict[str, Any]],
    books: List[Dict[str, Any]],
    vol_window: int = 50,
    trend_window: int = 20,
) -> List[Dict[str, Any]]:
    mid_history: List[float] = []
    derived: List[Dict[str, Any]] = []

    for idx, tick in enumerate(ticks):
        book = books[idx] if idx < len(books) else {}
        bids = book.get("bids") or []
        asks = book.get("asks") or []

        if not bids or not asks:
            raise ValueError(
                "Order book entries must include bids and asks; synthetic spreads are forbidden"
            )
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread = best_ask - best_bid
        if spread <= 0:
            raise ValueError("Non-positive spread detected; data malformed")
        depth = 0.0
        for lvl in bids[:2]:
            depth += lvl[1]
        for lvl in asks[:2]:
            depth += lvl[1]
        if depth <= 0:
            raise ValueError("Depth must be positive; synthetic depth is forbidden")
        liquidity = 1.0 / (spread * depth)

        mid = (best_bid + best_ask) / 2
        mid_history.append(mid)

        vol = _rolling_std(mid_history, vol_window)
        trend_slice = mid_history[-trend_window:]
        trend_slope = _slope(trend_slice)

        lookback_mid = (
            mid_history[-trend_window]
            if len(mid_history) >= trend_window
            else mid_history[0]
        )
        rolling_mean = _rolling_mean(mid_history, trend_window)
        momentum = mid - lookback_mid
        mean_rev = -(mid - rolling_mean)

        if "buy_volume" not in tick or "sell_volume" not in tick:
            raise ValueError(
                "buy_volume and sell_volume required for OFI; synthetic inference forbidden"
            )
        buy_vol = tick.get("buy_volume")
        sell_vol = tick.get("sell_volume")
        ofi = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-9)

        vol_regime = _volatility_regime(vol)
        liq_regime = _liquidity_regime(liquidity)
        trend_regime = _trend_regime(trend_slope)

        derived_tick = {
            **tick,
            "mid": mid,
            "bid": best_bid,
            "ask": best_ask,
            "spread": spread,
            "depth": depth,
            "rolling_volatility": vol,
            "liquidity": liquidity,
            "trend_slope": trend_slope,
            "ofi": ofi,
            "momentum": momentum,
            "mean_reversion": mean_rev,
            "regime": {
                "volatility": vol_regime,
                "liquidity": liq_regime,
                "trend": trend_regime,
            },
        }
        derived.append(derived_tick)

    return derived


def _volatility_regime(vol: float) -> str:
    v1, v2 = VOL_THRESHOLDS
    if vol < v1:
        return "LOW"
    if vol < v2:
        return "NORMAL"
    return "HIGH"


def _liquidity_regime(liquidity: float) -> str:
    l1, l2 = LIQ_THRESHOLDS
    if liquidity > l1:
        return "ROBUST"
    if liquidity > l2:
        return "NORMAL"
    return "FRAGILE"


def _trend_regime(trend: float) -> str:
    if trend > TREND_THRESHOLD:
        return "UP"
    if trend < -TREND_THRESHOLD:
        return "DOWN"
    return "SIDEWAYS"


def _majority_regime(derived_ticks: List[Dict[str, Any]], key: str) -> str:
    counts: Dict[str, int] = {}
    for t in derived_ticks:
        regime = t.get("regime", {}).get(key)
        if regime is None:
            continue
        counts[regime] = counts.get(regime, 0) + 1
    if not counts:
        return "UNKNOWN"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _regime_distribution(
    derived_ticks: List[Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    dist = {"volatility": {}, "liquidity": {}, "trend": {}}
    for t in derived_ticks:
        reg = t.get("regime", {})
        for k in dist.keys():
            val = reg.get(k)
            if val is None:
                continue
            dist[k][val] = dist[k].get(val, 0) + 1
    return dist


def build_scenario(
    derived_ticks: List[Dict[str, Any]],
    books: List[Dict[str, Any]],
    instrument: str,
    date: str,
    session: str,
    macro_regime: str,
    event_tags: List[str],
) -> Dict[str, Any]:
    volatility_regime = _majority_regime(derived_ticks, "volatility")
    liquidity_regime = _majority_regime(derived_ticks, "liquidity")
    trend_regime = _majority_regime(derived_ticks, "trend")

    scenario = {
        "instrument": instrument,
        "date": date,
        "session": session,
        "volatility_regime": volatility_regime,
        "liquidity_regime": liquidity_regime,
        "macro_regime": macro_regime,
        "event_tags": event_tags,
        "ticks": derived_ticks,
        "order_books": books,
        "news_events": [],
        "regime_distribution": _regime_distribution(derived_ticks),
    }
    return scenario


def save_scenario(scenario: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(scenario, f, ensure_ascii=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate real-data scenarios for replay testing"
    )
    parser.add_argument(
        "--ticks", required=True, help="Path to raw tick data (csv, json, or jsonl)"
    )
    parser.add_argument(
        "--books",
        required=False,
        default=None,
        help="Path to order book snapshots (json or jsonl)",
    )
    parser.add_argument(
        "--instrument", required=True, help="Instrument code, e.g., ES or NQ"
    )
    parser.add_argument("--date", required=True, help="Trading day YYYY-MM-DD")
    parser.add_argument(
        "--session", default="FULL", help="Session label, e.g., RTH or GLOBEX"
    )
    parser.add_argument("--macro-regime", default="UNKNOWN", help="Macro regime tag")
    parser.add_argument(
        "--event-tags", nargs="*", default=[], help="Event tags for the scenario"
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Override output filename (without extension)",
    )
    parser.add_argument(
        "--vol-window", type=int, default=50, help="Rolling volatility window"
    )
    parser.add_argument(
        "--trend-window", type=int, default=20, help="Trend slope window"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ticks = load_ticks(args.ticks)
    books = load_order_books(args.books, expected_len=len(ticks))
    if len(books) != len(ticks):
        raise ValueError(
            "Order book length must match tick length; padding/cropping is forbidden"
        )

    derived_ticks = compute_features(
        ticks, books, vol_window=args.vol_window, trend_window=args.trend_window
    )
    scenario = build_scenario(
        derived_ticks,
        books,
        instrument=args.instrument,
        date=args.date,
        session=args.session,
        macro_regime=args.macro_regime,
        event_tags=args.event_tags,
    )

    fname = args.output_name or args.date
    output_path = os.path.join("research", "scenarios", f"{fname}.json.gz")
    save_scenario(scenario, output_path)
    print(f"Saved scenario to {output_path} ({len(scenario['ticks'])} ticks)")


if __name__ == "__main__":
    main()
