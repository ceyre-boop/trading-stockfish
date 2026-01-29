"""
Deterministic advanced order book engine v1.
Builds multi-level L2 snapshots, imbalance, spread dynamics, hidden liquidity
heuristics, and queue position estimates. Replay/live consistent with
micro-update filtering.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class OrderBookFeaturesConfig:
    max_levels_per_side: int = 10
    aggregation_mode: str = "RAW"  # or "AGGREGATED_BY_TICK"
    tick_size: float = 0.01
    queue_fraction: float = 0.25
    spread_shift_threshold: float = 0.1  # 10% vs recent average
    hidden_aggressive_threshold: float = 5.0
    hidden_size_hold_ratio: float = 0.7


@dataclass(frozen=True)
class L2Level:
    price: float
    size: float


@dataclass(frozen=True)
class L2Snapshot:
    bids: List[L2Level]
    asks: List[L2Level]
    spread: float


_DEF_EPS = 1e-9


def _round_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(price / tick) * tick


def _parse_raw_l2(
    raw_l2: Sequence,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    if isinstance(raw_l2, dict):
        bids = raw_l2.get("bids", []) or []
        asks = raw_l2.get("asks", []) or []
        return _coerce_levels(bids), _coerce_levels(asks)
    bids: List[Tuple[float, float]] = []
    asks: List[Tuple[float, float]] = []
    for entry in raw_l2:
        if isinstance(entry, dict):
            price = float(entry.get("price", 0.0) or 0.0)
            size = float(entry.get("size", entry.get("volume", 0.0)) or 0.0)
            side = entry.get("side", "")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
            price, size, side = float(entry[0]), float(entry[1]), entry[2]
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            price, size, side = float(entry[0]), float(entry[1]), ""
        else:
            price = float(getattr(entry, "price", 0.0) or 0.0)
            size = float(getattr(entry, "size", getattr(entry, "volume", 0.0)) or 0.0)
            side = getattr(entry, "side", "")
        if str(side).lower().startswith("b"):
            bids.append((price, size))
        elif str(side).lower().startswith("a"):
            asks.append((price, size))
    return _coerce_levels(bids), _coerce_levels(asks)


def _coerce_levels(levels: Iterable) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    for lvl in levels:
        if isinstance(lvl, dict):
            price = float(lvl.get("price", 0.0) or 0.0)
            size = float(lvl.get("volume", lvl.get("size", 0.0)) or 0.0)
        elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
            price = float(lvl[0] or 0.0)
            size = float(lvl[1] or 0.0)
        else:
            price = float(getattr(lvl, "price", 0.0) or 0.0)
            size = float(getattr(lvl, "volume", getattr(lvl, "size", 0.0)) or 0.0)
        if size > 0:
            result.append((price, size))
    return result


def _aggregate(
    levels: List[Tuple[float, float]], tick_size: float, reverse: bool
) -> List[L2Level]:
    buckets: Dict[float, float] = {}
    for price, size in levels:
        b_price = _round_to_tick(price, tick_size)
        buckets[b_price] = buckets.get(b_price, 0.0) + size
    sorted_prices = sorted(buckets.keys(), reverse=reverse)
    return [L2Level(price=p, size=buckets[p]) for p in sorted_prices]


def _limit_levels(levels: List[L2Level], max_levels: int) -> List[L2Level]:
    if max_levels <= 0:
        return []
    return levels[:max_levels]


def _clean_micro_updates(levels: List[L2Level]) -> List[L2Level]:
    if not levels:
        return []
    max_size = max(l.size for l in levels)
    threshold = max_size * 0.01
    return [l for l in levels if l.size >= threshold]


def build_l2_snapshot(
    raw_l2: Sequence, config: OrderBookFeaturesConfig | None = None
) -> L2Snapshot:
    cfg = config or OrderBookFeaturesConfig()
    bids_raw, asks_raw = _parse_raw_l2(raw_l2)
    reverse_bids = True
    reverse_asks = False
    if cfg.aggregation_mode.upper() == "AGGREGATED_BY_TICK":
        bids_levels = _aggregate(bids_raw, cfg.tick_size, reverse=reverse_bids)
        asks_levels = _aggregate(asks_raw, cfg.tick_size, reverse=reverse_asks)
    else:
        bids_levels = [
            L2Level(price=p, size=s)
            for p, s in sorted(bids_raw, key=lambda x: (-x[0], -x[1]))
        ]
        asks_levels = [
            L2Level(price=p, size=s)
            for p, s in sorted(asks_raw, key=lambda x: (x[0], -x[1]))
        ]
    bids_levels = _clean_micro_updates(bids_levels)
    asks_levels = _clean_micro_updates(asks_levels)
    bids_levels = _limit_levels(bids_levels, cfg.max_levels_per_side)
    asks_levels = _limit_levels(asks_levels, cfg.max_levels_per_side)
    spread = (
        (asks_levels[0].price - bids_levels[0].price)
        if (bids_levels and asks_levels)
        else 0.0
    )
    return L2Snapshot(bids=bids_levels, asks=asks_levels, spread=spread)


def _imbalance(bids: List[L2Level], asks: List[L2Level]) -> Tuple[float, float]:
    bid_top = bids[0].size if bids else 0.0
    ask_top = asks[0].size if asks else 0.0
    top = (bid_top - ask_top) / (bid_top + ask_top + _DEF_EPS)
    bid_sum = sum(l.size for l in bids)
    ask_sum = sum(l.size for l in asks)
    multi = (bid_sum - ask_sum) / (bid_sum + ask_sum + _DEF_EPS)
    return top, multi


def _spread_dynamics(
    snapshot: L2Snapshot, recent_spreads: Sequence[float], cfg: OrderBookFeaturesConfig
) -> Tuple[int, bool, bool, str]:
    current_spread = snapshot.spread
    avg_spread = (
        sum(recent_spreads) / len(recent_spreads) if recent_spreads else current_spread
    )
    threshold = avg_spread * cfg.spread_shift_threshold
    widening = current_spread > avg_spread + threshold
    tightening = current_spread < max(avg_spread - threshold, 0.0)
    shift = "NORMAL"
    if widening:
        shift = "WIDENING"
    elif tightening:
        shift = "TIGHTENING"
    spread_ticks = (
        int(round(current_spread / (cfg.tick_size or 1.0))) if current_spread else 0
    )
    return spread_ticks, widening, tightening, shift


def _hidden_liquidity(
    snapshot: L2Snapshot,
    prev_snapshot: Optional[L2Snapshot],
    cfg: OrderBookFeaturesConfig,
    aggressive_buy: float,
    aggressive_sell: float,
) -> Tuple[bool, bool]:
    hidden_bid = False
    hidden_ask = False
    if prev_snapshot and snapshot.bids and prev_snapshot.bids:
        bid_price_same = snapshot.bids[0].price == prev_snapshot.bids[0].price
        if bid_price_same:
            hold_ratio = snapshot.bids[0].size / (prev_snapshot.bids[0].size + _DEF_EPS)
            if (
                aggressive_sell >= cfg.hidden_aggressive_threshold
                and hold_ratio >= cfg.hidden_size_hold_ratio
            ):
                hidden_bid = True
    if prev_snapshot and snapshot.asks and prev_snapshot.asks:
        ask_price_same = snapshot.asks[0].price == prev_snapshot.asks[0].price
        if ask_price_same:
            hold_ratio = snapshot.asks[0].size / (prev_snapshot.asks[0].size + _DEF_EPS)
            if (
                aggressive_buy >= cfg.hidden_aggressive_threshold
                and hold_ratio >= cfg.hidden_size_hold_ratio
            ):
                hidden_ask = True
    if not prev_snapshot:
        if snapshot.bids and aggressive_sell >= cfg.hidden_aggressive_threshold:
            if (
                snapshot.bids[0].size
                >= (snapshot.asks[0].size if snapshot.asks else snapshot.bids[0].size)
                * 1.2
            ):
                hidden_bid = True
        if snapshot.asks and aggressive_buy >= cfg.hidden_aggressive_threshold:
            if (
                snapshot.asks[0].size
                >= (snapshot.bids[0].size if snapshot.bids else snapshot.asks[0].size)
                * 1.2
            ):
                hidden_ask = True
    return hidden_bid, hidden_ask


def _queue_position(snapshot: L2Snapshot, cfg: OrderBookFeaturesConfig) -> float:
    if not snapshot.bids:
        return 0.0
    visible_before = snapshot.bids[0].size * cfg.queue_fraction
    total_visible = snapshot.bids[0].size
    estimate = visible_before / (total_visible + _DEF_EPS)
    return max(0.0, min(1.0, estimate))


def compute_orderbook_features(
    raw_l2: Sequence,
    config: OrderBookFeaturesConfig | None = None,
    recent_spreads: Sequence[float] | None = None,
    aggressive_buy: float = 0.0,
    aggressive_sell: float = 0.0,
    prev_snapshot: Optional[L2Snapshot] = None,
) -> Dict[str, object]:
    cfg = config or OrderBookFeaturesConfig()
    recent_spreads = recent_spreads or []
    snapshot = build_l2_snapshot(raw_l2, cfg)
    top_imb, multi_imb = _imbalance(snapshot.bids, snapshot.asks)
    spread_ticks, widening, tightening, shift = _spread_dynamics(
        snapshot, recent_spreads, cfg
    )
    hidden_bid, hidden_ask = _hidden_liquidity(
        snapshot, prev_snapshot, cfg, aggressive_buy, aggressive_sell
    )
    queue_estimate = _queue_position(snapshot, cfg)

    return {
        "l2_bids": [{"price": l.price, "size": l.size} for l in snapshot.bids],
        "l2_asks": [{"price": l.price, "size": l.size} for l in snapshot.asks],
        "top_level_imbalance": top_imb,
        "multi_level_imbalance": multi_imb,
        "spread_ticks": spread_ticks,
        "spread_widening": widening,
        "spread_tightening": tightening,
        "microstructure_shift": shift,
        "hidden_bid_liquidity": hidden_bid,
        "hidden_ask_liquidity": hidden_ask,
        "queue_position_estimate": queue_estimate,
    }
