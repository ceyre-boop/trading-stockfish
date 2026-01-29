"""
Liquidity Metrics v4.0-C
Compute spread, liquidity score, and stress flags from order book snapshots.
"""

from typing import Any, Dict, Iterable, Tuple


def _level_price(level: Any) -> float:
    if isinstance(level, (list, tuple)):
        return float(level[0])
    return float(level.get("price", 0.0))


def _level_volume(level: Any) -> float:
    if isinstance(level, (list, tuple)):
        return float(level[1])
    return float(level.get("volume", 0.0))


def compute_spread(order_book: Dict) -> float:
    if not order_book["bids"] or not order_book["asks"]:
        return float("inf")
    best_bid = _level_price(order_book["bids"][0])
    best_ask = _level_price(order_book["asks"][0])
    return float(best_ask - best_bid)


def compute_liquidity_score(order_book: Dict, top_n: int = 5) -> float:
    # Score: sum of depth at top N levels, penalized by spread
    depth = 0.0
    for side in ["bids", "asks"]:
        levels = order_book[side][:top_n]
        depth += sum(_level_volume(lvl) for lvl in levels)
    spread = compute_spread(order_book)
    if spread == 0:
        spread = 0.01  # avoid div by zero
    penalty = max(0.5, spread * 2.25)
    score = depth / penalty
    return float(score)


def detect_liquidity_stress(
    order_book: Dict, prev_order_book: Dict = None, top_n: int = 5
) -> Dict[str, bool]:
    flags = {
        "spread_spike_flag": False,
        "thin_book_flag": False,
        "one_sided_liquidity_flag": False,
    }
    spread = compute_spread(order_book)
    # Spread spike: spread > 2x median or > threshold
    if spread > 2.0:
        flags["spread_spike_flag"] = True
    # Thin book: depth at top N < threshold
    min_depth = min(
        sum(_level_volume(lvl) for lvl in order_book["bids"][:top_n]),
        sum(_level_volume(lvl) for lvl in order_book["asks"][:top_n]),
    )
    if min_depth < 10:
        flags["thin_book_flag"] = True
    # One-sided: one side has < 10% of the other
    bid_depth = sum(_level_volume(lvl) for lvl in order_book["bids"][:top_n])
    ask_depth = sum(_level_volume(lvl) for lvl in order_book["asks"][:top_n])
    if bid_depth < 0.1 * ask_depth or ask_depth < 0.1 * bid_depth:
        flags["one_sided_liquidity_flag"] = True
    return flags


def _coerce_order_book(book: Any) -> Dict[str, Any]:
    """Normalize OrderBookModel or dict into tuple-based levels."""
    bids_raw: Iterable[Any]
    asks_raw: Iterable[Any]
    if hasattr(book, "bids") and hasattr(book, "asks"):
        bids_raw = getattr(book, "bids", []) or []
        asks_raw = getattr(book, "asks", []) or []
    else:
        bids_raw = book.get("bids", []) if isinstance(book, dict) else []
        asks_raw = book.get("asks", []) if isinstance(book, dict) else []

    def _as_tuple(level: Any) -> Tuple[float, float]:
        if isinstance(level, (list, tuple)):
            return (float(level[0]), float(level[1]))
        if hasattr(level, "price") and hasattr(level, "volume"):
            return (float(level.price), float(level.volume))
        return (float(level.get("price", 0.0)), float(level.get("volume", 0.0)))

    return {
        "bids": [_as_tuple(lvl) for lvl in bids_raw],
        "asks": [_as_tuple(lvl) for lvl in asks_raw],
    }


def compute_liquidity_metrics(book: Any, top_n: int = 3) -> Dict[str, Any]:
    """Deterministic liquidity metrics wrapper used by tests.

    Args:
        book: OrderBookModel or mapping with bids/asks
        top_n: Number of levels to use for depth calculations

    Returns:
        Dict with spread, liquidity_score, and stress_flags
    """

    ob = _coerce_order_book(book)
    spread = compute_spread(ob)
    score = compute_liquidity_score(ob, top_n=top_n)
    stress = detect_liquidity_stress(ob, top_n=top_n)
    return {
        "spread": spread,
        "liquidity_score": score,
        "stress_flags": stress,
    }
