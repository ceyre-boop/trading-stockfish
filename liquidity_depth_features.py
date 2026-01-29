"""
Deterministic liquidity depth features.
"""

from typing import Dict, Iterable, Optional


def _sum_depth(levels: Optional[Iterable]) -> float:
    if not levels:
        return 0.0
    total = 0.0
    for lvl in levels:
        if isinstance(lvl, dict):
            total += float(lvl.get("volume", 0.0) or 0.0)
        elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
            total += float(lvl[1] or 0.0)
        else:
            total += float(getattr(lvl, "volume", 0.0) or 0.0)
    return max(total, 0.0)


def compute_depth_features(order_book: Dict) -> Dict[str, float]:
    bids = order_book.get("bids") if isinstance(order_book, dict) else None
    asks = order_book.get("asks") if isinstance(order_book, dict) else None
    bid_depth = _sum_depth(bids)
    ask_depth = _sum_depth(asks)
    total_depth = bid_depth + ask_depth
    imbalance = 0.0
    if total_depth > 0:
        imbalance = (bid_depth - ask_depth) / (total_depth + 1e-9)
    best_bid = (
        float(bids[0]["price"]) if bids and bids[0].get("price") is not None else None
    )
    best_ask = (
        float(asks[0]["price"]) if asks and asks[0].get("price") is not None else None
    )
    spread = (
        max(best_ask - best_bid, 0.0)
        if best_bid is not None and best_ask is not None
        else 0.0
    )
    return {
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "total_depth": total_depth,
        "depth_imbalance": imbalance,
        "top_of_book_spread": spread,
    }
