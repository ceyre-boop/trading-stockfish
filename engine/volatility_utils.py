"""
Shared deterministic volatility utilities.
"""

from typing import Iterable, Optional, Union

import numpy as np

Number = Union[int, float]
PriceInput = Optional[Union[Number, Iterable[Number]]]


def compute_atr(prices: PriceInput, window: int = 14, floor: float = 1e-8) -> float:
    """Compute a simple ATR-like measure with a deterministic floor.

    Accepts either an iterable of prices or a pre-computed numeric ATR. Always
    returns a non-negative float and applies a floor to avoid divide-by-zero
    artifacts in downstream volatility features.
    """
    # Direct numeric ATR input
    if prices is None:
        return float(max(floor, 0.0))

    if isinstance(prices, (int, float)):
        return float(max(prices, floor, 0.0))

    # Iterable of prices
    arr = np.asarray(list(prices), dtype=float)
    if arr.size < 2:
        return float(max(floor, 0.0))

    effective_window = max(1, min(window, arr.size - 1))
    diffs = np.abs(np.diff(arr))
    atr = float(np.mean(diffs[-effective_window:]))
    return float(max(atr, floor, 0.0))
