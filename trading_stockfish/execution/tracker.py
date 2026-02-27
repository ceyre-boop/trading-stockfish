"""
Execution quality tracking: slippage, fill rates, and round-trip latency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class Fill:
    """Record for a single order fill."""
    order_id: str
    intended_price: float
    filled_price: float
    quantity: float
    side: str           # "BUY" or "SELL"
    latency_ms: float   # time between order submission and fill (ms)
    filled: bool = True


@dataclass
class ExecutionStats:
    total_fills: int = 0
    fill_rate: float = 0.0          # filled / attempted
    mean_slippage_bps: float = 0.0  # average slippage in basis points
    mean_latency_ms: float = 0.0    # average fill latency


class ExecutionTracker:
    """Collects fill records and computes aggregate execution statistics."""

    def __init__(self) -> None:
        self._fills: List[Fill] = []
        self._attempted: int = 0

    def record_attempt(self) -> None:
        """Register that an order was attempted (whether filled or not)."""
        self._attempted += 1

    def record_fill(self, fill: Fill) -> None:
        """Add a fill record."""
        self._fills.append(fill)

    def slippage_bps(self, fill: Fill) -> float:
        """Slippage in basis points for a single fill."""
        if fill.intended_price == 0:
            return 0.0
        raw = (fill.filled_price - fill.intended_price) / fill.intended_price
        if fill.side.upper() == "BUY":
            slip = raw       # adverse for buyer: filled above intended
        else:
            slip = -raw      # adverse for seller: filled below intended
        return slip * 10_000

    def stats(self) -> ExecutionStats:
        """Compute aggregate statistics over all recorded fills."""
        n = len(self._fills)
        attempted = max(self._attempted, n)
        if n == 0:
            return ExecutionStats(total_fills=0, fill_rate=0.0)
        fill_rate = n / attempted if attempted > 0 else 1.0
        slippages = [self.slippage_bps(f) for f in self._fills]
        latencies = [f.latency_ms for f in self._fills]
        return ExecutionStats(
            total_fills=n,
            fill_rate=fill_rate,
            mean_slippage_bps=float(sum(slippages) / n),
            mean_latency_ms=float(sum(latencies) / n),
        )

    def reset(self) -> None:
        self._fills.clear()
        self._attempted = 0
