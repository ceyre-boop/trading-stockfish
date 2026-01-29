from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketState:
    """Lightweight market state for live testing harness."""

    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: float
    symbol: str = "EURUSD"
    bid: Optional[float] = None
    ask: Optional[float] = None
