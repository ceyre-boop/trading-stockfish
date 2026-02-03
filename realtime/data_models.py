"""
Real-Time Data Models - MarketUpdate, Price Ticks, Order Books

Core data structures for real-time market data and execution:
  - MarketUpdate: Universal wrapper for all market events
  - PriceTick: Level 1 quote data
  - OrderBookSnapshot: Level 2+ order book
  - OHLCVBar: Candlestick/bar data
  - NewsEvent: News with sentiment
  - MacroEvent: Macro economic data

Author: Trading Stockfish Development Team
Date: January 19, 2026
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class DataType(Enum):
    """Market data event types."""

    PRICE_TICK = "price_tick"
    ORDERBOOK_SNAPSHOT = "orderbook_snapshot"
    ORDERBOOK_UPDATE = "orderbook_update"
    OHLCV_BAR = "ohlcv_bar"
    NEWS = "news"
    MACRO = "macro"
    TRADE = "trade"
    EXECUTION = "execution"
    ERROR = "error"


# ============================================================================
# Core Data Structures
# ============================================================================


@dataclass
class PriceTick:
    """
    Level 1 price quote data.

    Represents the best bid/ask quote and last trade price.
    """

    symbol: str
    bid: float
    ask: float
    last: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    last_volume: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exchange: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @property
    def mid_price(self) -> float:
        """Mid price (average of bid and ask)."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Spread as absolute amount."""
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return (self.spread / self.mid_price) * 10000 if self.mid_price > 0 else 0


@dataclass
class OrderBookLevel:
    """Single level in order book (bid or ask)."""

    price: float
    quantity: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class OrderBookSnapshot:
    """
    Level 2 order book snapshot.

    Complete view of bids and asks at a point in time.
    """

    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exchange: str = "UNKNOWN"
    sequence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "bids": [b.to_dict() for b in self.bids],
            "asks": [a.to_dict() for a in self.asks],
            "timestamp": self.timestamp.isoformat(),
            "exchange": self.exchange,
            "sequence": self.sequence,
        }

    @property
    def best_bid(self) -> Optional[float]:
        """Best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        """Mid price from order book."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        """Spread from order book."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


@dataclass
class OHLCVBar:
    """OHLCV candlestick/bar data."""

    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime
    interval: str = "1m"  # 1m, 5m, 1h, 1d, etc
    exchange: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class NewsEvent:
    """News with sentiment analysis."""

    title: str
    content: str
    source: str
    timestamp: datetime
    symbols: List[str] = field(default_factory=list)
    sentiment: float = 0.0  # -1.0 to 1.0
    confidence: float = 0.0  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class MacroEvent:
    """Macro economic event."""

    event: str
    country: str
    value: float
    forecast: Optional[float] = None
    previous: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    importance: str = "medium"  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


# ============================================================================
# Universal Wrapper
# ============================================================================


@dataclass
class MarketUpdate:
    """
    Universal wrapper for all market events.

    Routes events through the system with sequence tracking and timing.
    """

    data_type: DataType
    payload: Any  # PriceTick, OrderBookSnapshot, etc
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sequence_number: int = 0
    source: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        payload_dict = None
        if hasattr(self.payload, "to_dict"):
            payload_dict = self.payload.to_dict()
        elif isinstance(self.payload, dict):
            payload_dict = self.payload

        return {
            "data_type": self.data_type.value,
            "payload": payload_dict,
            "timestamp": self.timestamp.isoformat(),
            "sequence_number": self.sequence_number,
            "source": self.source,
        }


# ============================================================================
# Type Hints
# ============================================================================

MarketEventType = (
    PriceTick | OrderBookSnapshot | OHLCVBar | NewsEvent | MacroEvent | Dict[str, Any]
)
