"""
Abstract Base Connector - Interface for all exchange connectors

Defines universal interface for:
  - Interactive Brokers (IBKR)
  - FIX Protocol
  - ZeroMQ (ZMQ)
  - Custom implementations

All connectors inherit from BaseConnector and implement 12 abstract methods.

Author: Trading Stockfish Development Team
Date: January 19, 2026
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set

from realtime.data_models import MarketUpdate

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class ConnectorStatus(Enum):
    """Connector connection status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class OrderSide(Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


# ============================================================================
# Order Class
# ============================================================================


class Order:
    """Universal order representation."""

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
    ):
        """
        Create order.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            order_type: MARKET, LIMIT, STOP, STOP_LIMIT
            price: Limit/stop price (required for LIMIT/STOP orders)
        """
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.price = price

        # Status tracking
        self.order_id: Optional[str] = None
        self.connector_order_id: Optional[str] = None
        self.status = OrderStatus.PENDING

        # Execution tracking
        self.filled_quantity = 0.0
        self.avg_fill_price = 0.0

        # Timing
        self.submit_time: Optional[datetime] = None
        self.fill_time: Optional[datetime] = None

        # Error handling
        self.rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "price": self.price,
            "order_id": self.order_id,
            "connector_order_id": self.connector_order_id,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "submit_time": self.submit_time.isoformat() if self.submit_time else None,
            "fill_time": self.fill_time.isoformat() if self.fill_time else None,
            "rejection_reason": self.rejection_reason,
        }


# ============================================================================
# Abstract Base Connector
# ============================================================================


class BaseConnector(ABC):
    """
    Abstract base class for all exchange connectors.

    Defines 12 abstract methods that all connectors must implement:

    Lifecycle (3):
      - connect(): Establish connection to exchange
      - disconnect(): Close connection
      - is_connected_check(): Verify connection status

    Subscriptions (5):
      - subscribe_price(symbol): Subscribe to price ticks
      - subscribe_orderbook(symbol, depth): Subscribe to order book
      - subscribe_news(): Subscribe to news events
      - subscribe_macro(): Subscribe to macro events
      - unsubscribe(symbol): Unsubscribe from symbol

    Data Normalization (4):
      - on_price_tick(data): Normalize price data to MarketUpdate
      - on_orderbook(data): Normalize orderbook to MarketUpdate
      - on_news(data): Normalize news to MarketUpdate
      - on_macro(data): Normalize macro to MarketUpdate

    Order Execution (3):
      - send_order(order): Submit order
      - cancel_order(order_id): Cancel order
      - get_order_status(order_id): Get order status
    """

    def __init__(self, name: str, router=None):
        """
        Initialize base connector.

        Args:
            name: Unique connector identifier (ibkr, fix, zmq, etc)
            router: DataFeedRouter to push normalized updates to
        """
        self.name = name
        self.router = router

        # Status
        self._status = ConnectorStatus.DISCONNECTED
        self.is_connected = False

        # Subscriptions
        self.subscribed_symbols: Set[str] = set()

        # Orders
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: Dict[str, Order] = {}

        # Statistics
        self.stats: Dict = {
            "connected_at": None,
            "last_update_time": None,
            "updates_received": 0,
            "updates_normalized": 0,
            "updates_by_type": {
                "price_tick": 0,
                "orderbook": 0,
                "news": 0,
                "macro": 0,
                "other": 0,
            },
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "order_errors": 0,
            "data_errors": 0,
            "connection_errors": 0,
            "reconnection_attempts": 0,
        }

        logger.info(f"Initialized {name} connector")

    # ============================================================================
    # Lifecycle (Abstract)
    # ============================================================================

    @abstractmethod
    def connect(self) -> bool:
        """Connect to exchange. Return True on success."""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from exchange."""
        pass

    @abstractmethod
    def is_connected_check(self) -> bool:
        """Check actual connection status."""
        pass

    # ============================================================================
    # Subscriptions (Abstract)
    # ============================================================================

    @abstractmethod
    def subscribe_price(self, symbol: str) -> bool:
        """Subscribe to real-time price ticks."""
        pass

    @abstractmethod
    def subscribe_orderbook(self, symbol: str, depth: int = 10) -> bool:
        """Subscribe to order book updates."""
        pass

    @abstractmethod
    def subscribe_news(self) -> bool:
        """Subscribe to news events."""
        pass

    @abstractmethod
    def subscribe_macro(self) -> bool:
        """Subscribe to macro events."""
        pass

    @abstractmethod
    def unsubscribe(self, symbol: str):
        """Unsubscribe from symbol."""
        pass

    # ============================================================================
    # Data Normalization (Abstract)
    # ============================================================================

    @abstractmethod
    def on_price_tick(self, exchange_data: Dict) -> Optional[MarketUpdate]:
        """Normalize price tick to MarketUpdate."""
        pass

    @abstractmethod
    def on_orderbook(self, exchange_data: Dict) -> Optional[MarketUpdate]:
        """Normalize orderbook to MarketUpdate."""
        pass

    @abstractmethod
    def on_news(self, exchange_data: Dict) -> Optional[MarketUpdate]:
        """Normalize news to MarketUpdate."""
        pass

    @abstractmethod
    def on_macro(self, exchange_data: Dict) -> Optional[MarketUpdate]:
        """Normalize macro to MarketUpdate."""
        pass

    # ============================================================================
    # Order Execution (Abstract)
    # ============================================================================

    @abstractmethod
    def send_order(self, order: Order) -> Optional[str]:
        """Submit order. Return order_id on success."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order. Return True on success."""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get current order status."""
        pass

    # ============================================================================
    # Shared Implementations
    # ============================================================================

    def push_update(self, update: MarketUpdate):
        """Push normalized market update to router."""
        try:
            self.stats["last_update_time"] = datetime.now(datetime.timezone.utc)

            # Route to DataFeedRouter if available
            if self.router:
                self.router.route_update(update)

            # Track by type
            if "price" in update.data_type.value:
                self.stats["updates_by_type"]["price_tick"] += 1
            elif "orderbook" in update.data_type.value:
                self.stats["updates_by_type"]["orderbook"] += 1
            elif "news" in update.data_type.value:
                self.stats["updates_by_type"]["news"] += 1
            elif "macro" in update.data_type.value:
                self.stats["updates_by_type"]["macro"] += 1
            else:
                self.stats["updates_by_type"]["other"] += 1

            self.stats["updates_normalized"] += 1

        except Exception as e:
            logger.error(f"{self.name}: Push update error: {str(e)}")
            self.stats["data_errors"] += 1

    def set_status(self, status: ConnectorStatus):
        """Update connection status."""
        self._status = status
        self.is_connected = status == ConnectorStatus.CONNECTED
        logger.debug(f"{self.name}: Status -> {status.value}")

    def get_status(self) -> str:
        """Get brief status."""
        return self._status.value

    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        return self.stats.copy()

    def log_stats(self):
        """Log statistics summary."""
        s = self.stats
        logger.info(f"\n{self.name} Statistics:")
        logger.info(f"  Connected: {self.is_connected}")
        logger.info(
            f"  Updates: {s['updates_received']} received, {s['updates_normalized']} normalized"
        )
        logger.info(
            f"  Orders: {s['orders_submitted']} submitted, {s['orders_filled']} filled"
        )
        logger.info(
            f"  Errors: data={s['data_errors']}, connection={s['connection_errors']}"
        )

    def handle_connection_error(self, error: Exception):
        """Handle connection errors."""
        self.stats["connection_errors"] += 1
        self.stats["reconnection_attempts"] += 1
        self.set_status(ConnectorStatus.ERROR)
        logger.error(f"{self.name}: Connection error: {str(error)}")
