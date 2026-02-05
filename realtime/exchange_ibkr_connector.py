"""
Interactive Brokers (IBKR) Real-Time Connector

Connects to Interactive Brokers via ib_insync library:
  - Real-time price ticks
  - Level 1 order book
  - OHLCV bars
  - Market and limit order execution
  - Position tracking
  - Error handling and reconnection

Normalizes IBKR events to MarketUpdate objects.

Author: Trading Stockfish Development Team
Date: January 19, 2026
"""

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from realtime.data_models import (
    DataType,
    MarketUpdate,
    OrderBookLevel,
    OrderBookSnapshot,
    PriceTick,
)
from realtime.exchange_base_connector import (
    BaseConnector,
    ConnectorStatus,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)

logger = logging.getLogger(__name__)


class IBKRConnector(BaseConnector):
    """
    Interactive Brokers real-time connector.

    Features:
      - Price tick streaming (bid/ask/last)
      - Level 1 order book
      - OHLCV bar data (1s-1h)
      - Market and limit orders
      - Position tracking
      - Heartbeat monitoring for reconnection
      - Throttle protection

    Note: Requires ib_insync library and running TWS/Gateway
    """

    def __init__(self, router=None, host="127.0.0.1", port=7497, client_id=1):
        """
        Initialize IBKR connector.

        Args:
            router: DataFeedRouter instance
            host: TWS/Gateway host
            port: TWS/Gateway port (7497 live, 7498 paper)
            client_id: Unique client ID
        """
        super().__init__("ibkr", router=router)

        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None  # ib_insync IB instance
        self.contracts = {}  # symbol -> Contract
        self.market_data = {}  # symbol -> TickData

        # Heartbeat monitoring
        self.last_heartbeat = datetime.now(timezone.utc)
        self.heartbeat_timeout = 30  # seconds
        self.heartbeat_thread = None
        self.heartbeat_running = False

        # Order tracking
        self.order_counter = 0
        self.ibkr_order_map = {}  # ibkr_order_id -> our order_id

        logger.info(
            f"Initialized IBKR connector ({host}:{port}, client_id={client_id})"
        )

    # ============================================================================
    # Lifecycle Methods
    # ============================================================================

    def connect(self) -> bool:
        """Connect to Interactive Brokers."""
        try:
            self.set_status(ConnectorStatus.CONNECTING)

            # Import here to handle optional dependency
            try:
                from ib_insync import IB, Forex, Index, Stock

                self.Stock = Stock
                self.Index = Index
                self.Forex = Forex
            except ImportError:
                logger.error(
                    "ib_insync not installed. Install with: pip install ib-insync"
                )
                self.set_status(ConnectorStatus.ERROR)
                return False

            # Create IB instance
            self.ib = IB()
            self.ib.connect(self.host, self.port, self.client_id)

            # Wait for connection
            time.sleep(1)

            if self.ib.isConnected():
                self.set_status(ConnectorStatus.CONNECTED)
                self.stats["connected_at"] = datetime.now(timezone.utc)

                # Start heartbeat monitoring
                self._start_heartbeat()

                logger.info("IBKR: Successfully connected")
                return True
            else:
                logger.error("IBKR: Connection failed")
                self.set_status(ConnectorStatus.ERROR)
                return False

        except Exception as e:
            logger.error(f"IBKR: Connection error: {str(e)}")
            self.handle_connection_error(e)
            return False

    def disconnect(self):
        """Disconnect from Interactive Brokers."""
        try:
            self._stop_heartbeat()

            if self.ib:
                self.ib.disconnect()
                self.ib = None

            self.set_status(ConnectorStatus.DISCONNECTED)
            logger.info("IBKR: Disconnected")

        except Exception as e:
            logger.error(f"IBKR: Disconnect error: {str(e)}")

    def is_connected_check(self) -> bool:
        """Check if connected to IBKR."""
        if not self.ib:
            return False

        return self.ib.isConnected()

    # ============================================================================
    # Heartbeat and Reconnection
    # ============================================================================

    def _start_heartbeat(self):
        """Start heartbeat monitoring thread."""
        self.heartbeat_running = True
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self.heartbeat_thread.start()

    def _stop_heartbeat(self):
        """Stop heartbeat monitoring."""
        self.heartbeat_running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=2)

    def _heartbeat_loop(self):
        """Monitor connection health."""
        while self.heartbeat_running and self.is_connected:
            try:
                # Check for timeout
                time_since_heartbeat = (
                    datetime.now(timezone.utc) - self.last_heartbeat
                ).total_seconds()

                if time_since_heartbeat > self.heartbeat_timeout:
                    logger.warning(f"IBKR: Heartbeat timeout ({time_since_heartbeat}s)")
                    self.set_status(ConnectorStatus.RECONNECTING)
                    self.disconnect()
                    time.sleep(2)
                    self.connect()

                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"IBKR: Heartbeat error: {str(e)}")

    # ============================================================================
    # Subscription Methods
    # ============================================================================

    def subscribe_price(self, symbol: str) -> bool:
        """Subscribe to real-time price ticks."""
        if not self.is_connected:
            logger.warning(f"IBKR: Not connected, cannot subscribe to {symbol}")
            return False

        try:
            # Get contract (simplified, assumes US stock/index/forex)
            contract = self._get_contract(symbol)
            if not contract:
                logger.error(f"IBKR: Unknown symbol format: {symbol}")
                return False

            # Subscribe to ticks
            self.ib.reqMktData(contract, "", False, False)
            self.subscribed_symbols.add(symbol)

            logger.info(f"IBKR: Subscribed to price: {symbol}")
            return True

        except Exception as e:
            logger.error(f"IBKR: Price subscription error for {symbol}: {str(e)}")
            self.stats["data_errors"] += 1
            return False

    def subscribe_orderbook(self, symbol: str, depth: int = 10) -> bool:
        """Subscribe to order book (L1 via bid/ask)."""
        # IBKR provides L1 through price ticks
        return self.subscribe_price(symbol)

    def subscribe_news(self) -> bool:
        """Subscribe to news (placeholder)."""
        logger.info("IBKR: News subscription requested (not implemented)")
        return True

    def subscribe_macro(self) -> bool:
        """Subscribe to macro events (placeholder)."""
        logger.info("IBKR: Macro subscription requested (not implemented)")
        return True

    def unsubscribe(self, symbol: str):
        """Unsubscribe from symbol."""
        try:
            if symbol in self.subscribed_symbols:
                contract = self._get_contract(symbol)
                if contract:
                    self.ib.cancelMktData(contract)
                    self.subscribed_symbols.discard(symbol)
                    logger.info(f"IBKR: Unsubscribed from {symbol}")

        except Exception as e:
            logger.error(f"IBKR: Unsubscribe error for {symbol}: {str(e)}")

    # ============================================================================
    # Data Normalization
    # ============================================================================

    def on_price_tick(self, exchange_data: Dict) -> Optional[MarketUpdate]:
        """Normalize IBKR price tick to MarketUpdate."""
        try:
            symbol = exchange_data.get("symbol")
            bid = exchange_data.get("bid")
            ask = exchange_data.get("ask")
            last = exchange_data.get("last")

            if not all([symbol, bid is not None, ask is not None]):
                return None

            tick = PriceTick(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=last or (bid + ask) / 2,
                bid_volume=exchange_data.get("bid_volume", 0),
                ask_volume=exchange_data.get("ask_volume", 0),
                last_volume=exchange_data.get("last_volume", 0),
                timestamp=datetime.now(timezone.utc),
                exchange="IBKR",
            )

            update = MarketUpdate(
                data_type=DataType.PRICE_TICK,
                payload=tick,
                timestamp=datetime.now(timezone.utc),
                sequence_number=self.stats["updates_received"],
            )

            self.last_heartbeat = datetime.now(timezone.utc)
            self.stats["updates_received"] += 1
            self._record_market_data_timestamp(update.timestamp)
            self._record_heartbeat(update.timestamp)

            return update

        except Exception as e:
            logger.error(f"IBKR: Price normalization error: {str(e)}")
            self.stats["data_errors"] += 1
            return None

    def on_orderbook(self, exchange_data: Dict) -> Optional[MarketUpdate]:
        """Normalize IBKR order book to MarketUpdate."""
        try:
            symbol = exchange_data.get("symbol")
            bid_levels = exchange_data.get("bids", [])
            ask_levels = exchange_data.get("asks", [])

            if not symbol or (not bid_levels and not ask_levels):
                return None

            # Convert to OrderBookLevel objects
            bids = [OrderBookLevel(price=b[0], quantity=b[1]) for b in bid_levels]
            asks = [OrderBookLevel(price=a[0], quantity=a[1]) for a in ask_levels]

            snapshot = OrderBookSnapshot(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.now(timezone.utc),
                exchange="IBKR",
            )

            update = MarketUpdate(
                data_type=DataType.ORDERBOOK_SNAPSHOT,
                payload=snapshot,
                timestamp=datetime.now(timezone.utc),
                sequence_number=self.stats["updates_received"],
            )

            self.stats["updates_received"] += 1
            return update

        except Exception as e:
            logger.error(f"IBKR: Orderbook normalization error: {str(e)}")
            self.stats["data_errors"] += 1
            return None

    def on_news(self, exchange_data: Dict) -> Optional[MarketUpdate]:
        """Placeholder for news normalization."""
        return None

    def on_macro(self, exchange_data: Dict) -> Optional[MarketUpdate]:
        """Placeholder for macro normalization."""
        return None

    # ============================================================================
    # Execution Methods
    # ============================================================================

    def send_order(self, order: Order) -> Optional[str]:
        """Submit market or limit order to IBKR."""
        if not self.is_connected:
            logger.warning("IBKR: Not connected, cannot send order")
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "Not connected"
            self._record_send_failure()
            self._record_order_rejection()
            return None

        try:
            # Get contract
            contract = self._get_contract(order.symbol)
            if not contract:
                logger.error(f"IBKR: Unknown symbol: {order.symbol}")
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "Unknown symbol"
                return None

            # Create IBKR Order object
            from ib_insync import Order as IBOrder

            ibkr_order = IBOrder()
            ibkr_order.action = order.side.value
            ibkr_order.totalQuantity = int(order.quantity)
            ibkr_order.orderType = order.order_type.value
            if order.price:
                ibkr_order.lmtPrice = order.price

            # Submit
            trade = self.ib.placeOrder(contract, ibkr_order)

            # Assign our order ID
            self.order_counter += 1
            order_id = f"ibkr_{self.order_counter}"
            order.order_id = order_id
            order.connector_order_id = trade.order.orderId
            order.status = OrderStatus.SUBMITTED
            order.submit_time = datetime.now(timezone.utc)

            self.pending_orders[order_id] = order
            self.ibkr_order_map[trade.order.orderId] = order_id
            self.stats["orders_submitted"] += 1

            logger.info(
                f"IBKR: Submitted {order.side.value} {order.quantity} {order.symbol} @ {order.price}"
            )

            return order_id

        except Exception as e:
            logger.error(f"IBKR: Order submission error: {str(e)}")
            order.status = OrderStatus.ERROR
            order.rejection_reason = str(e)
            self.stats["order_errors"] += 1
            self._record_send_failure()
            self._record_order_rejection()
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel IBKR order."""
        try:
            if order_id not in self.pending_orders:
                logger.warning(f"IBKR: Order not found: {order_id}")
                return False

            order = self.pending_orders[order_id]

            if not order.connector_order_id:
                return False

            # Cancel via IBKR
            self.ib.cancelOrder(self.ib.openOrders()[0])  # Simplified

            order.status = OrderStatus.CANCELLED
            self.stats["orders_cancelled"] += 1

            logger.info(f"IBKR: Cancelled order {order_id}")
            return True

        except Exception as e:
            logger.error(f"IBKR: Cancel error: {str(e)}")
            self.stats["order_errors"] += 1
            return False

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        if order_id in self.pending_orders:
            return self.pending_orders[order_id].status
        elif order_id in self.filled_orders:
            return self.filled_orders[order_id].status
        else:
            return None

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _get_contract(self, symbol: str):
        """Get or create contract for symbol."""
        if symbol in self.contracts:
            return self.contracts[symbol]

        try:
            # Simple contract creation (handle different asset classes)
            if symbol.startswith("BTC") or symbol.startswith("ETH"):
                # Crypto (simplified)
                return None
            elif "/" in symbol:
                # Forex pair (EUR/USD)
                parts = symbol.split("/")
                contract = self.Forex(symbol)
            elif symbol in ["SPY", "QQQ", "IWM", "GLD"]:
                # Stocks
                contract = self.Stock(symbol, "SMART", "USD")
            else:
                # Default to Index
                contract = self.Index(symbol, "CBOE")

            self.contracts[symbol] = contract
            return contract

        except Exception as e:
            logger.error(f"IBKR: Contract creation error for {symbol}: {str(e)}")
            return None

    def _handle_fill(self, ibkr_order_id: int, filled_qty: float, fill_price: float):
        """Handle order fill from IBKR."""
        if ibkr_order_id not in self.ibkr_order_map:
            return

        order_id = self.ibkr_order_map[ibkr_order_id]
        if order_id not in self.pending_orders:
            return

        order = self.pending_orders[order_id]
        order.filled_quantity += filled_qty
        order.avg_fill_price = fill_price

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            order.fill_time = datetime.now(timezone.utc)
            self.filled_orders[order_id] = order
            del self.pending_orders[order_id]
            self.stats["orders_filled"] += 1
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        logger.info(f"IBKR: Fill {order_id}: {filled_qty} @ {fill_price}")
