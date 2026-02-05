"""
ZeroMQ (ZMQ) Real-Time Connector

Connects to market data providers using ZeroMQ:
  - Crypto exchange feeds (Binance, Coinbase, etc.)
  - Custom data sources
  - JSON message format
  - Pub/Sub pattern for streaming
  - Fault-tolerant packet reordering
  - Heartbeat and reconnection

Supports multiple feed types:
  - Ticker: Latest price/volume
  - Depth: Order book updates
  - Trades: Individual trade events
  - Custom: Any JSON structure

Author: Trading Stockfish Development Team
Date: January 19, 2026
"""

import logging
import json
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any
from collections import defaultdict
from enum import Enum

from realtime.exchange_base_connector import (
    BaseConnector, ConnectorStatus, OrderSide, OrderType, OrderStatus, Order
)
from realtime.data_models import (
    MarketUpdate, DataType, PriceTick, OrderBookSnapshot, OrderBookLevel
)


logger = logging.getLogger(__name__)


class ZMQFeedType(Enum):
    """ZMQ feed types."""
    TICKER = "ticker"
    DEPTH = "depth"
    TRADES = "trades"
    CUSTOM = "custom"


class ZeroMQConnector(BaseConnector):
    """
    ZeroMQ real-time connector for crypto and custom feeds.
    
    Features:
      - Pub/Sub pattern for streaming data
      - JSON message parsing
      - Packet sequence tracking
      - Out-of-order message handling
      - Automatic reconnection
      - Heartbeat monitoring
    
    Message format:
        {
            "type": "ticker|depth|trade|custom",
            "symbol": "BTCUSD",
            "sequence": 12345,
            "data": {...},
            "timestamp": "2026-01-19T10:30:45.123Z"
        }
    """
    
    def __init__(self, router=None, endpoints: List[str] = None, 
                 feed_type: ZMQFeedType = ZMQFeedType.TICKER):
        """
        Initialize ZMQ connector.
        
        Args:
            router: DataFeedRouter instance
            endpoints: List of ZMQ endpoints (e.g., ["tcp://localhost:5555"])
            feed_type: Type of feed (ticker, depth, trades, custom)
        """
        super().__init__('zmq', router=router)
        
        self.endpoints = endpoints or ["tcp://127.0.0.1:5555"]
        self.feed_type = feed_type
        self.zmq_context = None
        self.zmq_socket = None
        
        # Message buffering for reordering
        self.message_buffer = defaultdict(dict)  # symbol -> {seq: msg}
        self.expected_sequence = defaultdict(int)  # symbol -> next expected seq
        self.buffer_timeout = 5  # seconds
        
        # Heartbeat
        self.last_heartbeat = datetime.now(timezone.utc)
        self.heartbeat_timeout = 30
        
        # Thread management
        self.reader_thread = None
        self.reorder_thread = None
        self.running = False
        
        logger.info(f"Initialized ZMQ connector ({endpoints}, {feed_type.value})")
    
    # ============================================================================
    # Lifecycle Methods
    # ============================================================================
    
    def connect(self) -> bool:
        """Connect to ZMQ endpoints."""
        try:
            self.set_status(ConnectorStatus.CONNECTING)
            
            # Import ZMQ
            try:
                import zmq
                self.zmq = zmq
            except ImportError:
                logger.error("zmq not installed. Install with: pip install pyzmq")
                self.set_status(ConnectorStatus.ERROR)
                return False
            
            # Create context and socket
            self.zmq_context = self.zmq.Context()
            self.zmq_socket = self.zmq_context.socket(self.zmq.SUB)
            self.zmq_socket.setsockopt(self.zmq.RCVTIMEO, 5000)  # 5s timeout
            
            # Subscribe to all topics
            self.zmq_socket.setsockopt_string(self.zmq.SUBSCRIBE, "")
            
            # Connect to endpoints
            for endpoint in self.endpoints:
                self.zmq_socket.connect(endpoint)
                logger.info(f"ZMQ: Connected to {endpoint}")
            
            # Start reader threads
            self.running = True
            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.reorder_thread = threading.Thread(target=self._reorder_loop, daemon=True)
            self.reader_thread.start()
            self.reorder_thread.start()
            
            time.sleep(1)
            
            self.set_status(ConnectorStatus.CONNECTED)
                self.stats['connected_at'] = datetime.now(timezone.utc)
            logger.info(f"ZMQ: Connected ({self.feed_type.value})")
            
            return True
        
        except Exception as e:
            logger.error(f"ZMQ: Connection error: {str(e)}")
            self.handle_connection_error(e)
            return False
    
    def disconnect(self):
        """Close ZMQ connection."""
        try:
            self.running = False
            
            if self.zmq_socket:
                self.zmq_socket.close()
                self.zmq_socket = None
            
            if self.zmq_context:
                self.zmq_context.term()
                self.zmq_context = None
            
            # Wait for threads
            if self.reader_thread:
                self.reader_thread.join(timeout=2)
            if self.reorder_thread:
                self.reorder_thread.join(timeout=2)
            
            self.set_status(ConnectorStatus.DISCONNECTED)
            logger.info("ZMQ: Disconnected")
        
        except Exception as e:
            logger.error(f"ZMQ: Disconnect error: {str(e)}")
    
    def is_connected_check(self) -> bool:
        """Check if ZMQ socket is active."""
        return self.zmq_socket is not None and self.is_connected
    
    # ============================================================================
    # Message Reception
    # ============================================================================
    
    def _reader_loop(self):
        """Receive ZMQ messages."""
        while self.running:
            try:
                if not self.zmq_socket:
                    time.sleep(1)
                    continue
                
                # Receive message
                msg_bytes = self.zmq_socket.recv()
                if not msg_bytes:
                    continue
                
                # Parse JSON
                try:
                    msg_data = json.loads(msg_bytes.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"ZMQ: Invalid JSON: {str(e)}")
                    self.stats['data_errors'] += 1
                    continue
                
                # Process message
                self._process_zmq_message(msg_data)
                self.last_heartbeat = datetime.now(timezone.utc)
                self._record_heartbeat(self.last_heartbeat)
            
            except self.zmq.error.Again:
                # Timeout - check connection
                if (datetime.now(timezone.utc) - self.last_heartbeat).total_seconds() > self.heartbeat_timeout:
                    logger.warning("ZMQ: Heartbeat timeout")
                    self.set_status(ConnectorStatus.RECONNECTING)
                    self.disconnect()
                    time.sleep(2)
                    self.connect()
            
            except Exception as e:
                logger.error(f"ZMQ: Reader error: {str(e)}")
                self.stats['connection_errors'] += 1
                self.handle_connection_error(e)
                time.sleep(1)
    
    def _process_zmq_message(self, msg: Dict[str, Any]):
        """Process received ZMQ message."""
        try:
            msg_type = msg.get('type', 'custom')
            symbol = msg.get('symbol', 'UNKNOWN')
            sequence = msg.get('sequence', 0)
            
            logger.debug(f"ZMQ: {msg_type} {symbol} seq={sequence}")
            
            # Buffer by sequence for reordering
            if sequence > 0:
                if symbol not in self.message_buffer:
                    self.expected_sequence[symbol] = sequence
                self.message_buffer[symbol][sequence] = msg
            else:
                # No sequence, process immediately
                self._normalize_and_push(msg, msg_type, symbol)
        
        except Exception as e:
            logger.error(f"ZMQ: Message processing error: {str(e)}")
            self.stats['data_errors'] += 1
    
    def _reorder_loop(self):
        """Process buffered messages in sequence order."""
        while self.running:
            try:
                for symbol in list(self.message_buffer.keys()):
                    buffer = self.message_buffer[symbol]
                    expected_seq = self.expected_sequence[symbol]
                    
                    # Process consecutive messages
                    while expected_seq in buffer:
                        msg = buffer[expected_seq]
                        msg_type = msg.get('type', 'custom')
                        
                        self._normalize_and_push(msg, msg_type, symbol)
                        
                        del buffer[expected_seq]
                        expected_seq += 1
                        self.expected_sequence[symbol] = expected_seq
                    
                    # Clean up old buffered messages
                    if buffer:
                        min_buffered_seq = min(buffer.keys())
                        if min_buffered_seq < expected_seq - 100:  # Gap too large
                            logger.warning(f"ZMQ: Packet loss detected for {symbol}")
                            # Clear stale buffer
                            for seq in list(buffer.keys()):
                                if seq < expected_seq:
                                    del buffer[seq]
                    
                    # Timeout cleanup
                    if not buffer:
                        del self.message_buffer[symbol]
                
                time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"ZMQ: Reorder error: {str(e)}")
                time.sleep(1)
    
    def _normalize_and_push(self, msg: Dict[str, Any], msg_type: str, symbol: str):
        """Normalize ZMQ message to MarketUpdate."""
        try:
            data = msg.get('data', {})
            timestamp = msg.get('timestamp')
            if timestamp and isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    timestamp = datetime.now(timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
            
            if msg_type == 'ticker':
                update = self._normalize_ticker(symbol, data, timestamp)
            elif msg_type == 'depth':
                update = self._normalize_depth(symbol, data, timestamp)
            elif msg_type == 'trades':
                update = self._normalize_trade(symbol, data, timestamp)
            else:  # custom
                update = None
            
            if update:
                self.push_update(update)
                self.stats['updates_received'] += 1
        
        except Exception as e:
            logger.error(f"ZMQ: Normalization error: {str(e)}")
            self.stats['data_errors'] += 1
    
    # ============================================================================
    # Data Normalization
    # ============================================================================
    
    def _normalize_ticker(self, symbol: str, data: Dict, timestamp: datetime) -> Optional[MarketUpdate]:
        """Normalize ticker message."""
        try:
            tick = PriceTick(
                symbol=symbol,
                bid=float(data.get('bid', 0)),
                ask=float(data.get('ask', 0)),
                last=float(data.get('last', 0)),
                bid_volume=float(data.get('bid_volume', 0)),
                ask_volume=float(data.get('ask_volume', 0)),
                last_volume=float(data.get('volume', 0)),
                timestamp=timestamp,
                exchange='ZMQ'
            )
            
            return MarketUpdate(
                data_type=DataType.PRICE_TICK,
                payload=tick,
                timestamp=timestamp,
                sequence_number=self.stats['updates_normalized']
            )
        except Exception as e:
            logger.error(f"ZMQ: Ticker normalization error: {str(e)}")
            return None
    
    def _normalize_depth(self, symbol: str, data: Dict, timestamp: datetime) -> Optional[MarketUpdate]:
        """Normalize depth (orderbook) message."""
        try:
            bids_data = data.get('bids', [])
            asks_data = data.get('asks', [])
            
            bids = [OrderBookLevel(price=float(b[0]), quantity=float(b[1])) for b in bids_data[:10]]
            asks = [OrderBookLevel(price=float(a[0]), quantity=float(a[1])) for a in asks_data[:10]]
            
            snapshot = OrderBookSnapshot(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=timestamp,
                exchange='ZMQ'
            )
            
            return MarketUpdate(
                data_type=DataType.ORDERBOOK_SNAPSHOT,
                payload=snapshot,
                timestamp=timestamp,
                sequence_number=self.stats['updates_normalized']
            )
        except Exception as e:
            logger.error(f"ZMQ: Depth normalization error: {str(e)}")
            return None
    
    def _normalize_trade(self, symbol: str, data: Dict, timestamp: datetime) -> Optional[MarketUpdate]:
        """Normalize trade message."""
        try:
            # Create synthetic PriceTick from trade
            price = float(data.get('price', 0))
            volume = float(data.get('quantity', 0))
            side = data.get('side', 'unknown')
            
            tick = PriceTick(
                symbol=symbol,
                bid=price,
                ask=price,
                last=price,
                last_volume=volume,
                timestamp=timestamp,
                exchange='ZMQ'
            )
            
            return MarketUpdate(
                data_type=DataType.PRICE_TICK,
                payload=tick,
                timestamp=timestamp,
                sequence_number=self.stats['updates_normalized']
            )
        except Exception as e:
            logger.error(f"ZMQ: Trade normalization error: {str(e)}")
            return None
    
    # ============================================================================
    # Subscription Methods
    # ============================================================================
    
    def subscribe_price(self, symbol: str) -> bool:
        """Subscribe to price ticker."""
        self.subscribed_symbols.add(symbol)
        logger.info(f"ZMQ: Subscribed to {symbol} (feed={self.feed_type.value})")
        return True
    
    def subscribe_orderbook(self, symbol: str, depth: int = 10) -> bool:
        """Subscribe to order book."""
        self.subscribed_symbols.add(symbol)
        logger.info(f"ZMQ: Subscribed to orderbook {symbol}")
        return True
    
    def subscribe_news(self) -> bool:
        """Subscribe to news."""
        logger.info("ZMQ: News subscription requested")
        return True
    
    def subscribe_macro(self) -> bool:
        """Subscribe to macro."""
        logger.info("ZMQ: Macro subscription requested")
        return True
    
    def unsubscribe(self, symbol: str):
        """Unsubscribe from symbol."""
        self.subscribed_symbols.discard(symbol)
        if symbol in self.message_buffer:
            del self.message_buffer[symbol]
        if symbol in self.expected_sequence:
            del self.expected_sequence[symbol]
        logger.info(f"ZMQ: Unsubscribed from {symbol}")
    
    # ============================================================================
    # Data Normalization Callbacks (required by base class)
    # ============================================================================
    
    def on_price_tick(self, exchange_data: Dict) -> Optional[MarketUpdate]:
        """Handled in normalization loop."""
        return None
    
    def on_orderbook(self, exchange_data: Dict) -> Optional[MarketUpdate]:
        """Handled in normalization loop."""
        return None
    
    def on_news(self, exchange_data: Dict) -> Optional[MarketUpdate]:
        """Placeholder."""
        return None
    
    def on_macro(self, exchange_data: Dict) -> Optional[MarketUpdate]:
        """Placeholder."""
        return None
    
    # ============================================================================
    # Order Execution (Placeholder - ZMQ typically data-only)
    # ============================================================================
    
    def send_order(self, order: Order) -> Optional[str]:
        """ZMQ connector is read-only for data."""
        logger.warning("ZMQ: Connector is read-only (data feeds only)")
        order.status = OrderStatus.REJECTED
        order.rejection_reason = "Read-only connector"
        self._record_order_rejection()
        return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Not supported."""
        return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Not supported."""
        return None
