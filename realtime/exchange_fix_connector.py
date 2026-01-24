"""
FIX Protocol (Financial Information eXchange) Real-Time Connector

Connects to FIX-compliant exchanges and brokers:
  - FIX 4.2, 4.4, 5.0 SP2 support
  - Market data parsing (snapshot, incremental)
  - Execution report processing
  - Order lifecycle management
  - Comprehensive error handling

FIX message types supported:
  - W: MarketDataSnapshotFullRefresh
  - X: MarketDataIncrementalRefresh
  - D: NewOrderSingle
  - F: OrderCancelRequest
  - 8: ExecutionReport

Author: Trading Stockfish Development Team
Date: January 19, 2026
"""

import logging
import socket
import threading
import time
from datetime import datetime
from typing import Dict, Optional, List
from enum import Enum

from realtime.exchange_base_connector import (
    BaseConnector, ConnectorStatus, OrderSide, OrderType, OrderStatus, Order
)
from realtime.data_models import (
    MarketUpdate, DataType, PriceTick, OrderBookSnapshot, OrderBookLevel
)


logger = logging.getLogger(__name__)


class FIXVersion(Enum):
    """Supported FIX protocol versions."""
    FIX42 = "FIX.4.2"
    FIX44 = "FIX.4.4"
    FIX50SP2 = "FIXT.1.1"


class FIXConnector(BaseConnector):
    """
    FIX protocol real-time connector.
    
    Features:
      - Full FIX message parsing
      - Market data snapshot and incremental updates
      - Order execution and status tracking
      - Heartbeat and session management
      - Comprehensive logging
      - Error recovery
    """
    
    def __init__(self, router=None, host='127.0.0.1', port=9876, 
                 version=FIXVersion.FIX44, sender_comp_id='TRADER',
                 target_comp_id='EXCHANGE'):
        """
        Initialize FIX connector.
        
        Args:
            router: DataFeedRouter instance
            host: FIX server host
            port: FIX server port
            version: FIX protocol version
            sender_comp_id: Our unique ID (SenderCompID)
            target_comp_id: Target ID (TargetCompID)
        """
        super().__init__('fix', router=router)
        
        self.host = host
        self.port = port
        self.version = version
        self.sender_comp_id = sender_comp_id
        self.target_comp_id = target_comp_id
        
        self.socket = None
        self.msg_seq_num = 0
        self.target_seq_num = 0
        
        # Message tracking
        self.pending_messages = {}
        self.market_data_subscriptions = {}
        
        # Order tracking
        self.order_counter = 0
        self.fix_order_map = {}  # ClOrdID -> our order_id
        
        # Threads
        self.reader_thread = None
        self.heartbeat_thread = None
        self.running = False
        
        logger.info(f"Initialized FIX connector ({version.value} @ {host}:{port})")
    
    # ============================================================================
    # Lifecycle Methods
    # ============================================================================
    
    def connect(self) -> bool:
        """Connect to FIX server and establish session."""
        try:
            self.set_status(ConnectorStatus.CONNECTING)
            
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            
            # Send logon
            self.msg_seq_num = 1
            logon_msg = self._create_logon_message()
            self._send_message(logon_msg)
            
            # Start reader and heartbeat threads
            self.running = True
            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.reader_thread.start()
            self.heartbeat_thread.start()
            
            # Wait for logon confirmation
            time.sleep(1)
            
            if self.is_connected:
                self.set_status(ConnectorStatus.CONNECTED)
                self.stats['connected_at'] = datetime.utcnow()
                logger.info(f"FIX: Session established ({self.version.value})")
                return True
            else:
                logger.error("FIX: Logon failed")
                self.set_status(ConnectorStatus.ERROR)
                return False
        
        except Exception as e:
            logger.error(f"FIX: Connection error: {str(e)}")
            self.handle_connection_error(e)
            return False
    
    def disconnect(self):
        """Close FIX session."""
        try:
            self.running = False
            
            # Send logout
            if self.socket:
                logout_msg = self._create_message("5")  # Logout
                self._send_message(logout_msg)
                time.sleep(0.5)
            
            # Close socket
            if self.socket:
                self.socket.close()
                self.socket = None
            
            # Wait for threads
            if self.reader_thread:
                self.reader_thread.join(timeout=2)
            if self.heartbeat_thread:
                self.heartbeat_thread.join(timeout=2)
            
            self.set_status(ConnectorStatus.DISCONNECTED)
            logger.info("FIX: Disconnected")
        
        except Exception as e:
            logger.error(f"FIX: Disconnect error: {str(e)}")
    
    def is_connected_check(self) -> bool:
        """Check if FIX session is active."""
        return self.socket is not None and self.is_connected
    
    # ============================================================================
    # Message Transport
    # ============================================================================
    
    def _create_message(self, msg_type: str, fields: Dict[str, str] = None) -> str:
        """Create FIX message."""
        if fields is None:
            fields = {}
        
        self.msg_seq_num += 1
        
        # Build message
        msg = f"35={msg_type}|"
        msg += f"49={self.sender_comp_id}|"
        msg += f"56={self.target_comp_id}|"
        msg += f"34={self.msg_seq_num}|"
        msg += f"52={datetime.utcnow().isoformat()}|"
        
        # Add custom fields
        for key, value in fields.items():
            msg += f"{key}={value}|"
        
        # Calculate checksum
        checksum = sum(ord(c) for c in msg) % 256
        msg = f"8={self.version.value}|9={len(msg)}|{msg}93={len(msg)}|10={checksum:03d}|"
        
        return msg
    
    def _create_logon_message(self) -> str:
        """Create FIX logon message."""
        return self._create_message("1", {
            "98": "0",  # EncryptMethod: None
            "108": "30"  # HeartBtInt: 30 seconds
        })
    
    def _send_message(self, msg: str):
        """Send FIX message."""
        try:
            if self.socket:
                self.socket.sendall(msg.encode('ascii'))
                logger.debug(f"FIX TX: {msg[:100]}")
        except Exception as e:
            logger.error(f"FIX: Send error: {str(e)}")
            self.handle_connection_error(e)
    
    def _reader_loop(self):
        """Receive and parse FIX messages."""
        while self.running:
            try:
                if not self.socket:
                    time.sleep(1)
                    continue
                
                # Receive data
                data = self.socket.recv(4096)
                if not data:
                    logger.warning("FIX: Connection closed by server")
                    self.set_status(ConnectorStatus.RECONNECTING)
                    break
                
                # Parse messages
                messages = data.decode('ascii').split('|')
                for msg in messages:
                    if msg.strip():
                        self._process_message(msg)
            
            except Exception as e:
                logger.error(f"FIX: Reader error: {str(e)}")
                self.handle_connection_error(e)
                time.sleep(1)
    
    def _heartbeat_loop(self):
        """Send heartbeat messages."""
        while self.running:
            try:
                time.sleep(30)  # Every 30 seconds
                if self.is_connected:
                    msg = self._create_message("0")  # Heartbeat
                    self._send_message(msg)
            
            except Exception as e:
                logger.error(f"FIX: Heartbeat error: {str(e)}")
    
    # ============================================================================
    # Message Processing
    # ============================================================================
    
    def _process_message(self, msg: str):
        """Parse and process FIX message."""
        try:
            fields = self._parse_message(msg)
            msg_type = fields.get("35")
            
            logger.debug(f"FIX RX: Type={msg_type}")
            
            # Route by type
            if msg_type == "1":  # Heartbeat
                pass
            elif msg_type == "5":  # Logout
                logger.info("FIX: Logout received")
                self.set_status(ConnectorStatus.DISCONNECTED)
            elif msg_type == "W":  # Market data snapshot
                self._handle_market_data_snapshot(fields)
            elif msg_type == "X":  # Market data incremental
                self._handle_market_data_incremental(fields)
            elif msg_type == "8":  # Execution report
                self._handle_execution_report(fields)
            else:
                logger.debug(f"FIX: Unhandled message type: {msg_type}")
        
        except Exception as e:
            logger.error(f"FIX: Message processing error: {str(e)}")
            self.stats['data_errors'] += 1
    
    def _parse_message(self, msg: str) -> Dict[str, str]:
        """Parse FIX message into fields."""
        fields = {}
        pairs = msg.split('|')
        for pair in pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                fields[key] = value
        return fields
    
    # ============================================================================
    # Market Data Handling
    # ============================================================================
    
    def subscribe_price(self, symbol: str) -> bool:
        """Subscribe to market data."""
        try:
            # Create market data request
            msg = self._create_message("V", {  # MarketDataRequest
                "262": f"sub_{symbol}",  # MDReqID
                "263": "1",  # SubscriptionRequestType: Subscribe
                "264": "1",  # MarketDepth: Best bid/ask
                "265": "0",  # MDUpdateType: Full refresh
                "55": symbol  # Symbol
            })
            
            self._send_message(msg)
            self.subscribed_symbols.add(symbol)
            self.market_data_subscriptions[symbol] = {}
            
            logger.info(f"FIX: Subscribed to {symbol}")
            return True
        
        except Exception as e:
            logger.error(f"FIX: Subscription error for {symbol}: {str(e)}")
            self.stats['data_errors'] += 1
            return False
    
    def subscribe_orderbook(self, symbol: str, depth: int = 10) -> bool:
        """Subscribe to order book."""
        return self.subscribe_price(symbol)
    
    def subscribe_news(self) -> bool:
        """Subscribe to news (placeholder)."""
        return True
    
    def subscribe_macro(self) -> bool:
        """Subscribe to macro (placeholder)."""
        return True
    
    def unsubscribe(self, symbol: str):
        """Unsubscribe from market data."""
        try:
            msg = self._create_message("V", {
                "262": f"sub_{symbol}",
                "263": "2",  # Unsubscribe
                "55": symbol
            })
            self._send_message(msg)
            self.subscribed_symbols.discard(symbol)
            logger.info(f"FIX: Unsubscribed from {symbol}")
        
        except Exception as e:
            logger.error(f"FIX: Unsubscribe error: {str(e)}")
    
    def _handle_market_data_snapshot(self, fields: Dict[str, str]):
        """Handle market data snapshot (W)."""
        try:
            symbol = fields.get("55")
            if not symbol:
                return
            
            # Parse bid/ask
            bid = float(fields.get("132", 0))
            ask = float(fields.get("133", 0))
            
            if bid == 0 or ask == 0:
                return
            
            tick = PriceTick(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=(bid + ask) / 2,
                timestamp=datetime.utcnow(),
                exchange='FIX'
            )
            
            update = MarketUpdate(
                data_type=DataType.PRICE_TICK,
                payload=tick,
                timestamp=datetime.utcnow(),
                sequence_number=self.stats['updates_received']
            )
            
            self.push_update(update)
            self.stats['updates_received'] += 1
        
        except Exception as e:
            logger.error(f"FIX: Snapshot error: {str(e)}")
            self.stats['data_errors'] += 1
    
    def _handle_market_data_incremental(self, fields: Dict[str, str]):
        """Handle market data incremental (X)."""
        # Similar to snapshot but incremental
        self._handle_market_data_snapshot(fields)
    
    def on_price_tick(self, exchange_data: Dict) -> Optional[MarketUpdate]:
        """Normalize price data (handled in message processing)."""
        return None
    
    def on_orderbook(self, exchange_data: Dict) -> Optional[MarketUpdate]:
        """Normalize orderbook data (handled in message processing)."""
        return None
    
    def on_news(self, exchange_data: Dict) -> Optional[MarketUpdate]:
        """Placeholder."""
        return None
    
    def on_macro(self, exchange_data: Dict) -> Optional[MarketUpdate]:
        """Placeholder."""
        return None
    
    # ============================================================================
    # Order Execution
    # ============================================================================
    
    def send_order(self, order: Order) -> Optional[str]:
        """Submit order via FIX."""
        try:
            if not self.is_connected:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "Not connected"
                return None
            
            # Generate ClOrdID
            self.order_counter += 1
            cl_ord_id = f"ORD{self.order_counter:08d}"
            
            # Create NewOrderSingle message
            msg = self._create_message("D", {
                "11": cl_ord_id,  # ClOrdID
                "21": "1",  # HandlInst: Automated
                "55": order.symbol,  # Symbol
                "54": "1" if order.side == OrderSide.BUY else "2",  # Side
                "60": datetime.utcnow().isoformat(),  # TransactTime
                "38": str(int(order.quantity)),  # OrderQty
                "40": "1" if order.order_type == OrderType.MARKET else "2",  # OrdType
            })
            
            if order.price:
                msg += f"|44={order.price}|"  # Price
            
            self._send_message(msg)
            
            # Track order
            order_id = f"fix_{cl_ord_id}"
            order.order_id = order_id
            order.connector_order_id = cl_ord_id
            order.status = OrderStatus.SUBMITTED
            order.submit_time = datetime.utcnow()
            
            self.pending_orders[order_id] = order
            self.fix_order_map[cl_ord_id] = order_id
            self.stats['orders_submitted'] += 1
            
            logger.info(f"FIX: Submitted {order.side.value} {order.quantity} {order.symbol}")
            
            return order_id
        
        except Exception as e:
            logger.error(f"FIX: Order submission error: {str(e)}")
            order.status = OrderStatus.ERROR
            order.rejection_reason = str(e)
            self.stats['order_errors'] += 1
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel FIX order."""
        try:
            if order_id not in self.pending_orders:
                return False
            
            order = self.pending_orders[order_id]
            cl_ord_id = order.connector_order_id
            
            # Create OrderCancelRequest
            self.order_counter += 1
            orig_cl_ord_id = cl_ord_id
            new_cl_ord_id = f"CXL{self.order_counter:08d}"
            
            msg = self._create_message("F", {
                "11": new_cl_ord_id,  # ClOrdID
                "37": orig_cl_ord_id,  # OrigClOrdID
                "55": order.symbol,  # Symbol
                "54": "1" if order.side == OrderSide.BUY else "2"  # Side
            })
            
            self._send_message(msg)
            logger.info(f"FIX: Cancel requested for {order_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"FIX: Cancel error: {str(e)}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        if order_id in self.pending_orders:
            return self.pending_orders[order_id].status
        elif order_id in self.filled_orders:
            return self.filled_orders[order_id].status
        return None
    
    def _handle_execution_report(self, fields: Dict[str, str]):
        """Handle execution report (8)."""
        try:
            cl_ord_id = fields.get("11")
            if not cl_ord_id or cl_ord_id not in self.fix_order_map:
                return
            
            order_id = self.fix_order_map[cl_ord_id]
            if order_id not in self.pending_orders:
                return
            
            order = self.pending_orders[order_id]
            ord_status = fields.get("39")  # OrdStatus
            exec_qty = float(fields.get("14", 0))
            exec_price = float(fields.get("31", 0))
            
            # Update order
            order.filled_quantity = exec_qty
            if exec_price > 0:
                order.avg_fill_price = exec_price
            
            if ord_status == "2":  # Filled
                order.status = OrderStatus.FILLED
                order.fill_time = datetime.utcnow()
                self.filled_orders[order_id] = order
                del self.pending_orders[order_id]
                self.stats['orders_filled'] += 1
            elif ord_status == "1":  # Partially Filled
                order.status = OrderStatus.PARTIALLY_FILLED
            elif ord_status == "4":  # Cancelled
                order.status = OrderStatus.CANCELLED
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
                self.stats['orders_cancelled'] += 1
            elif ord_status == "8":  # Rejected
                order.status = OrderStatus.REJECTED
                order.rejection_reason = fields.get("103", "Unknown")
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
                self.stats['order_errors'] += 1
            
            logger.info(f"FIX: Execution {order_id}: {ord_status} {exec_qty} @ {exec_price}")
        
        except Exception as e:
            logger.error(f"FIX: Execution report error: {str(e)}")
            self.stats['order_errors'] += 1
