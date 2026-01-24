"""
Exchange Manager - Multi-Connector Orchestrator

Manages multiple exchange connectors simultaneously:
  - Unified API for all exchanges
  - Subscription routing
  - Order execution across connectors
  - Health monitoring and failover
  - Statistics aggregation
  - Graceful shutdown

Supported connectors:
  - Interactive Brokers (IBKR)
  - FIX Protocol (FIX)
  - ZeroMQ (ZMQ) - data feeds
  - Custom implementations

Author: Trading Stockfish Development Team
Date: January 19, 2026
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

from realtime.exchange_base_connector import BaseConnector, ConnectorStatus, Order, OrderStatus


logger = logging.getLogger(__name__)


class FailoverStrategy(Enum):
    """Failover strategies."""
    ROUND_ROBIN = "round_robin"
    PRIMARY_BACKUP = "primary_backup"
    BEST_AVAILABLE = "best_available"


class ExchangeManager:
    """
    Centralized manager for multiple exchange connectors.
    
    Features:
      - Unified subscription API
      - Automatic failover
      - Health monitoring
      - Statistics aggregation
      - Thread-safe operations
    
    Example:
        manager = ExchangeManager()
        manager.add_connector(ibkr_connector, primary=True)
        manager.add_connector(fix_connector, primary=False)
        manager.add_connector(zmq_connector, data_only=True)
        
        manager.start_all()
        manager.subscribe_price(['SPY', 'QQQ'])
        manager.send_order(order)
        manager.stop_all()
    """
    
    def __init__(self, failover_strategy: FailoverStrategy = FailoverStrategy.BEST_AVAILABLE):
        """
        Initialize ExchangeManager.
        
        Args:
            failover_strategy: Strategy for selecting connectors
        """
        self.connectors: Dict[str, BaseConnector] = {}
        self.connector_order = []  # Ordered list of connector names
        self.primary_connector: Optional[str] = None
        self.data_only_connectors: Set[str] = set()  # ZMQ, etc
        self.execution_connectors: Set[str] = set()  # IBKR, FIX
        
        self.failover_strategy = failover_strategy
        self.current_connector_idx = 0  # For round-robin
        
        # Subscription tracking
        self.symbol_subscriptions: Dict[str, List[str]] = {}  # symbol -> [connectors]
        
        # Order tracking
        self.orders: Dict[str, Order] = {}  # order_id -> Order
        self.order_lock = threading.Lock()
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.health_thread = None
        self.running = False
        
        logger.info("Initialized ExchangeManager")
    
    # ============================================================================
    # Connector Management
    # ============================================================================
    
    def add_connector(self, connector: BaseConnector, name: str = None,
                     primary: bool = False, data_only: bool = False):
        """
        Add connector to manager.
        
        Args:
            connector: BaseConnector instance
            name: Unique connector name (defaults to connector.name)
            primary: If True, use as primary connector
            data_only: If True, don't send orders to this connector
        """
        if name is None:
            name = connector.name
        
        self.connectors[name] = connector
        self.connector_order.append(name)
        
        if data_only:
            self.data_only_connectors.add(name)
        else:
            self.execution_connectors.add(name)
        
        if primary:
            self.primary_connector = name
        
        logger.info(f"Added {name} connector (primary={primary}, data_only={data_only})")
    
    def remove_connector(self, name: str) -> bool:
        """Remove connector."""
        try:
            if name in self.connectors:
                connector = self.connectors[name]
                if connector.is_connected:
                    connector.disconnect()
                
                del self.connectors[name]
                self.connector_order.remove(name)
                self.data_only_connectors.discard(name)
                self.execution_connectors.discard(name)
                
                if self.primary_connector == name:
                    self.primary_connector = None
                
                logger.info(f"Removed {name} connector")
                return True
        except Exception as e:
            logger.error(f"Error removing connector {name}: {str(e)}")
        
        return False
    
    def get_connector(self, name: str) -> Optional[BaseConnector]:
        """Get connector by name."""
        return self.connectors.get(name)
    
    def list_connectors(self) -> List[Tuple[str, str, bool]]:
        """List all connectors with status."""
        result = []
        for name in self.connector_order:
            connector = self.connectors[name]
            status = "CONNECTED" if connector.is_connected else "DISCONNECTED"
            is_primary = (name == self.primary_connector)
            result.append((name, status, is_primary))
        return result
    
    # ============================================================================
    # Lifecycle
    # ============================================================================
    
    def start_all(self) -> bool:
        """Connect all connectors."""
        logger.info("Starting all connectors...")
        
        success_count = 0
        for name in self.connector_order:
            connector = self.connectors[name]
            try:
                if connector.connect():
                    success_count += 1
                    logger.info(f"✓ {name} connected")
                else:
                    logger.warning(f"✗ {name} failed to connect")
            except Exception as e:
                logger.error(f"✗ {name} error: {str(e)}")
        
        if success_count == 0:
            logger.error("No connectors successfully connected!")
            return False
        
        # Start health monitoring
        self.running = True
        self.health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_thread.start()
        
        logger.info(f"ExchangeManager: {success_count}/{len(self.connectors)} connectors started")
        return success_count > 0
    
    def stop_all(self):
        """Disconnect all connectors."""
        logger.info("Stopping all connectors...")
        
        self.running = False
        if self.health_thread:
            self.health_thread.join(timeout=2)
        
        for name in self.connector_order:
            try:
                connector = self.connectors[name]
                if connector.is_connected:
                    connector.disconnect()
                    logger.info(f"✓ {name} disconnected")
            except Exception as e:
                logger.error(f"✗ {name} disconnect error: {str(e)}")
        
        logger.info("All connectors stopped")
    
    # ============================================================================
    # Subscription Management
    # ============================================================================
    
    def subscribe_price(self, symbols: List[str], 
                       connectors: List[str] = None) -> bool:
        """
        Subscribe to price data on specified connectors.
        
        Args:
            symbols: List of symbols
            connectors: Connector names (None = all, "primary" = primary only)
        
        Returns:
            True if at least one subscription succeeded
        """
        if connectors is None:
            connectors = self.connector_order
        
        success = False
        for symbol in symbols:
            for conn_name in connectors:
                if conn_name not in self.connectors:
                    continue
                
                connector = self.connectors[conn_name]
                try:
                    if connector.subscribe_price(symbol):
                        if symbol not in self.symbol_subscriptions:
                            self.symbol_subscriptions[symbol] = []
                        if conn_name not in self.symbol_subscriptions[symbol]:
                            self.symbol_subscriptions[symbol].append(conn_name)
                        success = True
                        logger.debug(f"✓ {symbol} subscribed on {conn_name}")
                    else:
                        logger.warning(f"✗ {symbol} subscription failed on {conn_name}")
                except Exception as e:
                    logger.error(f"Subscription error for {symbol} on {conn_name}: {str(e)}")
        
        return success
    
    def subscribe_orderbook(self, symbols: List[str], depth: int = 10,
                           connectors: List[str] = None) -> bool:
        """Subscribe to order book data."""
        if connectors is None:
            connectors = self.connector_order
        
        success = False
        for symbol in symbols:
            for conn_name in connectors:
                if conn_name not in self.connectors:
                    continue
                
                connector = self.connectors[conn_name]
                try:
                    if connector.subscribe_orderbook(symbol, depth):
                        success = True
                except Exception as e:
                    logger.error(f"Orderbook subscription error for {symbol}: {str(e)}")
        
        return success
    
    def unsubscribe(self, symbols: List[str], 
                   connectors: List[str] = None):
        """Unsubscribe from symbols."""
        if connectors is None:
            connectors = self.connector_order
        
        for symbol in symbols:
            for conn_name in connectors:
                if conn_name in self.connectors:
                    try:
                        self.connectors[conn_name].unsubscribe(symbol)
                    except Exception as e:
                        logger.error(f"Unsubscribe error for {symbol}: {str(e)}")
            
            if symbol in self.symbol_subscriptions:
                del self.symbol_subscriptions[symbol]
    
    # ============================================================================
    # Order Execution
    # ============================================================================
    
    def send_order(self, order: Order, connector_name: str = None) -> Optional[str]:
        """
        Submit order to exchange.
        
        Args:
            order: Order object
            connector_name: Specific connector, or None for best available
        
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Select connector
            if connector_name:
                if connector_name not in self.connectors:
                    logger.error(f"Unknown connector: {connector_name}")
                    return None
                target_connector = self.connectors[connector_name]
            else:
                target_connector = self._select_execution_connector()
                if not target_connector:
                    logger.error("No execution connectors available")
                    return None
            
            # Send order
            order_id = target_connector.send_order(order)
            
            if order_id:
                with self.order_lock:
                    self.orders[order_id] = order
                logger.info(f"Order {order_id} sent to {target_connector.name}")
            
            return order_id
        
        except Exception as e:
            logger.error(f"Order submission error: {str(e)}")
            order.status = OrderStatus.ERROR
            order.rejection_reason = str(e)
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        try:
            with self.order_lock:
                if order_id not in self.orders:
                    logger.warning(f"Order not found: {order_id}")
                    return False
                
                order = self.orders[order_id]
            
            # Find connector with this order
            for connector in self.connectors.values():
                if connector.name in self.execution_connectors:
                    if connector.cancel_order(order_id):
                        logger.info(f"Order {order_id} cancelled")
                        return True
            
            logger.warning(f"Could not cancel order {order_id}")
            return False
        
        except Exception as e:
            logger.error(f"Cancel error: {str(e)}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        with self.order_lock:
            if order_id in self.orders:
                return self.orders[order_id].status
        return None
    
    # ============================================================================
    # Health Monitoring
    # ============================================================================
    
    def _health_check_loop(self):
        """Monitor connector health."""
        while self.running:
            try:
                for name in self.connector_order:
                    connector = self.connectors[name]
                    
                    # Check connection
                    if connector.is_connected and not connector.is_connected_check():
                        logger.warning(f"{name}: Connection lost")
                        connector.set_status(ConnectorStatus.RECONNECTING)
                        
                        # Attempt reconnect
                        if connector.connect():
                            logger.info(f"{name}: Reconnected")
                        else:
                            logger.error(f"{name}: Reconnection failed")
                
                time.sleep(self.health_check_interval)
            
            except Exception as e:
                logger.error(f"Health check error: {str(e)}")
                time.sleep(5)
    
    def get_health_status(self) -> Dict[str, Dict]:
        """Get health status for all connectors."""
        status = {}
        for name in self.connector_order:
            connector = self.connectors[name]
            status[name] = {
                'connected': connector.is_connected,
                'status': connector.get_status(),
                'stats': connector.get_stats()
            }
        return status
    
    # ============================================================================
    # Statistics
    # ============================================================================
    
    def get_stats(self) -> Dict:
        """Aggregate statistics from all connectors."""
        stats = {
            'connectors': {},
            'total': {
                'connected': 0,
                'updates_received': 0,
                'updates_normalized': 0,
                'orders_submitted': 0,
                'orders_filled': 0,
                'orders_cancelled': 0,
                'errors': 0
            }
        }
        
        for name in self.connector_order:
            connector = self.connectors[name]
            connector_stats = connector.get_stats()
            stats['connectors'][name] = connector_stats
            
            if connector.is_connected:
                stats['total']['connected'] += 1
            
            stats['total']['updates_received'] += connector_stats.get('updates_received', 0)
            stats['total']['updates_normalized'] += connector_stats.get('updates_normalized', 0)
            stats['total']['orders_submitted'] += connector_stats.get('orders_submitted', 0)
            stats['total']['orders_filled'] += connector_stats.get('orders_filled', 0)
            stats['total']['orders_cancelled'] += connector_stats.get('orders_cancelled', 0)
            stats['total']['errors'] += connector_stats.get('data_errors', 0)
        
        return stats
    
    # ============================================================================
    # Selection Logic
    # ============================================================================
    
    def _select_execution_connector(self) -> Optional[BaseConnector]:
        """Select connector for order execution."""
        candidates = [
            self.connectors[name] for name in self.execution_connectors
            if name in self.connectors and self.connectors[name].is_connected
        ]
        
        if not candidates:
            return None
        
        if self.failover_strategy == FailoverStrategy.PRIMARY_BACKUP:
            # Use primary if available
            if self.primary_connector and self.primary_connector in [c.name for c in candidates]:
                return self.connectors[self.primary_connector]
            return candidates[0]
        
        elif self.failover_strategy == FailoverStrategy.ROUND_ROBIN:
            connector = candidates[self.current_connector_idx % len(candidates)]
            self.current_connector_idx += 1
            return connector
        
        else:  # BEST_AVAILABLE
            # Return connector with best health
            best = max(candidates, key=lambda c: (c.is_connected, -c.stats.get('data_errors', 0)))
            return best
    
    # ============================================================================
    # Context Manager Support
    # ============================================================================
    
    def __enter__(self):
        """Context manager entry."""
        self.start_all()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_all()
    
    def __str__(self):
        """String representation."""
        status = ", ".join([f"{name}:{c.get_status()}" for name, c in self.connectors.items()])
        return f"ExchangeManager({status})"
