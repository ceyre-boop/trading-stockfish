"""
Safety Layer - Phase RT-3

Implements real-time anomaly detection and safety checks for live trading.

Features:
  - Data sanity checks (time gaps, negative prices, absurd jumps)
  - Execution sanity checks (fills outside ranges, repeated rejects)
  - Feed health checks (stale data detection)
  - Automatic severity assessment and event generation
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Any, List, Set
import threading


class SafetyEventType(Enum):
    """Types of safety events."""
    DATA_ANOMALY = "DATA_ANOMALY"
    EXECUTION_ANOMALY = "EXECUTION_ANOMALY"
    FEED_STALE = "FEED_STALE"
    PRICE_SPIKE = "PRICE_SPIKE"
    ORDER_REJECT_LOOP = "ORDER_REJECT_LOOP"
    NEGATIVE_PRICE = "NEGATIVE_PRICE"
    TIME_GAP = "TIME_GAP"
    ORDERBOOK_INVALID = "ORDERBOOK_INVALID"


class SafetySeverity(Enum):
    """Event severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class SafetyEvent:
    """Safety system event."""
    timestamp: float
    event_type: SafetyEventType
    severity: str  # "info", "warning", "critical"
    symbol: str
    message: str
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'event_type': self.event_type.value,
            'severity': self.severity,
            'symbol': self.symbol,
            'message': self.message,
            'data': self.data or {}
        }


class SafetyLayer:
    """
    Real-time safety monitoring and anomaly detection.
    
    Monitors:
    - Price data sanity (negative prices, absurd jumps, time gaps)
    - Order book validity (bid/ask crossing, negative prices)
    - Execution integrity (fills in valid ranges, rejection patterns)
    - Feed health (stale data detection, connection gaps)
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SafetyLayer.
        
        Args:
            logger: Optional logger instance
            config: Optional configuration overrides
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # Configuration with defaults
        self.max_price_jump_pct = self.config.get('max_price_jump_pct', 20.0)
        self.max_time_gap_s = self.config.get('max_time_gap_s', 30.0)
        self.max_execution_time_s = self.config.get('max_execution_time_s', 5.0)
        self.reject_loop_threshold = self.config.get('reject_loop_threshold', 5)
        self.stale_data_threshold_s = self.config.get('stale_data_threshold_s', 10.0)
        self.bid_ask_spread_multiplier = self.config.get('bid_ask_spread_multiplier', 10.0)
        
        # Symbol tracking
        self._last_price: Dict[str, float] = {}
        self._last_update_time: Dict[str, float] = {}
        self._last_bid: Dict[str, float] = {}
        self._last_ask: Dict[str, float] = {}
        
        # Rejection tracking
        self._recent_rejects: Dict[str, List[float]] = {}
        self._reject_window_s = 60.0  # 1-minute window
        
        # Statistics
        self._total_anomalies = 0
        self._anomaly_counts: Dict[SafetyEventType, int] = {
            event_type: 0 for event_type in SafetyEventType
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("SafetyLayer initialized with config: %s", self.config)
    
    # ===== Price Data Checks =====
    
    def check_price_tick(self, symbol: str, bid: float, ask: float, last: float, timestamp: float) -> Optional[SafetyEvent]:
        """
        Check price tick for anomalies.
        
        Returns SafetyEvent if anomaly detected, None otherwise.
        """
        with self._lock:
            try:
                # Check negative prices
                if bid < 0 or ask < 0 or last < 0:
                    event = self._create_event(
                        SafetyEventType.NEGATIVE_PRICE,
                        SafetySeverity.CRITICAL.value,
                        symbol,
                        f"Negative price detected: bid={bid}, ask={ask}, last={last}",
                        {'bid': bid, 'ask': ask, 'last': last}
                    )
                    return event
                
                # Check bid > ask (crossed market)
                if bid > ask and ask > 0:
                    event = self._create_event(
                        SafetyEventType.ORDERBOOK_INVALID,
                        SafetySeverity.WARNING.value,
                        symbol,
                        f"Crossed market: bid={bid} > ask={ask}",
                        {'bid': bid, 'ask': ask}
                    )
                    return event
                
                # Check for price spike
                if symbol in self._last_price:
                    last_price = self._last_price[symbol]
                    if last_price > 0:
                        price_change_pct = abs((last - last_price) / last_price) * 100
                        
                        if price_change_pct > self.max_price_jump_pct:
                            event = self._create_event(
                                SafetyEventType.PRICE_SPIKE,
                                SafetySeverity.WARNING.value,
                                symbol,
                                f"Price jump {price_change_pct:.2f}% from {last_price} to {last}",
                                {
                                    'previous_price': last_price,
                                    'current_price': last,
                                    'change_pct': price_change_pct
                                }
                            )
                            return event
                
                # Check time gap
                if symbol in self._last_update_time:
                    time_gap = timestamp - self._last_update_time[symbol]
                    
                    if time_gap > self.max_time_gap_s:
                        event = self._create_event(
                            SafetyEventType.TIME_GAP,
                            SafetySeverity.WARNING.value,
                            symbol,
                            f"Time gap {time_gap:.1f}s exceeds threshold {self.max_time_gap_s}s",
                            {
                                'time_gap_s': time_gap,
                                'threshold_s': self.max_time_gap_s
                            }
                        )
                        return event
                
                # Update tracking
                self._last_price[symbol] = last
                self._last_bid[symbol] = bid
                self._last_ask[symbol] = ask
                self._last_update_time[symbol] = timestamp
                
                return None
            
            except Exception as e:
                self.logger.error("Price tick check error: %s", e, exc_info=True)
                return None
    
    # ===== Order Book Checks =====
    
    def check_orderbook(self, symbol: str, bid_levels: List[tuple], ask_levels: List[tuple], timestamp: float) -> Optional[SafetyEvent]:
        """
        Check order book for anomalies.
        
        Args:
            symbol: Trading symbol
            bid_levels: List of (price, quantity) bid levels
            ask_levels: List of (price, quantity) ask levels
            timestamp: Update timestamp
        
        Returns SafetyEvent if anomaly detected, None otherwise.
        """
        with self._lock:
            try:
                if not bid_levels or not ask_levels:
                    return None
                
                best_bid = bid_levels[0][0]
                best_ask = ask_levels[0][0]
                
                # Check negative prices
                if best_bid < 0 or best_ask < 0:
                    event = self._create_event(
                        SafetyEventType.NEGATIVE_PRICE,
                        SafetySeverity.CRITICAL.value,
                        symbol,
                        f"Negative orderbook prices: bid={best_bid}, ask={best_ask}",
                        {'best_bid': best_bid, 'best_ask': best_ask}
                    )
                    return event
                
                # Check crossed market
                if best_bid > best_ask and best_ask > 0:
                    event = self._create_event(
                        SafetyEventType.ORDERBOOK_INVALID,
                        SafetySeverity.CRITICAL.value,
                        symbol,
                        f"Crossed orderbook: bid={best_bid} > ask={best_ask}",
                        {'best_bid': best_bid, 'best_ask': best_ask}
                    )
                    return event
                
                # Check spread sanity
                if best_ask > 0:
                    spread_pct = ((best_ask - best_bid) / best_ask) * 100
                    
                    # Get historical spread
                    if symbol in self._last_ask and symbol in self._last_bid:
                        hist_spread = self._last_ask[symbol] - self._last_bid[symbol]
                        if hist_spread > 0:
                            hist_spread_pct = (hist_spread / self._last_ask[symbol]) * 100
                            spread_multiple = spread_pct / hist_spread_pct if hist_spread_pct > 0 else 1
                            
                            if spread_multiple > self.bid_ask_spread_multiplier:
                                event = self._create_event(
                                    SafetyEventType.DATA_ANOMALY,
                                    SafetySeverity.WARNING.value,
                                    symbol,
                                    f"Spread anomaly: {spread_pct:.4f}% ({spread_multiple:.1f}x normal)",
                                    {
                                        'current_spread_pct': spread_pct,
                                        'historical_spread_pct': hist_spread_pct,
                                        'multiplier': spread_multiple
                                    }
                                )
                                return event
                
                return None
            
            except Exception as e:
                self.logger.error("Orderbook check error: %s", e, exc_info=True)
                return None
    
    # ===== Execution Checks =====
    
    def check_execution(
        self,
        symbol: str,
        order_price: float,
        fill_price: float,
        fill_quantity: float,
        order_quantity: float
    ) -> Optional[SafetyEvent]:
        """
        Check order execution for anomalies.
        
        Args:
            symbol: Trading symbol
            order_price: Original order price
            fill_price: Execution fill price
            fill_quantity: Filled quantity
            order_quantity: Original order quantity
        
        Returns SafetyEvent if anomaly detected, None otherwise.
        """
        with self._lock:
            try:
                # Check negative prices
                if order_price < 0 or fill_price < 0:
                    event = self._create_event(
                        SafetyEventType.NEGATIVE_PRICE,
                        SafetySeverity.CRITICAL.value,
                        symbol,
                        f"Negative execution price: order={order_price}, fill={fill_price}",
                        {'order_price': order_price, 'fill_price': fill_price}
                    )
                    return event
                
                # Check fill price significantly different from order price
                if order_price > 0:
                    price_diff_pct = abs((fill_price - order_price) / order_price) * 100
                    
                    # For market orders, allow larger slippage
                    slippage_threshold = 5.0  # 5% typical slippage
                    
                    if price_diff_pct > slippage_threshold:
                        event = self._create_event(
                            SafetyEventType.EXECUTION_ANOMALY,
                            SafetySeverity.WARNING.value,
                            symbol,
                            f"Execution slippage {price_diff_pct:.2f}% (order={order_price}, fill={fill_price})",
                            {
                                'order_price': order_price,
                                'fill_price': fill_price,
                                'slippage_pct': price_diff_pct
                            }
                        )
                        return event
                
                # Check partial fill consistency
                if order_quantity > 0:
                    fill_ratio = fill_quantity / order_quantity
                    
                    if fill_ratio > 1.0:
                        event = self._create_event(
                            SafetyEventType.EXECUTION_ANOMALY,
                            SafetySeverity.CRITICAL.value,
                            symbol,
                            f"Fill quantity {fill_quantity} exceeds order {order_quantity}",
                            {
                                'fill_quantity': fill_quantity,
                                'order_quantity': order_quantity,
                                'fill_ratio': fill_ratio
                            }
                        )
                        return event
                
                return None
            
            except Exception as e:
                self.logger.error("Execution check error: %s", e, exc_info=True)
                return None
    
    # ===== Rejection Pattern Detection =====
    
    def record_order_rejection(self, symbol: str, timestamp: float) -> Optional[SafetyEvent]:
        """
        Record order rejection and detect rejection loops.
        
        Returns SafetyEvent if rejection loop detected, None otherwise.
        """
        with self._lock:
            try:
                if symbol not in self._recent_rejects:
                    self._recent_rejects[symbol] = []
                
                # Add this rejection
                self._recent_rejects[symbol].append(timestamp)
                
                # Clean old rejections (older than window)
                window_start = timestamp - self._reject_window_s
                self._recent_rejects[symbol] = [
                    t for t in self._recent_rejects[symbol] if t >= window_start
                ]
                
                # Check if threshold exceeded
                reject_count = len(self._recent_rejects[symbol])
                
                if reject_count >= self.reject_loop_threshold:
                    event = self._create_event(
                        SafetyEventType.ORDER_REJECT_LOOP,
                        SafetySeverity.CRITICAL.value,
                        symbol,
                        f"Rejection loop detected: {reject_count} rejections in {self._reject_window_s}s",
                        {
                            'reject_count': reject_count,
                            'window_s': self._reject_window_s,
                            'threshold': self.reject_loop_threshold
                        }
                    )
                    return event
                
                return None
            
            except Exception as e:
                self.logger.error("Rejection check error: %s", e, exc_info=True)
                return None
    
    # ===== Feed Health Checks =====
    
    def check_feed_staleness(self, symbol: str, current_time: float) -> Optional[SafetyEvent]:
        """
        Check if feed data is stale.
        
        Returns SafetyEvent if stale, None otherwise.
        """
        with self._lock:
            try:
                if symbol not in self._last_update_time:
                    return None
                
                time_since_update = current_time - self._last_update_time[symbol]
                
                if time_since_update > self.stale_data_threshold_s:
                    event = self._create_event(
                        SafetyEventType.FEED_STALE,
                        SafetySeverity.WARNING.value,
                        symbol,
                        f"Feed stale for {time_since_update:.1f}s (threshold: {self.stale_data_threshold_s}s)",
                        {
                            'time_since_update_s': time_since_update,
                            'threshold_s': self.stale_data_threshold_s,
                            'last_update': datetime.fromtimestamp(self._last_update_time[symbol]).isoformat()
                        }
                    )
                    return event
                
                return None
            
            except Exception as e:
                self.logger.error("Feed staleness check error: %s", e, exc_info=True)
                return None
    
    # ===== Event Creation =====
    
    def _create_event(
        self,
        event_type: SafetyEventType,
        severity: str,
        symbol: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> SafetyEvent:
        """Create and track safety event."""
        self._total_anomalies += 1
        self._anomaly_counts[event_type] += 1
        
        event = SafetyEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            symbol=symbol,
            message=message,
            data=data
        )
        
        self.logger.warning(
            "Safety event: %s [%s] %s: %s",
            event_type.value, severity, symbol, message
        )
        
        return event
    
    # ===== Statistics =====
    
    def reset(self) -> None:
        """Reset safety layer state."""
        with self._lock:
            self._last_price.clear()
            self._last_update_time.clear()
            self._last_bid.clear()
            self._last_ask.clear()
            self._recent_rejects.clear()
            self.logger.info("Safety layer state reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get safety layer statistics."""
        with self._lock:
            return {
                'total_anomalies': self._total_anomalies,
                'anomaly_counts': {
                    k.value: v for k, v in self._anomaly_counts.items()
                },
                'tracked_symbols': len(self._last_price),
                'symbols_with_rejects': len(self._recent_rejects)
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        with self._lock:
            tracked_symbols = set(self._last_update_time.keys())
            stale_symbols = set()
            current_time = time.time()
            
            for symbol in tracked_symbols:
                if current_time - self._last_update_time[symbol] > self.stale_data_threshold_s:
                    stale_symbols.add(symbol)
            
            return {
                'tracked_symbols': len(tracked_symbols),
                'stale_symbols': list(stale_symbols),
                'total_anomalies': self._total_anomalies,
                'health_status': 'healthy' if len(stale_symbols) == 0 else 'degraded'
            }
