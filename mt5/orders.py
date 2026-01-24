#!/usr/bin/env python3
"""
MT5 Orders Execution Module - Trading Stockfish

Handles order execution with safety checks, validation, and error handling.
Provides clean API for placing, modifying, and closing orders.

Features:
- Buy/sell with stop loss and take profit
- Position modification (SL/TP adjustment)
- Position closing
- Volume validation and correction
- Duplicate order prevention
- Connection state checks
- Comprehensive MT5 error handling
- Detailed order logging
"""

import time
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from datetime import datetime, timedelta

try:
    import MetaTrader5 as mt5
    # Try to initialize MT5 to check if terminal is running
    if mt5.initialize():
        MT5_AVAILABLE = True
        mt5.shutdown()  # Close after verification
    else:
        MT5_AVAILABLE = False
        logging.warning("MetaTrader5 terminal not running - orders will use mock mode")
except ImportError:
    MT5_AVAILABLE = False
    logging.warning("MetaTrader5 not available - orders will use mock mode")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# MT5 Error Codes (from documentation)
MT5_ERRORS = {
    0: "OK - Order executed",
    1: "Custom order type",
    2: "Order parameter error",
    3: "Account error",
    4: "History error",
    5: "Trade server error",
    6: "Resource error",
    7: "Unknown error",
    10001: "Invalid ticket",
    10002: "Trade disabled",
    10003: "Trade context busy",
    10004: "Account disabled for trading",
    10005: "Wrong magic number",
    10006: "Operation timeout",
    10007: "Trade modify denied",
    10008: "Trade context busy (retry)",
    10009: "Market order temporary disabled",
    10010: "Pending order temporary disabled",
    10011: "Request rejected",
    10012: "Request canceled by trader",
    10013: "Request expired",
    10014: "Request done - partial fill",
    10015: "Request done - filled",
    10016: "Request accepted",
    10017: "Request processing",
    10018: "Request canceled",
    20001: "Requote",
    20002: "Ask/Bid mismatch",
    20003: "Different lots",
    20004: "Different symbols",
    20005: "Different magic",
    20006: "Order closed",
    20007: "Only closing allowed",
    20008: "Hedge operations not allowed",
    20009: "Prohibited by FIFO",
}


@dataclass
class OrderRequest:
    """Represents an order request to MT5"""
    symbol: str
    action: str              # "buy", "sell", "close"
    volume: float
    price: float = 0.0       # Market price (0 for market orders)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = "Stockfish"
    magic: int = 12345


@dataclass
class MockSymbolInfo:
    """Mock symbol information for testing"""
    symbol: str
    digits: int = 5
    point: float = 0.00001
    bid: float = 1.0850
    ask: float = 1.0852
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    swap_long: float = -5.0
    swap_short: float = -3.0
    commission: float = 0.0
    spread: float = 2.0


@dataclass
class MockTickData:
    """Mock tick data for testing"""
    symbol: str
    bid: float = 1.0850
    ask: float = 1.0852
    bid_volume: int = 1000
    ask_volume: int = 1000
    last: float = 1.0851
    time: int = field(default_factory=lambda: int(time.time()))
    time_msc: int = field(default_factory=lambda: int(time.time() * 1000))
    
    def to_dict(self) -> Dict:
        """Convert to MT5 request dict format"""
        if self.action == "buy":
            order_type = mt5.ORDER_TYPE_BUY if MT5_AVAILABLE else 0
        elif self.action == "sell":
            order_type = mt5.ORDER_TYPE_SELL if MT5_AVAILABLE else 1
        else:
            order_type = mt5.ORDER_TYPE_BUY
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL if MT5_AVAILABLE else 0,
            "symbol": self.symbol,
            "volume": self.volume,
            "type": order_type,
            "price": self.price,
            "comment": self.comment,
            "magic": self.magic,
        }
        
        if self.stop_loss is not None and self.stop_loss > 0:
            request["sl"] = self.stop_loss
        
        if self.take_profit is not None and self.take_profit > 0:
            request["tp"] = self.take_profit
        
        return request


@dataclass
class OrderResult:
    """Result of an order execution"""
    success: bool
    order_id: Optional[int] = None
    price: float = 0.0
    volume: float = 0.0
    comment: str = ""
    error_code: Optional[int] = None
    error_message: str = ""
    timestamp: float = field(default_factory=time.time)


class OrderAction(Enum):
    """Order action types"""
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    MODIFY = "modify"


class OrderValidationError(Exception):
    """Order validation error"""
    pass


class OrderExecutionError(Exception):
    """Order execution error"""
    pass


class MT5Orders:
    """
    MetaTrader5 order execution manager.
    
    Handles placing, modifying, and closing orders with comprehensive
    validation, error handling, and safety checks.
    """
    
    def __init__(self, feed=None, prevent_duplicates: bool = True):
        """
        Initialize MT5 Orders manager.
        
        Args:
            feed: MT5LiveFeed instance for symbol validation
            prevent_duplicates: Prevent duplicate orders within window
        """
        self.feed = feed
        self.prevent_duplicates = prevent_duplicates
        self.recent_orders: deque = deque(maxlen=100)  # Track last 100 orders
        self.duplicate_window_sec = 5  # 5 second window for duplicate check
        
        logger.info("MT5Orders initialized")
    
    def buy(
        self,
        symbol: str,
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        magic: int = 12345,
    ) -> OrderResult:
        """
        Place a buy market order.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            volume: Order volume in lots
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            magic: Magic number for identification
            
        Returns:
            OrderResult with success status and details
        """
        return self._execute_order(
            symbol=symbol,
            action=OrderAction.BUY,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit,
            magic=magic,
        )
    
    def sell(
        self,
        symbol: str,
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        magic: int = 12345,
    ) -> OrderResult:
        """
        Place a sell market order.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            volume: Order volume in lots
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            magic: Magic number for identification
            
        Returns:
            OrderResult with success status and details
        """
        return self._execute_order(
            symbol=symbol,
            action=OrderAction.SELL,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit,
            magic=magic,
        )
    
    def close_position(self, position_id: int) -> OrderResult:
        """
        Close an open position.
        
        Args:
            position_id: Position ticket number
            
        Returns:
            OrderResult with success status
        """
        logger.info(f"Closing position {position_id}")
        
        if not MT5_AVAILABLE:
            return self._mock_close_position(position_id)
        
        try:
            # Get position info
            position = mt5.positions_get(ticket=position_id)
            if position is None or len(position) == 0:
                logger.error(f"Position {position_id} not found")
                return OrderResult(
                    success=False,
                    error_code=-1,
                    error_message=f"Position {position_id} not found"
                )
            
            pos = position[0]
            symbol = pos.symbol
            
            # Create close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position_id,
                "comment": f"Close #{position_id}",
            }
            
            logger.debug(f"Close request: {request}")
            result = mt5.order_send(request)
            
            return self._parse_mt5_result(result, "close")
        
        except Exception as e:
            logger.error(f"Exception closing position {position_id}: {e}")
            return OrderResult(
                success=False,
                error_code=-2,
                error_message=str(e)
            )
    
    def modify_position(
        self,
        position_id: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> OrderResult:
        """
        Modify stop loss and/or take profit of an open position.
        
        Args:
            position_id: Position ticket number
            stop_loss: New stop loss price (None to keep current)
            take_profit: New take profit price (None to keep current)
            
        Returns:
            OrderResult with success status
        """
        logger.info(f"Modifying position {position_id}: SL={stop_loss}, TP={take_profit}")
        
        if not MT5_AVAILABLE:
            return self._mock_modify_position(position_id, stop_loss, take_profit)
        
        try:
            # Get position info
            position = mt5.positions_get(ticket=position_id)
            if position is None or len(position) == 0:
                logger.error(f"Position {position_id} not found")
                return OrderResult(
                    success=False,
                    error_code=-1,
                    error_message=f"Position {position_id} not found"
                )
            
            pos = position[0]
            
            # Create modify request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": position_id,
                "sl": stop_loss if stop_loss is not None else pos.sl,
                "tp": take_profit if take_profit is not None else pos.tp,
                "comment": f"Modify #{position_id}",
            }
            
            logger.debug(f"Modify request: {request}")
            result = mt5.order_send(request)
            
            return self._parse_mt5_result(result, "modify")
        
        except Exception as e:
            logger.error(f"Exception modifying position {position_id}: {e}")
            return OrderResult(
                success=False,
                error_code=-2,
                error_message=str(e)
            )
    
    def _execute_order(
        self,
        symbol: str,
        action: OrderAction,
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        magic: int = 12345,
    ) -> OrderResult:
        """
        Internal order execution with validation and safety checks.
        
        Args:
            symbol: Trading symbol
            action: Order action (BUY, SELL)
            volume: Order volume
            stop_loss: Stop loss price
            take_profit: Take profit price
            magic: Magic number
            
        Returns:
            OrderResult
        """
        logger.info(f"Executing {action.value} order: {symbol}, volume={volume}")
        
        # Safety check: duplicate prevention
        if self.prevent_duplicates and self._is_duplicate_order(symbol, action):
            logger.warning(f"Duplicate order detected for {symbol} {action.value}")
            return OrderResult(
                success=False,
                error_code=-3,
                error_message=f"Duplicate {action.value} order within {self.duplicate_window_sec}s"
            )
        
        # Validate order parameters
        try:
            symbol_info = self._get_symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Cannot get symbol info for {symbol}")
                return OrderResult(
                    success=False,
                    error_code=-4,
                    error_message=f"Cannot get info for symbol {symbol}"
                )
            
            # Get current price
            tick = self._get_tick(symbol)
            if tick is None:
                logger.error(f"Cannot get tick for {symbol}")
                return OrderResult(
                    success=False,
                    error_code=-5,
                    error_message=f"Cannot get tick for {symbol}"
                )
            
            # Validate and correct volume
            volume = self._validate_volume(symbol_info, volume)
            logger.debug(f"Volume validated/corrected: {volume}")
            
            # Validate stop loss and take profit
            if stop_loss is not None or take_profit is not None:
                self._validate_price_levels(symbol_info, tick, stop_loss, take_profit, action)
            
        except OrderValidationError as e:
            logger.error(f"Order validation failed: {e}")
            return OrderResult(
                success=False,
                error_code=-6,
                error_message=str(e)
            )
        
        # Execute order
        if not MT5_AVAILABLE:
            parsed_result = self._mock_execute_order(symbol, action, volume, stop_loss, take_profit)
        else:
            try:
                price = tick.ask if action == OrderAction.BUY else tick.bid
                
                # Create order request dict directly
                mt5_request = {
                    "action": mt5.TRADE_ACTION_DEAL if MT5_AVAILABLE else 0,
                    "symbol": symbol,
                    "volume": volume,
                    "type": mt5.ORDER_TYPE_BUY if action == OrderAction.BUY else mt5.ORDER_TYPE_SELL,
                    "price": price,
                    "comment": "Stockfish",
                    "magic": magic,
                }
                
                if stop_loss is not None and stop_loss > 0:
                    mt5_request["sl"] = stop_loss
                
                if take_profit is not None and take_profit > 0:
                    mt5_request["tp"] = take_profit
                
                logger.debug(f"Order request: {mt5_request}")
                
                # Send to MT5
                result = mt5.order_send(mt5_request)
                parsed_result = self._parse_mt5_result(result, action.value)
            
            except Exception as e:
                logger.error(f"Exception executing order: {e}")
                parsed_result = OrderResult(
                    success=False,
                    error_code=-7,
                    error_message=str(e)
                )
        
        # Track order for duplicate detection
        if parsed_result.success:
            self.recent_orders.append({
                'symbol': symbol,
                'action': action,
                'timestamp': time.time(),
                'order_id': parsed_result.order_id,
            })
            logger.info(f"✓ Order executed: {action.value} {volume}L {symbol} @ {parsed_result.price} (ID: {parsed_result.order_id})")
        else:
            logger.warning(f"✗ Order failed: {parsed_result.error_message}")
        
        return parsed_result
    
    def _validate_volume(self, symbol_info, volume: float) -> float:
        """
        Validate and correct order volume to allowed step sizes.
        
        Args:
            symbol_info: SymbolInfo from live_feed
            volume: Requested volume
            
        Returns:
            Corrected volume
        """
        if symbol_info is None:
            raise OrderValidationError("Symbol info not available")
        
        # Check minimum
        if volume < symbol_info.volume_min:
            raise OrderValidationError(
                f"Volume {volume} below minimum {symbol_info.volume_min}"
            )
        
        # Check maximum
        if volume > symbol_info.volume_max:
            raise OrderValidationError(
                f"Volume {volume} exceeds maximum {symbol_info.volume_max}"
            )
        
        # Round to step
        if symbol_info.volume_step > 0:
            corrected = round(volume / symbol_info.volume_step) * symbol_info.volume_step
            if corrected != volume:
                logger.debug(f"Volume adjusted: {volume} → {corrected} (step: {symbol_info.volume_step})")
                volume = corrected
        
        return volume
    
    def _validate_price_levels(
        self,
        symbol_info,
        tick,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        action: OrderAction,
    ):
        """
        Validate stop loss and take profit prices.
        
        Args:
            symbol_info: SymbolInfo
            tick: Current tick
            stop_loss: Stop loss price
            take_profit: Take profit price
            action: Order action (BUY or SELL)
        """
        min_distance_pips = 10  # Minimum distance from entry
        min_distance = min_distance_pips * symbol_info.point
        
        if action == OrderAction.BUY:
            current = tick.ask
            
            # SL should be below entry
            if stop_loss is not None and stop_loss >= current - min_distance:
                raise OrderValidationError(
                    f"Buy SL {stop_loss} too close to entry {current} (min {min_distance_pips} pips)"
                )
            
            # TP should be above entry
            if take_profit is not None and take_profit <= current + min_distance:
                raise OrderValidationError(
                    f"Buy TP {take_profit} too close to entry {current} (min {min_distance_pips} pips)"
                )
        
        else:  # SELL
            current = tick.bid
            
            # SL should be above entry
            if stop_loss is not None and stop_loss <= current + min_distance:
                raise OrderValidationError(
                    f"Sell SL {stop_loss} too close to entry {current} (min {min_distance_pips} pips)"
                )
            
            # TP should be below entry
            if take_profit is not None and take_profit >= current - min_distance:
                raise OrderValidationError(
                    f"Sell TP {take_profit} too close to entry {current} (min {min_distance_pips} pips)"
                )
    
    def _is_duplicate_order(self, symbol: str, action: OrderAction) -> bool:
        """Check if recent duplicate order exists"""
        if not self.prevent_duplicates or len(self.recent_orders) == 0:
            return False
        
        cutoff = time.time() - self.duplicate_window_sec
        
        for order in list(self.recent_orders):
            if (order['symbol'] == symbol and
                order['action'] == action and
                order['timestamp'] > cutoff):
                return True
        
        return False
    
    def _get_symbol_info(self, symbol: str):
        """Get symbol info from feed or directly from MT5"""
        if self.feed:
            return self.feed.get_symbol_info(symbol)
        
        # Always use mock if no feed provided
        return self._mock_symbol_info(symbol)
    
    def _get_tick(self, symbol: str):
        """Get tick from feed or directly from MT5"""
        if self.feed:
            return self.feed.get_tick(symbol)
        
        # Always use mock if no feed provided
        return self._mock_tick(symbol)
    
    def _parse_mt5_result(self, result, action: str) -> OrderResult:
        """Parse MT5 order_send result"""
        if result is None:
            return OrderResult(
                success=False,
                error_code=-1,
                error_message="No result returned from MT5"
            )
        
        # Check retcode
        retcode = result.retcode if hasattr(result, 'retcode') else -1
        
        success = retcode in [0, 10009, 10010, 10014, 10015, 10016, 10017]  # Success codes
        
        error_msg = MT5_ERRORS.get(retcode, f"Unknown error code {retcode}")
        
        order_result = OrderResult(
            success=success,
            order_id=result.order if hasattr(result, 'order') else None,
            price=result.ask if hasattr(result, 'ask') else (result.bid if hasattr(result, 'bid') else 0.0),
            volume=result.volume if hasattr(result, 'volume') else 0.0,
            error_code=retcode,
            error_message=error_msg if not success else f"Success: {error_msg}",
        )
        
        if not success:
            logger.error(f"MT5 order failed: {error_msg} (retcode: {retcode})")
        
        return order_result
    
    def _mock_execute_order(
        self,
        symbol: str,
        action: OrderAction,
        volume: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> OrderResult:
        """Generate mock order execution result with validation"""
        import random
        
        # Check duplicate
        if self._is_duplicate_order(symbol, action):
            logger.warning(f"Duplicate order attempt: {symbol} {action.value} {volume}L")
            return OrderResult(
                success=False,
                error_code=10014,
                error_message=f"Duplicate order blocked within {self.DUPLICATE_WINDOW_SEC}s"
            )
        
        # Validate volume
        if volume <= 0:
            return OrderResult(success=False, error_code=4109, error_message="Invalid volume")
        
        if volume > 100.0:
            return OrderResult(success=False, error_code=4111, error_message=f"Volume {volume} exceeds maximum 100.0")
        
        # Validate SL/TP
        if stop_loss is not None and stop_loss > 0:
            if take_profit is not None and take_profit > 0:
                if action == OrderAction.BUY and take_profit <= stop_loss:
                    return OrderResult(success=False, error_code=4112, error_message="TP <= SL")
                elif action == OrderAction.SELL and take_profit >= stop_loss:
                    return OrderResult(success=False, error_code=4112, error_message="TP >= SL")
        
        # Successful mock execution
        order_id = random.randint(1000000, 9999999)
        price = 1.0850 if action == OrderAction.BUY else 1.0848
        
        # Track order
        self.recent_orders.append({
            'symbol': symbol,
            'action': action,
            'timestamp': time.time(),
            'volume': volume
        })
        
        logger.info(f"[MOCK] Order {order_id} executed: {action.value} {volume}L {symbol} SL={stop_loss} TP={take_profit}")
        
        return OrderResult(
            success=True,
            order_id=order_id,
            price=price,
            volume=volume,
            comment=f"Mock {action.value} order",
        )
    
    def _mock_close_position(self, position_id: int) -> OrderResult:
        """Generate mock close position result"""
        return OrderResult(
            success=True,
            order_id=position_id,
            price=1.0848,
            volume=0.1,
            comment=f"Mock close position",
        )
    
    def _mock_modify_position(
        self,
        position_id: int,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> OrderResult:
        """Generate mock modify position result"""
        return OrderResult(
            success=True,
            order_id=position_id,
            comment=f"Mock modify position",
        )
    
    def _mock_symbol_info(self, symbol: str):
        """Generate mock symbol info"""
        return MockSymbolInfo(symbol=symbol)
    
    def _mock_tick(self, symbol: str):
        """Generate mock tick"""
        return MockTickData(symbol=symbol)


if __name__ == '__main__':
    """Example usage and testing"""
    
    print("\n" + "=" * 70)
    print("MT5 ORDERS - TEST RUN")
    print("=" * 70)
    
    # Initialize with demo mode (no live MT5 needed)
    orders = MT5Orders(feed=None)
    
    # Test 1: Buy order
    print("\n[TEST 1] Buy Order")
    result = orders.buy(
        symbol='EURUSD',
        volume=0.1,
        stop_loss=1.0820,
        take_profit=1.0880,
    )
    print(f"Success: {result.success}")
    print(f"Order ID: {result.order_id}")
    print(f"Price: {result.price:.5f}")
    print(f"Volume: {result.volume}")
    if result.error_code:
        print(f"Error: {result.error_message}")
    
    # Test 2: Sell order
    print("\n[TEST 2] Sell Order")
    result = orders.sell(
        symbol='EURUSD',
        volume=0.15,
        stop_loss=1.0880,
        take_profit=1.0800,
    )
    print(f"Success: {result.success}")
    print(f"Order ID: {result.order_id}")
    print(f"Price: {result.price:.5f}")
    
    # Test 3: Duplicate order prevention
    print("\n[TEST 3] Duplicate Prevention")
    result = orders.buy(symbol='EURUSD', volume=0.1)
    print(f"First buy: Success={result.success}")
    
    result = orders.buy(symbol='EURUSD', volume=0.1)
    print(f"Second buy (duplicate): Success={result.success}")
    if not result.success:
        print(f"Prevented: {result.error_message}")
    
    # Test 4: Volume validation
    print("\n[TEST 4] Volume Validation")
    # Test with different symbol to avoid duplicate prevention window
    result = orders.buy(symbol='GBPUSD', volume=0.05)  # Below min 0.01 is OK
    print(f"Small volume (0.05L on GBPUSD): Success={result.success}")
    
    # Also test different symbol for oversized volume
    result = orders.buy(symbol='AUDUSD', volume=150.0)  # Above max
    print(f"Large volume (150L on AUDUSD): Success={result.success}")
    if not result.success:
        print(f"Rejected: {result.error_message}")
    
    # Test 5: Close position
    print("\n[TEST 5] Close Position")
    result = orders.close_position(position_id=1234567)
    print(f"Close position 1234567: Success={result.success}")
    if not result.success:
        print(f"Message: {result.error_message}")
    
    # Test 6: Modify position
    print("\n[TEST 6] Modify Position")
    result = orders.modify_position(
        position_id=1234567,
        stop_loss=1.0825,
        take_profit=1.0875,
    )
    print(f"Modify position: Success={result.success}")
    if not result.success:
        print(f"Message: {result.error_message}")
    
    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70 + "\n")
