#!/usr/bin/env python3
"""
MT5 Live Feed Module - Trading Stockfish

Manages MetaTrader5 connection and provides real-time market data feeds.
Handles ticks, candles, and symbol information with automatic reconnection.

Features:
- Persistent MT5 connection management
- Automatic reconnection with exponential backoff
- Multi-timeframe candle fetching
- Tick data with spread calculation
- Data validation and staleness detection
- Structured data objects ready for state_builder
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logging.warning("MetaTrader5 not available - will run in mock mode")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Timeframe mapping
TIMEFRAMES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "H1": 60,
}

MT5_TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1,
    "M5": mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
    "M15": mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
    "H1": mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60,
}


# Data Classes for structured returns
@dataclass
class TickData:
    """Represents a single tick in market data"""

    symbol: str
    bid: float
    ask: float
    bid_volume: int
    ask_volume: int
    last: float
    time: int
    time_msc: int

    @property
    def spread(self) -> float:
        """Calculate spread in pips (for EURUSD, GBPUSD, etc.)"""
        return (self.ask - self.bid) * 10000

    @property
    def mid_price(self) -> float:
        """Midpoint between bid and ask"""
        return (self.bid + self.ask) / 2


@dataclass
class SymbolInfo:
    """Symbol information and trading parameters"""

    symbol: str
    digits: int  # Decimal places (5 for EURUSD)
    point: float  # Minimum price change (0.00001 for EURUSD)
    bid: float  # Current bid
    ask: float  # Current ask
    volume_min: float  # Minimum position size
    volume_max: float  # Maximum position size
    volume_step: float  # Position size increment
    swap_long: float  # Swap for long positions
    swap_short: float  # Swap for short positions
    commission: float  # Commission per trade
    spread: float  # Current spread in pips

    def format_price(self, price: float) -> str:
        """Format price with correct decimal places"""
        format_str = f"{{:.{self.digits}f}}"
        return format_str.format(price)

    def round_lot(self, volume: float) -> float:
        """Round volume to nearest valid lot size"""
        if self.volume_step == 0:
            return volume
        rounded = round(volume / self.volume_step) * self.volume_step
        return max(self.volume_min, min(rounded, self.volume_max))


@dataclass
class CandleData:
    """OHLC candle data for a specific timeframe"""

    symbol: str
    timeframe: str  # 'M1', 'M5', 'M15', 'H1', etc.
    open: float
    high: float
    low: float
    close: float
    volume: int
    time: int  # Unix timestamp
    count: int  # Number of candles fetched

    @property
    def range(self) -> float:
        """High-Low range"""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Open-Close range (candle body)"""
        return abs(self.close - self.open)

    @property
    def direction(self) -> str:
        """Candle direction: 'up', 'down', or 'doji'"""
        if self.close > self.open:
            return "up"
        elif self.close < self.open:
            return "down"
        else:
            return "doji"


class ConnectionStatus(Enum):
    """MT5 connection states"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class LiveFeedError(Exception):
    """Base exception for live feed errors"""

    pass


class ConnectionError(LiveFeedError):
    """MT5 connection error"""

    pass


class DataValidationError(LiveFeedError):
    """Data validation error"""

    pass


class MT5LiveFeed:
    """
    Manages MetaTrader5 connection and data fetching.

    Handles reconnection, data validation, and provides consistent
    structured data objects for state building.
    """

    def __init__(
        self,
        account: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        use_demo: bool = False,
    ):
        """
        Initialize MT5 Live Feed.

        Args:
            account: MT5 account number (optional, uses default if None)
            password: MT5 password (optional)
            server: MT5 server name (optional)
            max_retries: Maximum connection retry attempts
            retry_delay: Initial retry delay in seconds (grows exponentially)
            use_demo: Use mock data instead of live MT5 (for testing)
        """
        self.account = account
        self.password = password
        self.server = server
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_demo = use_demo

        self.status = ConnectionStatus.DISCONNECTED
        self.last_error: Optional[str] = None
        self.last_connection_time: Optional[float] = None
        self.connection_attempts = 0

        # Symbol cache
        self._symbol_cache: Dict[str, SymbolInfo] = {}
        self._cache_timestamp: Dict[str, float] = {}
        self._cache_ttl = 60  # Cache symbol info for 60 seconds

        logger.info("MT5LiveFeed initialized")

        # Only connect if not in demo/mock mode
        if not self.use_demo:
            self.connect()

    def connect(self) -> bool:
        """
        Establish connection to MetaTrader5.

        Returns:
            True if connected, False otherwise
        """
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available - running in mock mode")
            self.status = ConnectionStatus.CONNECTED
            self.last_connection_time = time.time()
            return True

        if self.status == ConnectionStatus.CONNECTED:
            return True

        self.status = ConnectionStatus.CONNECTING
        self.connection_attempts = 0

        while self.connection_attempts < self.max_retries:
            try:
                self.connection_attempts += 1
                delay = min(self.retry_delay * (2**self.connection_attempts), 30)

                logger.debug(
                    f"MT5 connection attempt {self.connection_attempts}/{self.max_retries}"
                )

                if mt5.initialize():
                    logger.info("MT5 connected successfully")
                    self.status = ConnectionStatus.CONNECTED
                    self.last_connection_time = time.time()
                    self.last_error = None
                    return True
                else:
                    error_code, error_msg = mt5.last_error()
                    self.last_error = f"{error_code}: {error_msg}"
                    logger.warning(f"MT5 init failed: {self.last_error}")

                    if self.connection_attempts < self.max_retries:
                        logger.debug(f"Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)

            except Exception as e:
                self.last_error = str(e)
                logger.error(f"Connection exception: {e}")
                if self.connection_attempts < self.max_retries:
                    time.sleep(self.retry_delay)

        self.status = ConnectionStatus.ERROR
        logger.error(f"Failed to connect after {self.max_retries} attempts")
        return False

    def disconnect(self):
        """Safely disconnect from MetaTrader5"""
        if MT5_AVAILABLE:
            try:
                mt5.shutdown()
                logger.info("MT5 disconnected")
            except Exception as e:
                logger.error(f"Disconnect error: {e}")

        self.status = ConnectionStatus.DISCONNECTED

    def is_connected(self) -> bool:
        """Check if currently connected to MT5"""
        if self.use_demo:
            return True
        return self.status == ConnectionStatus.CONNECTED

    def get_tick(self, symbol: str) -> Optional[TickData]:
        """
        Fetch current tick data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')

        Returns:
            TickData object or None if failed
        """
        if self.use_demo:
            return self._mock_tick(symbol)

        if not self.is_connected():
            logger.warning(f"Not connected, cannot fetch tick for {symbol}")
            return None

        try:
            if not MT5_AVAILABLE:
                return None

            tick = mt5.symbol_info_tick(symbol)

            if tick is None:
                logger.warning(f"No tick data available for {symbol}")
                return None

            tick_data = TickData(
                symbol=symbol,
                bid=tick.bid,
                ask=tick.ask,
                bid_volume=tick.bid_volume,
                ask_volume=tick.ask_volume,
                last=tick.last,
                time=tick.time,
                time_msc=tick.time_msc,
            )

            logger.debug(
                f"{symbol} tick: bid={tick.bid:.5f}, ask={tick.ask:.5f}, spread={tick_data.spread:.1f}pips"
            )
            return tick_data

        except Exception as e:
            logger.error(f"Error fetching tick for {symbol}: {e}")
            return None

    def get_symbol_info(
        self, symbol: str, use_cache: bool = True
    ) -> Optional[SymbolInfo]:
        """
        Fetch symbol information and trading parameters.

        Args:
            symbol: Trading symbol
            use_cache: Use cached info if available

        Returns:
            SymbolInfo object or None if failed
        """
        if self.use_demo:
            return self._mock_symbol_info(symbol)

        # Check cache
        if use_cache and symbol in self._symbol_cache:
            cache_age = time.time() - self._cache_timestamp.get(symbol, 0)
            if cache_age < self._cache_ttl:
                logger.debug(f"Using cached symbol info for {symbol}")
                return self._symbol_cache[symbol]

        if not self.is_connected():
            logger.warning(f"Not connected, cannot fetch symbol info for {symbol}")
            return None

        try:
            if not MT5_AVAILABLE:
                return None

            info = mt5.symbol_info(symbol)
            if info is None:
                logger.warning(f"No symbol info available for {symbol}")
                return None

            # Get current tick for bid/ask
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(f"No tick data for {symbol}, using symbol info prices")
                bid = info.bid
                ask = info.ask
            else:
                bid = tick.bid
                ask = tick.ask

            symbol_info = SymbolInfo(
                symbol=symbol,
                digits=info.digits,
                point=info.point,
                bid=bid,
                ask=ask,
                volume_min=info.volume_min,
                volume_max=info.volume_max,
                volume_step=info.volume_step,
                swap_long=info.swap_long,
                swap_short=info.swap_short,
                commission=info.commission,
                spread=(ask - bid) * (10**info.digits),
            )

            # Cache it
            self._symbol_cache[symbol] = symbol_info
            self._cache_timestamp[symbol] = time.time()

            logger.debug(
                f"Symbol info for {symbol}: digits={info.digits}, volume_min={info.volume_min}, volume_max={info.volume_max}"
            )
            return symbol_info

        except Exception as e:
            logger.error(f"Error fetching symbol info for {symbol}: {e}")
            return None

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100,
        offset: int = 0,
    ) -> Optional[List[CandleData]]:
        """
        Fetch candle data for a symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe code ('M1', 'M5', 'M15', 'H1')
            count: Number of candles to fetch
            offset: Offset from current candle (0 = current, 1 = previous, etc.)

        Returns:
            List of CandleData objects or None if failed
        """
        if self.use_demo:
            return self._mock_candles(symbol, timeframe, count)

        if not self.is_connected():
            logger.warning(
                f"Not connected, cannot fetch candles for {symbol} {timeframe}"
            )
            return None

        try:
            if not MT5_AVAILABLE or timeframe not in MT5_TIMEFRAMES:
                logger.error(f"Invalid timeframe: {timeframe}")
                return None

            mt5_tf = MT5_TIMEFRAMES[timeframe]

            # Fetch candles from offset position
            candles = mt5.copy_rates_from_pos(symbol, mt5_tf, offset, count)

            if candles is None or len(candles) == 0:
                logger.warning(f"No candle data for {symbol} {timeframe}")
                return None

            # Convert to CandleData objects
            candle_list = []
            for i, candle in enumerate(candles):
                cd = CandleData(
                    symbol=symbol,
                    timeframe=timeframe,
                    open=float(candle["open"]),
                    high=float(candle["high"]),
                    low=float(candle["low"]),
                    close=float(candle["close"]),
                    volume=int(candle["tick_volume"]),
                    time=int(candle["time"]),
                    count=len(candles),
                )
                candle_list.append(cd)

            logger.debug(
                f"{symbol} {timeframe}: fetched {len(candle_list)} candles, latest close={candles[-1]['close']:.5f}"
            )
            return candle_list

        except Exception as e:
            logger.error(f"Error fetching candles for {symbol} {timeframe}: {e}")
            return None

    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[CandleData]:
        """
        Fetch only the latest candle for a symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe code

        Returns:
            CandleData object (latest candle) or None if failed
        """
        candles = self.get_candles(symbol, timeframe, count=1)
        if candles and len(candles) > 0:
            return candles[0]
        return None

    def get_multitf_candles(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
        count: int = 100,
    ) -> Optional[Dict[str, List[CandleData]]]:
        """
        Fetch candles for multiple timeframes at once.

        Args:
            symbol: Trading symbol
            timeframes: List of timeframe codes (default: ['M1', 'M5', 'M15', 'H1'])
            count: Candles per timeframe

        Returns:
            Dict of {timeframe: List[CandleData]} or None if any failed
        """
        if timeframes is None:
            timeframes = ["M1", "M5", "M15", "H1"]

        results = {}
        for tf in timeframes:
            candles = self.get_candles(symbol, tf, count=count)
            if candles is None:
                logger.warning(f"Failed to fetch {tf} candles for {symbol}")
                results[tf] = None
            else:
                results[tf] = candles

        logger.debug(f"Multi-TF candles fetched for {symbol}: {list(results.keys())}")
        return results

    def validate_tick(self, tick: TickData, max_age_sec: int = 60) -> Tuple[bool, str]:
        """
        Validate tick data for staleness and consistency.

        Args:
            tick: TickData object to validate
            max_age_sec: Maximum age in seconds before considered stale

        Returns:
            Tuple of (is_valid: bool, reason: str)
        """
        if tick is None:
            return False, "Tick is None"

        age = time.time() - tick.time
        if age > max_age_sec:
            return False, f"Tick is stale: {age:.1f}s old"

        if tick.bid >= tick.ask:
            return False, "Invalid bid/ask: bid >= ask"

        if tick.bid <= 0 or tick.ask <= 0:
            return False, "Invalid bid/ask: negative or zero"

        if tick.spread > 100:  # Sanity check for extreme spread
            return False, f"Spread too wide: {tick.spread:.1f} pips"

        return True, "Tick valid"

    def validate_candles(
        self,
        candles: Optional[List[CandleData]],
        min_count: int = 20,
        max_age_sec: int = 300,
    ) -> Tuple[bool, str]:
        """
        Validate candle data.

        Args:
            candles: List of CandleData objects
            min_count: Minimum required candles
            max_age_sec: Maximum age of latest candle

        Returns:
            Tuple of (is_valid: bool, reason: str)
        """
        if candles is None:
            return False, "Candles is None"

        if len(candles) == 0:
            return False, "No candles"

        if len(candles) < min_count:
            return False, f"Insufficient candles: {len(candles)} < {min_count}"

        latest = candles[-1]
        age = time.time() - latest.time
        if age > max_age_sec:
            return False, f"Candles stale: {age:.1f}s old"

        # Check for valid OHLC
        for candle in candles[-5:]:  # Check last 5 candles
            if candle.high < candle.low:
                return False, f"Invalid candle: high < low"
            if candle.open < candle.low or candle.open > candle.high:
                return False, f"Invalid candle: open outside range"
            if candle.close < candle.low or candle.close > candle.high:
                return False, f"Invalid candle: close outside range"

        return True, "Candles valid"

    def get_connection_status(self) -> Dict:
        """Get current connection status and diagnostics"""
        uptime = None
        if self.last_connection_time:
            uptime = time.time() - self.last_connection_time

        return {
            "status": self.status.value,
            "is_connected": self.is_connected(),
            "connection_attempts": self.connection_attempts,
            "last_error": self.last_error,
            "uptime_seconds": uptime,
            "use_demo": self.use_demo,
        }

    def _mock_tick(self, symbol: str) -> TickData:
        """Generate mock tick data for testing"""
        base_prices = {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2750,
            "USDJPY": 149.50,
            "AUDUSD": 0.6850,
        }

        base = base_prices.get(symbol, 1.0850)
        return TickData(
            symbol=symbol,
            bid=base,
            ask=base + 0.0002,
            bid_volume=1000,
            ask_volume=1000,
            last=base,
            time=int(time.time()),
            time_msc=int(time.time() * 1000),
        )

    def _mock_symbol_info(self, symbol: str) -> SymbolInfo:
        """Generate mock symbol info for testing"""
        return SymbolInfo(
            symbol=symbol,
            digits=5,
            point=0.00001,
            bid=1.0850,
            ask=1.0852,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            swap_long=-5.0,
            swap_short=-3.0,
            commission=0.0,
            spread=2.0,
        )

    def _mock_candles(
        self, symbol: str, timeframe: str, count: int
    ) -> List[CandleData]:
        """Generate mock candle data for testing"""
        candles = []
        base_price = 1.0850
        current_time = int(time.time())

        for i in range(count):
            # Generate realistic OHLC
            open_p = base_price + (i * 0.0001)
            close_p = open_p + 0.00005
            high_p = max(open_p, close_p) + 0.00008
            low_p = min(open_p, close_p) - 0.00005

            candles.append(
                CandleData(
                    symbol=symbol,
                    timeframe=timeframe,
                    open=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    volume=1000,
                    time=current_time - (count - i) * TIMEFRAMES[timeframe] * 60,
                    count=count,
                )
            )

        return candles


# Global instance
_live_feed: Optional[MT5LiveFeed] = None


def initialize_feed(use_demo: bool = False) -> MT5LiveFeed:
    """Initialize global live feed instance"""
    global _live_feed
    _live_feed = MT5LiveFeed(use_demo=use_demo)
    return _live_feed


def get_feed() -> MT5LiveFeed:
    """Get global live feed instance"""
    global _live_feed
    if _live_feed is None:
        _live_feed = initialize_feed()
    return _live_feed


if __name__ == "__main__":
    """Example usage and testing"""

    print("\n" + "=" * 70)
    print("MT5 LIVE FEED - TEST RUN")
    print("=" * 70)

    # Initialize with demo mode
    feed = MT5LiveFeed(use_demo=True)

    # Test 1: Connection status
    print("\n[TEST 1] Connection Status")
    status = feed.get_connection_status()
    print(f"Status: {status['status']}")
    print(f"Connected: {status['is_connected']}")

    # Test 2: Fetch tick data
    print("\n[TEST 2] Fetch Tick Data")
    tick = feed.get_tick("EURUSD")
    if tick:
        print(f"Symbol: {tick.symbol}")
        print(f"Bid: {tick.bid:.5f}")
        print(f"Ask: {tick.ask:.5f}")
        print(f"Spread: {tick.spread:.2f} pips")
        print(f"Mid: {tick.mid_price:.5f}")

        # Validate
        is_valid, reason = feed.validate_tick(tick)
        print(f"Validation: {reason}")

    # Test 3: Fetch symbol info
    print("\n[TEST 3] Symbol Info")
    symbol_info = feed.get_symbol_info("EURUSD")
    if symbol_info:
        print(f"Symbol: {symbol_info.symbol}")
        print(f"Digits: {symbol_info.digits}")
        print(f"Point: {symbol_info.point}")
        print(f"Volume Min: {symbol_info.volume_min}")
        print(f"Volume Max: {symbol_info.volume_max}")
        print(f"Spread: {symbol_info.spread:.2f} pips")

    # Test 4: Fetch candles
    print("\n[TEST 4] Fetch Candles (H1)")
    candles = feed.get_candles("EURUSD", "H1", count=5)
    if candles:
        print(f"Fetched {len(candles)} candles")
        latest = candles[-1]
        print(f"Latest H1:")
        print(
            f"  OHLC: {latest.open:.5f} / {latest.high:.5f} / {latest.low:.5f} / {latest.close:.5f}"
        )
        print(f"  Volume: {latest.volume}")
        print(f"  Direction: {latest.direction}")

        # Validate
        is_valid, reason = feed.validate_candles(candles, min_count=3)
        print(f"Validation: {reason}")

    # Test 5: Multi-timeframe candles
    print("\n[TEST 5] Multi-Timeframe Candles")
    mtf = feed.get_multitf_candles(
        "EURUSD", timeframes=["M1", "M5", "M15", "H1"], count=3
    )
    if mtf:
        for tf, candles in mtf.items():
            if candles:
                print(
                    f"{tf}: {len(candles)} candles, latest close={candles[-1].close:.5f}"
                )
            else:
                print(f"{tf}: No data")

    # Test 6: Disconnect
    print("\n[TEST 6] Disconnect")
    feed.disconnect()
    status = feed.get_connection_status()
    print(f"Status after disconnect: {status['status']}")

    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70 + "\n")
