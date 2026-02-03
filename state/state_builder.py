#!/usr/bin/env python3
"""
State Builder Module - Trading Stockfish

Builds structured market state dictionaries from live MetaTrader5 data.
Fetches ticks, spreads, candles across multiple timeframes, calculates indicators,
and assembles a complete state snapshot for decision-making.

State Schema:
{
    'timestamp': float,           # Unix timestamp
    'symbol': str,                # e.g., 'EURUSD'
    'tick': {
        'bid': float,
        'ask': float,
        'spread': float,          # ask - bid in pips
        'last_tick_time': int,    # seconds since epoch
    },
    'candles': {
        'M1': {...},              # 1-minute candles with indicators
        'M5': {...},
        'M15': {...},
        'H1': {...},
    },
    'indicators': {
        'rsi_14': float,          # Relative Strength Index
        'sma_50': float,          # Simple Moving Average
        'sma_200': float,
        'atr_14': float,          # Average True Range
        'volatility': float,      # Current volatility metric
    },
    'trend': {
        'regime': str,            # 'uptrend', 'downtrend', 'sideways'
        'strength': float,        # 0-1, confidence in trend
    },
    'sentiment': {
        'score': float,           # -1 to 1, -1 = bearish, 1 = bullish
        'confidence': float,      # 0-1, confidence in sentiment
        'source': str,            # 'news', 'manual', 'placeholder'
    },
    'health': {
        'is_stale': bool,         # True if data is older than threshold
        'last_update': float,     # Unix timestamp of last successful update
        'errors': list,           # List of warnings/non-fatal errors
    }
}
"""


import logging
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.amd_features import AMDFeatures
from engine.liquidity_metrics import (
    compute_liquidity_score,
    compute_spread,
    detect_liquidity_stress,
)

# Microstructure imports
from engine.order_book_model import OrderBookModel
from engine.order_flow_features import OrderFlowFeatures
from engine.trend_structure import compute_trend_structure
from engine.volatility_features import VolatilityFeatures
from engine.volatility_utils import compute_atr
from session_regime import compute_session_regime

try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logging.warning("MetaTrader5 not available - will run in mock mode")


def _session_modifiers(session_label: str) -> Dict[str, float]:
    """Return deterministic session multipliers used by downstream policy/evaluator."""

    base = {
        "volatility_scale": 1.0,
        "liquidity_scale": 1.0,
        "trade_freq_scale": 1.0,
        "risk_scale": 1.0,
    }
    label = (session_label or "").upper()
    if label == "GLOBEX":
        base.update(
            {"volatility_scale": 0.8, "liquidity_scale": 0.7, "risk_scale": 0.9}
        )
    elif label == "PREMARKET":
        base.update({"volatility_scale": 0.9, "liquidity_scale": 0.8})
    elif label == "RTH_OPEN":
        base.update(
            {
                "volatility_scale": 1.2,
                "liquidity_scale": 1.2,
                "trade_freq_scale": 1.3,
                "risk_scale": 1.1,
            }
        )
    elif label == "MIDDAY":
        base.update({"volatility_scale": 0.9, "liquidity_scale": 1.0})
    elif label == "POWER_HOUR":
        base.update(
            {"volatility_scale": 1.3, "liquidity_scale": 0.9, "risk_scale": 1.2}
        )
    elif label == "CLOSE":
        base.update({"volatility_scale": 1.1, "liquidity_scale": 0.8})
    return base


# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
_last_tick_timestamp: Optional[float] = None

# Configuration constants
STALE_TICK_THRESHOLD = 60  # seconds - warn if tick older than this
STALE_CANDLE_THRESHOLD = 300  # seconds - warn if candle older than this
MT5_INIT_TIMEOUT = 5  # seconds
MT5_CONNECTION_RETRIES = 3
MT5_RETRY_DELAY = 1  # seconds

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


class StateBuilderError(Exception):
    """Base exception for state builder errors"""

    pass


class MT5ConnectionError(StateBuilderError):
    """Raised when MT5 connection fails"""

    pass


def initialize_mt5(timeout: int = MT5_INIT_TIMEOUT) -> bool:
    """
    Initialize MetaTrader5 connection with retry logic.

    Args:
        timeout: Initialization timeout in seconds

    Returns:
        True if initialization successful, False otherwise
    """
    if not MT5_AVAILABLE:
        logger.warning("MT5 not available - running in mock mode")
        return False

    for attempt in range(MT5_CONNECTION_RETRIES):
        try:
            logger.debug(
                f"MT5 initialization attempt {attempt + 1}/{MT5_CONNECTION_RETRIES}"
            )
            if mt5.initialize():
                logger.info("MetaTrader5 initialized successfully")
                return True
            else:
                error_code, error_msg = mt5.last_error()
                logger.warning(
                    f"MT5 init failed (attempt {attempt + 1}): {error_code} - {error_msg}"
                )
                if attempt < MT5_CONNECTION_RETRIES - 1:
                    time.sleep(MT5_RETRY_DELAY)
        except Exception as e:
            logger.error(f"MT5 initialization exception (attempt {attempt + 1}): {e}")
            if attempt < MT5_CONNECTION_RETRIES - 1:
                time.sleep(MT5_RETRY_DELAY)

    logger.error("MT5 initialization failed after all retry attempts")
    return False


def shutdown_mt5():
    """Safely shutdown MetaTrader5 connection"""
    if MT5_AVAILABLE:
        try:
            mt5.shutdown()
            logger.info("MetaTrader5 shutdown complete")
        except Exception as e:
            logger.error(f"MT5 shutdown error: {e}")


def fetch_tick_data(symbol: str, use_demo: bool = False) -> Optional[Dict]:
    """
    Fetch current tick data (bid, ask, spread).

    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        use_demo: If True, use mock data instead of live MT5

    Returns:
        Dict with bid, ask, spread, timestamp. None if failed.
    """
    if use_demo:
        raise RuntimeError("Synthetic/demo tick data is forbidden; use live MT5")
    if not MT5_AVAILABLE:
        raise RuntimeError("MT5 is not available; cannot fabricate tick data")

    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.warning(f"No tick data for {symbol}")
            return None

        bid = tick.bid
        ask = tick.ask
        spread = (ask - bid) * 10000  # Convert to pips for EURUSD-like pairs

        tick_data = {
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "last_tick_time": tick.time,
            "timestamp": time.time(),
        }

        logger.debug(f"{symbol} tick: bid={bid}, ask={ask}, spread={spread:.2f} pips")
        return tick_data

    except Exception as e:
        logger.error(f"Error fetching tick data for {symbol}: {e}")
        return None


def fetch_candles(
    symbol: str, timeframe: str, count: int = 100, use_demo: bool = False
) -> Optional[Dict]:
    """
    Fetch candle data for a specific timeframe.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        timeframe: Timeframe code ('M1', 'M5', 'M15', 'H1')
        count: Number of candles to fetch
        use_demo: If True, use mock data instead of live MT5

    Returns:
        Dict with OHLC data and indicators. None if failed.
    """
    if use_demo:
        raise RuntimeError("Synthetic/demo candle data is forbidden; use live MT5")
    if not MT5_AVAILABLE:
        raise RuntimeError("MT5 is not available; cannot fabricate candle data")

    try:
        if timeframe not in MT5_TIMEFRAMES:
            logger.error(f"Unknown timeframe: {timeframe}")
            return None

        mt5_tf = MT5_TIMEFRAMES[timeframe]
        candles = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)

        if candles is None or len(candles) == 0:
            logger.warning(f"No candle data for {symbol} {timeframe}")
            return None

        # Convert to DataFrame-like structure
        close_prices = np.array([c["close"] for c in candles])
        high_prices = np.array([c["high"] for c in candles])
        low_prices = np.array([c["low"] for c in candles])

        # Calculate indicators
        rsi = _calculate_rsi(close_prices, period=14)
        sma_50 = _calculate_sma(close_prices, period=50)
        sma_200 = _calculate_sma(close_prices, period=200)
        atr = _calculate_atr(high_prices, low_prices, close_prices, period=14)

        last_candle = candles[-1]

        candle_data = {
            "timeframe": timeframe,
            "count": len(candles),
            "latest": {
                "open": float(last_candle["open"]),
                "high": float(last_candle["high"]),
                "low": float(last_candle["low"]),
                "close": float(last_candle["close"]),
                "volume": int(last_candle["tick_volume"]),
                "time": int(last_candle["time"]),
            },
            "indicators": {
                "rsi_14": float(rsi) if not np.isnan(rsi) else None,
                "sma_50": float(sma_50) if not np.isnan(sma_50) else None,
                "sma_200": float(sma_200) if not np.isnan(sma_200) else None,
                "atr_14": float(atr) if not np.isnan(atr) else None,
            },
        }

        logger.debug(
            f"{symbol} {timeframe}: close={last_candle['close']}, rsi={rsi:.2f}, atr={atr:.2f}"
        )
        return candle_data

    except Exception as e:
        logger.error(f"Error fetching candles for {symbol} {timeframe}: {e}")
        return None


def _calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return np.nan

    deltas = np.diff(prices)
    seed = deltas[: period + 1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period

    rs = up / down if down != 0 else 0
    rsi = 100.0 - (100.0 / (1.0 + rs))

    for d in deltas[period + 1 :]:
        up = (up * (period - 1) + (d if d > 0 else 0)) / period
        down = (down * (period - 1) + (-d if d < 0 else 0)) / period
        rs = up / down if down != 0 else 0
        rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def _calculate_sma(prices: np.ndarray, period: int = 50) -> float:
    """Calculate Simple Moving Average (returns latest value)"""
    if len(prices) < period:
        return np.nan
    return np.mean(prices[-period:])


def _calculate_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> float:
    """Calculate Average True Range"""
    if len(high) < period:
        return np.nan

    tr = np.maximum(
        high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])
    )
    atr = np.mean(tr[-period:])
    return atr


def detect_trend_regime(
    rsi: float, sma_50: float, sma_200: float, close: float
) -> Dict:
    """
    Detect trend regime based on indicators.

    Args:
        rsi: RSI value (0-100)
        sma_50: 50-period moving average
        sma_200: 200-period moving average
        close: Latest close price

    Returns:
        Dict with regime type and strength
    """
    regime = "sideways"
    strength = 0.0

    if sma_50 is None or sma_200 is None or close is None:
        logger.warning("Cannot determine trend - missing data")
        return {"regime": regime, "strength": strength}

    # Trend detection logic
    if sma_50 > sma_200:
        regime = "uptrend"
        # Strength based on distance and RSI
        distance = (sma_50 - sma_200) / sma_200
        strength = min(0.9, 0.5 + distance * 100)  # Cap at 0.9
        if rsi > 70:
            strength = min(strength, 0.7)  # Overbought reduces confidence
    elif sma_50 < sma_200:
        regime = "downtrend"
        distance = (sma_200 - sma_50) / sma_200
        strength = min(0.9, 0.5 + distance * 100)
        if rsi < 30:
            strength = min(strength, 0.7)  # Oversold reduces confidence
    else:
        regime = "sideways"
        strength = 0.5

    return {"regime": regime, "strength": strength}


def interpret_news_sentiment(headlines: Optional[List[str]] = None) -> Dict:
    """
    Placeholder for news sentiment interpretation.

    In production, this would integrate with an LLM (OpenAI, Claude, etc.)
    to analyze news headlines and market sentiment.

    Args:
        headlines: List of news headlines (optional)

    Returns:
        Dict with sentiment score (-1 to 1), confidence, and source
    """
    # Placeholder implementation - returns neutral sentiment
    logger.debug("News sentiment interpretation (placeholder)")

    sentiment_data = {
        "score": 0.0,  # Range: -1 (bearish) to 1 (bullish), 0 (neutral)
        "confidence": 0.0,  # Range: 0 to 1
        "source": "placeholder",
        "headlines_processed": 0,
    }

    if headlines and len(headlines) > 0:
        # TODO: Integrate with LLM API (OpenAI, Claude, Gemini)
        # For now, just count positive/negative keywords
        sentiment_data["headlines_processed"] = len(headlines)
        logger.debug(f"Processed {len(headlines)} headlines (mock)")

    return sentiment_data


def calculate_volatility(atr: float, close: float) -> float:
    """
    Calculate volatility metric (ATR-based).

    Args:
        atr: Average True Range value
        close: Current close price

    Returns:
        Volatility as percentage
    """
    if atr is None or close is None or close == 0:
        return 0.0

    volatility = (atr / close) * 100  # Convert to percentage
    return volatility


def check_data_health(tick_time: int, candle_time: int) -> Dict:
    """
    Check health of market data (staleness, gaps, etc.).

    Args:
        tick_time: Timestamp of last tick
        candle_time: Timestamp of last candle

    Returns:
        Dict with health status and warnings
    """
    current_time = time.time()
    tick_age = current_time - tick_time
    candle_age = current_time - candle_time

    health = {
        "is_stale": False,
        "last_update": current_time,
        "errors": [],
    }

    if tick_age > STALE_TICK_THRESHOLD:
        health["is_stale"] = True
        health["errors"].append(f"Stale tick: {tick_age:.1f}s old")
        logger.warning(f"Stale tick detected: {tick_age:.1f} seconds old")

    if candle_age > STALE_CANDLE_THRESHOLD:
        health["is_stale"] = True
        health["errors"].append(f"Stale candle: {candle_age:.1f}s old")
        logger.warning(f"Stale candle detected: {candle_age:.1f} seconds old")

    return health


def build_state(
    symbol: str = "EURUSD",
    use_demo: bool = False,
    order_book_events: Optional[list] = None,
) -> Optional[Dict]:
    """
    Build complete market state dictionary.

    Fetches live data from MetaTrader5, calculates indicators, detects trends,
    and assembles a structured state for decision-making.

    Args:
        symbol: Trading symbol (default: 'EURUSD')
        use_demo: If True, use mock data instead of live MT5 (for testing)

    Returns:
        Complete state dictionary. None if critical data fetch failed.
    """

    logger.info(f"Building state for {symbol} (demo={use_demo})")

    # Fetch tick data
    tick_data = fetch_tick_data(symbol, use_demo=use_demo)
    if tick_data is None:
        logger.error(f"Failed to fetch tick data for {symbol}")
        return None
    _ensure_valid_tick_data(tick_data)

    # Fetch candles for all timeframes
    candles = {}
    candle_times = []
    for timeframe in TIMEFRAMES.keys():
        candle = fetch_candles(symbol, timeframe, count=100, use_demo=use_demo)
        if candle is not None:
            candles[timeframe] = candle
            candle_times.append(candle["latest"]["time"])
        else:
            logger.warning(f"Failed to fetch {timeframe} candles for {symbol}")
            candles[timeframe] = None

    # Extract indicators from H1 (primary decision timeframe)
    indicators = {
        "rsi_14": None,
        "sma_50": None,
        "sma_200": None,
        "atr_14": None,
        "volatility": None,
    }

    if candles["H1"] and candles["H1"]["indicators"]:
        indicators = candles["H1"]["indicators"].copy()
        # Calculate volatility
        atr_value = compute_atr(indicators.get("atr_14"), window=14)
        indicators["atr_14"] = atr_value
        indicators["volatility"] = calculate_volatility(atr_value, tick_data["bid"])
    _ensure_valid_volatility(indicators)

    # Detect trend
    trend = detect_trend_regime(
        indicators["rsi_14"],
        indicators["sma_50"],
        indicators["sma_200"],
        tick_data["bid"],
    )

    # Get sentiment (placeholder)
    sentiment = interpret_news_sentiment()

    # Check data health
    latest_candle_time = max(candle_times) if candle_times else int(time.time())
    health = check_data_health(tick_data["last_tick_time"], latest_candle_time)

    # Order book integration (Phase v4.0-A only)
    vol_features = VolatilityFeatures(window=50, use_microstructure_realism=True)
    order_book = None
    order_flow_features = None
    amd_detector = AMDFeatures(window=50)
    mid_prices: List[float] = []
    volumes: List[float] = []
    if order_book_events:
        order_book = OrderBookModel(depth=5)
        order_flow_features = OrderFlowFeatures(lookback=10)
        for event in order_book_events:
            order_book.update_from_event(event)
            # For order flow, use the current book snapshot and event
            order_flow_features.update(order_book.get_depth_snapshot(), event)
            best_bid, best_ask = order_book.get_best_bid_ask()
            if best_bid is not None and best_ask is not None:
                mid_prices.append((best_bid + best_ask) / 2)
                vol_features.compute(mid_prices[-1])
            if event.get("type") == "trade":
                volumes.append(float(event.get("size", 1.0)))
    if not mid_prices and tick_data:
        mid_prices.append((tick_data.get("bid", 0.0) + tick_data.get("ask", 0.0)) / 2)
        vol_features.compute(mid_prices[-1], candle_data=indicators)

    # Session regime + deterministic modifiers for downstream policy/evaluator
    session_label = "UNKNOWN"
    try:
        ts_source = float(tick_data.get("last_tick_time") or time.time())
        session_label = compute_session_regime(ts_source).value
    except Exception:
        session_label = "UNKNOWN"
    session_modifiers = _session_modifiers(session_label)

    # Assemble complete state
    state = {
        "timestamp": time.time(),
        "symbol": symbol,
        "tick": tick_data,
        "candles": candles,
        "indicators": indicators,
        "trend": trend,
        "sentiment": sentiment,
        "health": health,
        "session_regime": session_label,
        "session_context": {
            "session": session_label,
            "modifiers": session_modifiers,
        },
    }

    if order_book:
        ob_snapshot = order_book.get_depth_snapshot()
        _ensure_depth(ob_snapshot)
        state["order_book"] = ob_snapshot
        # Attach liquidity metrics
        state["spread"] = compute_spread(ob_snapshot)
        state["liquidity_score"] = compute_liquidity_score(ob_snapshot)
        state["liquidity_stress_flags"] = detect_liquidity_stress(ob_snapshot)
        _ensure_liquidity_metrics(state)
    if order_flow_features:
        state["order_flow_features"] = order_flow_features.compute_features()

    atr_hint = compute_atr(indicators.get("atr_14"), window=14)
    state["volatility_state"] = vol_features.compute(
        mid_prices[-1] if mid_prices else tick_data.get("bid", 0.0),
        candle_data={
            "atr": atr_hint,
            "atr_14": atr_hint,
        },
    )

    trend_struct = compute_trend_structure(
        mid_prices,
        highs=(
            [candle["latest"]["high"] for candle in candles.values() if candle]
            if candles
            else None
        ),
        lows=(
            [candle["latest"]["low"] for candle in candles.values() if candle]
            if candles
            else None
        ),
        window=20,
        volatility_state=state["volatility_state"],
    )
    state.update(trend_struct)

    liquidity_hint = {"liquidity_shock": bool(state.get("liquidity_stress_flags"))}
    state["amd_state"] = amd_detector.compute(
        price_series=mid_prices,
        volume_series=volumes,
        liquidity_state=liquidity_hint,
    )

    logger.info(f"State built successfully for {symbol}")
    return state


def validate_state(state: Optional[Dict]) -> Tuple[bool, List[str]]:
    """
    Validate state dictionary structure and contents.

    Args:
        state: State dictionary to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    if state is None:
        return False, ["State is None"]

    required_keys = [
        "timestamp",
        "symbol",
        "tick",
        "candles",
        "indicators",
        "trend",
        "sentiment",
        "health",
    ]
    for key in required_keys:
        if key not in state:
            errors.append(f"Missing required key: {key}")

    if "tick" in state and state["tick"]:
        tick_keys = ["bid", "ask", "spread", "last_tick_time"]
        for key in tick_keys:
            if key not in state["tick"]:
                errors.append(f"Missing tick key: {key}")
        bid = state["tick"].get("bid")
        ask = state["tick"].get("ask")
        spread = state["tick"].get("spread")
        if bid is None or ask is None or spread is None:
            errors.append("Tick missing bid/ask/spread values")
        else:
            if bid <= 0 or ask <= 0:
                errors.append("Tick has non-positive prices")
            if ask <= bid:
                errors.append("Tick spread not positive")
            if spread <= 0:
                errors.append("Spread must be positive")

    if "health" in state and state["health"].get("is_stale"):
        errors.append("State data is stale")

    is_valid = len(errors) == 0
    return is_valid, errors


def _mock_tick_data(symbol: str) -> Dict:
    raise RuntimeError("Mock tick data is disabled; real market data required")


def _ensure_valid_tick_data(tick_data: Dict) -> None:
    global _last_tick_timestamp
    required = ["bid", "ask", "spread", "last_tick_time", "timestamp"]
    for field in required:
        if tick_data.get(field) is None:
            raise StateBuilderError(f"Tick missing required field {field}")

    bid = float(tick_data["bid"])
    ask = float(tick_data["ask"])
    spread = float(tick_data["spread"])
    ts = float(tick_data["last_tick_time"])

    if any(math.isnan(val) or math.isinf(val) for val in [bid, ask, spread, ts]):
        raise StateBuilderError("Tick contains NaN/inf values")
    if bid <= 0 or ask <= 0:
        raise StateBuilderError("Tick has non-positive prices")
    if ask <= bid:
        raise StateBuilderError("Tick spread not positive (ask <= bid)")
    if spread <= 0:
        raise StateBuilderError("Tick spread must be positive")
    if _last_tick_timestamp is not None and ts <= _last_tick_timestamp:
        raise StateBuilderError("Tick timestamp not monotonic increasing")
    _last_tick_timestamp = ts


def _ensure_valid_volatility(indicators: Dict) -> None:
    vol = indicators.get("volatility")
    if vol is None:
        return
    if math.isnan(vol) or math.isinf(vol) or vol < 0:
        raise StateBuilderError("Volatility metric invalid")


def _ensure_depth(order_book_snapshot: Dict) -> None:
    bids = (
        order_book_snapshot.get("bids")
        if isinstance(order_book_snapshot, dict)
        else None
    )
    asks = (
        order_book_snapshot.get("asks")
        if isinstance(order_book_snapshot, dict)
        else None
    )
    if not bids or not asks:
        raise StateBuilderError("Order book depth missing bids/asks")


def _ensure_liquidity_metrics(state: Dict) -> None:
    for key in ["spread", "liquidity_score"]:
        val = state.get(key)
        if val is None or math.isnan(val) or math.isinf(val):
            raise StateBuilderError(f"Invalid liquidity metric: {key}")
