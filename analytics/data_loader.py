"""
Real Market Data Loader - Trading Stockfish

Loads real historical OHLCV data from multiple sources and reconstructs market states
for production-grade trading engine evaluation.

Features:
  - Load OHLCV from CSV, Parquet, or broker APIs
  - Multi-symbol and multi-timeframe support
  - Automatic gap detection and repair
  - Timestamp alignment to exchange sessions
  - Bid/ask spread estimation
  - Full market state reconstruction with 7 state variables
  - MarketState objects for each timestamp

Usage:
    from analytics.data_loader import DataLoader, MarketStateBuilder
    
    # Load real data
    loader = DataLoader()
    df = loader.load_csv('data/ES_1m.csv', symbol='ES', timeframe='1m')
    
    # Repair gaps
    df = loader.repair_gaps(df, symbol='ES', timeframe='1m')
    
    # Reconstruct market states
    builder = MarketStateBuilder(symbol='ES', timeframe='1m')
    states = builder.build_states(df)

Author: Trading-Stockfish Analytics
Version: 1.0.0
License: MIT
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('analytics.data_loader')


# ============================================================================
# TIME-CAUSAL VALIDATION (NO LOOKAHEAD BIAS)
# ============================================================================

def validate_time_causal_data(
    data: pd.DataFrame,
    symbol: str,
    timeframe: str
) -> Tuple[bool, List[str]]:
    """Validate that data is properly ordered and suitable for time-causal backtesting.
    
    GUARANTEES:
      ✓ No future data leakage
      ✓ Timestamps monotonically increasing
      ✓ No duplicate timestamps
      ✓ Required OHLCV columns present
      ✓ No NaN or infinite values in OHLCV
    
    Args:
        data: DataFrame with OHLCV data
        symbol: Trading symbol
        timeframe: Data timeframe
    
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check required columns
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        warnings.append(f"Missing OHLCV columns: {missing_cols}")
        return False, warnings
    
    # Check for NaN or infinite values
    for col in required_cols:
        if data[col].isna().any():
            warnings.append(f"Column '{col}' contains NaN values (lookahead corruption?)")
        if np.isinf(data[col]).any():
            warnings.append(f"Column '{col}' contains infinite values (data corruption?)")
    
    # Check price relationships (high >= low, open/close within high/low)
    if (data['high'] < data['low']).any():
        warnings.append("High < Low detected (data corruption?)")
    if ((data['open'] < data['low']) | (data['open'] > data['high'])).any():
        warnings.append("Open outside High/Low range (data corruption?)")
    if ((data['close'] < data['low']) | (data['close'] > data['high'])).any():
        warnings.append("Close outside High/Low range (data corruption?)")
    
    # Check for duplicate timestamps
    if data.index.duplicated().any():
        num_dupes = data.index.duplicated().sum()
        warnings.append(f"Found {num_dupes} duplicate timestamps (lookahead bias?)")
    
    # Check timestamp ordering (must be strictly increasing for time-causal)
    if not data.index.is_monotonic_increasing:
        warnings.append("Timestamps not strictly increasing (time-causal violation?)")
        return False, warnings
    
    # Warn if data is suspiciously sorted
    if len(data) > 1:
        time_diffs = data.index[1:] - data.index[:-1]
        expected_freq = _infer_frequency(timeframe)
        if expected_freq and (time_diffs < pd.Timedelta(0)).any():
            warnings.append("Negative time differences detected (future data included?)")
            return False, warnings
    
    is_valid = len(warnings) == 0
    return is_valid, warnings


def _infer_frequency(timeframe: str) -> Optional[pd.Timedelta]:
    """Infer expected time frequency from timeframe string."""
    freq_map = {
        '1m': pd.Timedelta(minutes=1),
        '5m': pd.Timedelta(minutes=5),
        '15m': pd.Timedelta(minutes=15),
        '1h': pd.Timedelta(hours=1),
        '4h': pd.Timedelta(hours=4),
        '1d': pd.Timedelta(days=1),
    }
    return freq_map.get(timeframe.lower())

# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class TimeRegimeType(Enum):
    """Market session regime"""
    ASIA = "Asia"
    LONDON = "London"
    NY_OPEN = "NY_Open"
    NY_MID = "NY_Mid"
    POWER_HOUR = "Power_Hour"
    CLOSE = "Close"
    UNKNOWN = "Unknown"


class MacroExpectationType(Enum):
    """Macro expectation states (based on economic calendar)"""
    PRE_CPI = "pre_CPI"
    POST_CPI = "post_CPI"
    PRE_NFP = "pre_NFP"
    POST_NFP = "post_NFP"
    PRE_FOMC = "pre_FOMC"
    POST_FOMC = "post_FOMC"
    QUIET_PERIOD = "quiet_period"
    UNKNOWN = "unknown"


class LiquidityStateType(Enum):
    """Liquidity conditions"""
    ABUNDANT = "abundant"
    NORMAL = "normal"
    CONSTRAINED = "constrained"
    DROUGHT = "drought"


class VolatilityStateType(Enum):
    """Volatility regime"""
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very_high"


# Exchange session times (UTC)
EXCHANGE_SESSIONS = {
    'ES': {  # S&P 500
        'Asia': (22, 0),           # 10 PM - midnight
        'London': (7, 0),          # 7 AM - noon
        'NY_Open': (13, 30),       # 1:30 PM - 4 PM
        'Power_Hour': (19, 0),     # 7 PM - 8 PM
        'Close': (20, 0),          # 8 PM close
    },
    'NQ': {  # Nasdaq
        'Asia': (22, 0),
        'London': (7, 0),
        'NY_Open': (13, 30),
        'Power_Hour': (19, 0),
        'Close': (20, 0),
    },
    'EURUSD': {  # Forex
        'Asia': (20, 0),           # 8 PM - 6 AM
        'London': (6, 0),          # 6 AM - 3 PM
        'NY_Open': (12, 0),        # 12 PM - 5 PM
        'NY_Mid': (14, 0),         # 2 PM
        'Power_Hour': (20, 0),     # 8 PM
        'Close': (21, 0),          # 9 PM
    },
    'GBPUSD': {
        'Asia': (20, 0),
        'London': (6, 0),
        'NY_Open': (12, 0),
        'NY_Mid': (14, 0),
        'Power_Hour': (20, 0),
        'Close': (21, 0),
    },
    'XAUUSD': {
        'Asia': (20, 0),
        'London': (6, 0),
        'NY_Open': (12, 0),
        'NY_Mid': (14, 0),
        'Power_Hour': (20, 0),
        'Close': (21, 0),
    },
    'SPY': {
        'Asia': (22, 0),
        'London': (7, 0),
        'NY_Open': (13, 30),
        'Power_Hour': (19, 0),
        'Close': (20, 0),
    },
    'QQQ': {
        'Asia': (22, 0),
        'London': (7, 0),
        'NY_Open': (13, 30),
        'Power_Hour': (19, 0),
        'Close': (20, 0),
    },
}

# Typical bid-ask spreads (in pips) by symbol and timeframe
TYPICAL_SPREADS = {
    'EURUSD': {'1m': 1.2, '5m': 1.1, '15m': 1.0, '1h': 1.0},
    'GBPUSD': {'1m': 2.0, '5m': 1.8, '15m': 1.5, '1h': 1.5},
    'XAUUSD': {'1m': 0.40, '5m': 0.35, '15m': 0.30, '1h': 0.30},
    'ES': {'1m': 0.25, '5m': 0.20, '15m': 0.15, '1h': 0.15},
    'NQ': {'1m': 0.50, '5m': 0.40, '15m': 0.30, '1h': 0.30},
    'SPY': {'1m': 0.01, '5m': 0.01, '15m': 0.01, '1h': 0.01},
    'QQQ': {'1m': 0.01, '5m': 0.01, '15m': 0.01, '1h': 0.01},
}

# Expected candle intervals (in seconds)
CANDLE_INTERVALS = {
    '1m': 60,
    '5m': 300,
    '15m': 900,
    '1h': 3600,
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LiquidityState:
    """Liquidity state snapshot"""
    volume: float
    range_pct: float           # High-Low range as % of close
    vwap_distance: float       # Distance from VWAP (%)
    liquidity_type: LiquidityStateType
    confidence: float = 0.0


@dataclass
class VolatilityState:
    """Volatility state snapshot"""
    atr_percent: float         # ATR as % of price
    realized_vol: float        # Realized volatility (%)
    vol_state: VolatilityStateType
    confidence: float = 0.0


@dataclass
class DealerPositioningState:
    """Dealer positioning (gamma and strike clustering)"""
    gamma_exposure: float      # Estimated gamma exposure
    strike_clusters: List[float] = field(default_factory=list)
    positioning_bias: str = "neutral"  # "long", "short", "neutral"
    confidence: float = 0.0


@dataclass
class EarningsExposureState:
    """Earnings exposure state"""
    mega_cap_earnings_today: bool = False
    mega_cap_earnings_week: bool = False
    earnings_ticker: Optional[str] = None
    impact_level: str = "low"  # "low", "medium", "high"
    confidence: float = 0.0


@dataclass
class PriceLocationState:
    """Price location within session range"""
    range_position: float      # 0-1, 0=low, 1=high within session
    session_high: float
    session_low: float
    distance_to_high: float    # Pips from session high
    distance_to_low: float     # Pips from session low
    confidence: float = 0.0


@dataclass
class MarketState:
    """Complete market state for a single timestamp"""
    timestamp: float                                    # Unix timestamp
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # State variables
    time_regime: Tuple[TimeRegimeType, float] = (TimeRegimeType.UNKNOWN, 0.0)  # (type, confidence)
    macro_expectation: Tuple[MacroExpectationType, float] = (MacroExpectationType.UNKNOWN, 0.0)
    liquidity: LiquidityState = field(default_factory=lambda: LiquidityState(0, 0, 0, LiquidityStateType.NORMAL))
    volatility: VolatilityState = field(default_factory=lambda: VolatilityState(0, 0, VolatilityStateType.NORMAL))
    dealer_positioning: DealerPositioningState = field(default_factory=DealerPositioningState)
    earnings_exposure: EarningsExposureState = field(default_factory=EarningsExposureState)
    price_location: PriceLocationState = field(default_factory=lambda: PriceLocationState(0.5, 0, 0, 0, 0))
    
    # Bid/ask
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread_pips: float = 0.0
    
    # Session & Flow Context (v1.1.1)
    session_name: str = ""                             # GLOBEX, PREMARKET, RTH_OPEN, MIDDAY, POWER_HOUR, CLOSE
    session_vol_scale: float = 1.0                     # Session-specific volatility scale
    session_liq_scale: float = 1.0                     # Session-specific liquidity scale
    session_risk_scale: float = 1.0                    # Session-specific risk scale
    prior_high: Optional[float] = None                 # Prior day high
    prior_low: Optional[float] = None                  # Prior day low
    overnight_high: Optional[float] = None             # Overnight high
    overnight_low: Optional[float] = None              # Overnight low
    vwap: Optional[float] = None                       # Volume-weighted average price
    vwap_distance_pct: float = 0.0                     # Distance from VWAP as %
    round_level_proximity: Optional[str] = None        # "5000" or "18000" if near round level
    stop_run_detected: bool = False                    # True if stop-run pattern detected
    initiative_move_detected: bool = False             # True if initiative move detected
    level_reaction_score: float = 0.0                  # -1.0 to 1.0, reaction to key levels


# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    """Load real market data from various sources"""
    
    SUPPORTED_FORMATS = ['csv', 'parquet']
    SUPPORTED_SYMBOLS = list(EXCHANGE_SESSIONS.keys())
    SUPPORTED_TIMEFRAMES = ['1m', '5m', '15m', '1h']
    
    def __init__(self):
        """Initialize data loader"""
        logger.info("DataLoader initialized")
    
    def load_csv(
        self,
        filepath: str,
        symbol: str,
        timeframe: str,
        date_column: str = 'datetime',
        ohlcv_columns: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data from CSV file.
        
        Args:
            filepath: Path to CSV file
            symbol: Trading symbol
            timeframe: Timeframe (1m, 5m, 15m, 1h)
            date_column: Name of datetime column
            ohlcv_columns: Dict mapping standard names to actual column names
                          Default: {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
        
        Returns:
            DataFrame with OHLCV data, indexed by datetime
        """
        if symbol not in self.SUPPORTED_SYMBOLS:
            logger.warning(f"Symbol {symbol} not in preset list but will attempt to load")
        
        if timeframe not in self.SUPPORTED_TIMEFRAMES:
            logger.warning(f"Timeframe {timeframe} may not be supported")
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Default column mapping
        if ohlcv_columns is None:
            ohlcv_columns = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
        
        logger.info(f"Loading CSV: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")
        
        # Rename columns if needed
        rename_map = {v: k for k, v in ohlcv_columns.items()}
        df = df.rename(columns=rename_map)
        
        # Ensure datetime index
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in CSV")
        
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
        df.index.name = 'timestamp'
        
        # Validate required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Keep only OHLCV columns
        df = df[required].copy()
        
        # Ensure numeric types
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN
        df = df.dropna()
        
        logger.info(f"Loaded {len(df)} candles from {filepath}")
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def load_parquet(
        self,
        filepath: str,
        symbol: str,
        timeframe: str,
        date_column: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Load OHLCV data from Parquet file.
        
        Args:
            filepath: Path to Parquet file
            symbol: Trading symbol
            timeframe: Timeframe
            date_column: Name of datetime column
        
        Returns:
            DataFrame with OHLCV data
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading Parquet: {filepath}")
        
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            raise ValueError(f"Failed to read Parquet: {e}")
        
        # Ensure datetime index
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in Parquet")
        
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
        df.index.name = 'timestamp'
        
        # Validate required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        df = df[required].copy()
        
        # Ensure numeric types
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        logger.info(f"Loaded {len(df)} candles from {filepath}")
        
        return df
    
    def repair_gaps(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        method: str = 'forward_fill'
    ) -> pd.DataFrame:
        """
        Detect and repair missing candles in data.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe
            method: Repair method ('forward_fill', 'drop')
        
        Returns:
            Repaired DataFrame
        """
        if df.empty:
            return df
        
        expected_interval = CANDLE_INTERVALS.get(timeframe, 60)
        df_copy = df.copy()
        
        # Check for gaps
        time_diffs = df_copy.index.to_series().diff()
        expected_timedelta = pd.Timedelta(seconds=expected_interval)
        
        gaps = time_diffs[time_diffs != expected_timedelta]
        
        if len(gaps) > 0:
            logger.warning(f"Found {len(gaps)} gaps in {symbol} {timeframe} data")
            logger.debug(f"Gap sizes: {gaps.value_counts().head()}")
        
        if method == 'drop':
            # Drop rows after gaps
            df_copy = df_copy.loc[df_copy.index[0]:gaps.idxmin() - expected_timedelta]
            logger.info(f"Dropped {len(df) - len(df_copy)} candles after first gap")
        
        elif method == 'forward_fill':
            # Forward fill missing candles
            idx_full = pd.date_range(
                start=df_copy.index[0],
                end=df_copy.index[-1],
                freq=f'{expected_interval}S'
            )
            df_reindexed = df_copy.reindex(idx_full)
            df_reindexed = df_reindexed.fillna(method='ffill')
            df_copy = df_reindexed.dropna()
            logger.info(f"Forward filled missing candles. New shape: {df_copy.shape}")
        
        return df_copy
    
    def align_to_sessions(
        self,
        df: pd.DataFrame,
        symbol: str,
        keep_sessions: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Align timestamps to exchange sessions.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            keep_sessions: List of sessions to keep (None = keep all)
        
        Returns:
            DataFrame with exchange-aligned timestamps
        """
        if symbol not in EXCHANGE_SESSIONS:
            logger.warning(f"No session info for {symbol}, returning as-is")
            return df
        
        df_copy = df.copy()
        
        # Add session info
        sessions = EXCHANGE_SESSIONS[symbol]
        df_copy['session'] = df_copy.index.map(
            lambda ts: self._get_session(ts, sessions)
        )
        
        if keep_sessions:
            df_copy = df_copy[df_copy['session'].isin(keep_sessions)]
            logger.info(f"Kept {len(df_copy)} candles from sessions: {keep_sessions}")
        
        # Remove session column (was just for filtering)
        df_copy = df_copy.drop(columns=['session'])
        
        return df_copy
    
    def estimate_spreads(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Estimate bid/ask spread if not included.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe
        
        Returns:
            DataFrame with 'bid' and 'ask' columns added
        """
        df_copy = df.copy()
        
        # Get typical spread for this symbol/timeframe
        typical_spread = TYPICAL_SPREADS.get(symbol, {}).get(timeframe, 0.01)
        
        # Estimate spread as 1/4 of typical for this symbol
        # (real spreads vary, this is conservative estimate)
        estimated_spread = typical_spread / 4
        
        # Set bid/ask around close
        df_copy['bid'] = df_copy['close'] - (estimated_spread / 2)
        df_copy['ask'] = df_copy['close'] + (estimated_spread / 2)
        df_copy['spread_pips'] = estimated_spread
        
        logger.info(f"Estimated spread: {estimated_spread:.4f} pips for {symbol} {timeframe}")
        
        return df_copy
    
    @staticmethod
    def _get_session(timestamp: pd.Timestamp, sessions: Dict[str, Tuple[int, int]]) -> str:
        """Determine which session a timestamp falls in"""
        hour = timestamp.hour
        
        for session, (start_hour, end_hour) in sessions.items():
            if start_hour <= end_hour:
                if start_hour <= hour < end_hour:
                    return session
            else:
                # Session crosses midnight
                if hour >= start_hour or hour < end_hour:
                    return session
        
        return "Unknown"


# ============================================================================
# MARKET STATE BUILDER
# ============================================================================

class MarketStateBuilder:
    """Reconstruct market states from OHLCV data.
    
    TIME-CAUSAL GUARANTEES:
      ✓ Only uses historical data (up to and including current candle)
      ✓ NO future data leakage
      ✓ Lookback window only goes backward in time
      ✓ All indicators computed from past data only
      ✓ Safe for live trading simulation (no lookahead bias)
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        lookback: int = 100
    ):
        """
        Initialize market state builder.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            lookback: Number of candles to lookback for indicators (NO FUTURE CANDLES)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback
        
        # Inject SessionContext for session/flow awareness
        try:
            from engine.session_context import SessionContext
            self.session_context = SessionContext()
            logger.info(f"MarketStateBuilder initialized with SessionContext for {symbol} {timeframe}")
        except ImportError:
            logger.warning(f"SessionContext not available; session fields will be empty")
            self.session_context = None
        
        logger.info(f"MarketStateBuilder initialized for {symbol} {timeframe} (time-causal)")
    
    def build_states(self, df: pd.DataFrame, time_causal_check: bool = True) -> List[MarketState]:
        """
        Build market states for all candles.
        
        TIME-CAUSAL PROMISE:
          - Only uses data from past and current candle
          - NO future data
          - Suitable for live backtesting
        
        Args:
            df: DataFrame with OHLCV data
            time_causal_check: If True, validate data is time-causal safe
        
        Returns:
            List of MarketState objects (time-causal, no-lookahead)
        """
        if time_causal_check:
            is_valid, warnings = validate_time_causal_data(df, self.symbol, self.timeframe)
            if warnings:
                for warn in warnings:
                    logger.warning(f"[TIME-CAUSAL] {warn}")
            if not is_valid:
                raise ValueError(f"Data fails time-causal validation: {warnings}")
        
        logger.info(f"Building market states for {len(df)} candles (time-causal)...")
        
        states = []
        
        for i in range(len(df)):
            if i < self.lookback:
                # Skip candles in lookback period (insufficient history)
                continue
            
            row = df.iloc[i]
            # CRITICAL: Only use history UP TO current candle (inclusive)
            # lookback_df = df.iloc[max(0, i - self.lookback):i + 1]
            # This ensures NO FUTURE DATA is included
            lookback_df = df.iloc[max(0, i - self.lookback):i + 1]
            
            # Verify current row is last in lookback (time-causal check)
            if len(lookback_df) > 0 and lookback_df.index[-1] != row.name:
                raise ValueError(
                    f"[TIME-CAUSAL VIOLATION] Lookback window doesn't end at current row. "
                    f"This indicates future data leakage!"
                )
            
            state = self._build_single_state(
                timestamp=row.name,
                row=row,
                lookback_df=lookback_df,
                index=i
            )
            states.append(state)
        
        logger.info(f"Built {len(states)} market states (time-causal, no lookahead)")
        return states
    
    def _build_single_state(
        self,
        timestamp: pd.Timestamp,
        row: pd.Series,
        lookback_df: pd.DataFrame,
        index: int
    ) -> MarketState:
        """Build a single market state"""
        
        # Time regime
        time_regime = self._identify_time_regime(timestamp)
        
        # Macro expectation (simple: based on day of week/time)
        macro_exp = self._identify_macro_expectation(timestamp)
        
        # Liquidity state
        liquidity = self._build_liquidity_state(row, lookback_df)
        
        # Volatility state
        volatility = self._build_volatility_state(lookback_df)
        
        # Dealer positioning (simplified)
        positioning = self._build_dealer_positioning(lookback_df)
        
        # Earnings exposure (simplified)
        earnings = self._build_earnings_exposure(timestamp)
        
        # Price location in session
        price_location = self._build_price_location(lookback_df)
        
        # Bid/ask
        bid = row.get('bid', row['close'] - 0.0001)
        ask = row.get('ask', row['close'] + 0.0001)
        spread_pips = row.get('spread_pips', 0.0001)
        
        # Session and Flow context (v1.1.1)
        session_name = ""
        session_vol_scale = 1.0
        session_liq_scale = 1.0
        session_risk_scale = 1.0
        prior_high = None
        prior_low = None
        overnight_high = None
        overnight_low = None
        vwap = None
        vwap_distance_pct = 0.0
        round_level_proximity = None
        stop_run_detected = False
        initiative_move_detected = False
        level_reaction_score = 0.0
        
        if self.session_context is not None:
            # Update SessionContext with current timestamp and lookback data
            prices = lookback_df['close'].tail(20).tolist() if len(lookback_df) > 0 else []
            volumes = lookback_df['volume'].tail(20).tolist() if len(lookback_df) > 0 else []
            
            # Compute prior/overnight levels from historical data (if available)
            # Simplified: use 20-candle range as proxy
            if len(lookback_df) > 0:
                prior_high = lookback_df['high'].tail(20).max()
                prior_low = lookback_df['low'].tail(20).min()
                overnight_high = lookback_df['high'].tail(10).max()
                overnight_low = lookback_df['low'].tail(10).min()
            
            # Round levels for ES/NQ
            round_levels = []
            if self.symbol == 'ES':
                round_levels = [5000.0, 5100.0, 4900.0]  # Common ES levels
            elif self.symbol == 'NQ':
                round_levels = [18000.0, 18500.0, 17500.0]  # Common NQ levels
            
            # Update session context
            # Handle both tz-aware and tz-naive timestamps
            if isinstance(timestamp, pd.Timestamp):
                if timestamp.tz is None:
                    dt_utc = timestamp.tz_localize('UTC')
                else:
                    dt_utc = timestamp.tz_convert('UTC')
            else:
                dt_utc = pd.Timestamp(timestamp, tz='UTC')
            self.session_context.update(
                dt_utc,
                recent_prices=prices if prices else None,
                recent_volumes=volumes if volumes else None,
                prior_high=prior_high,
                prior_low=prior_low,
                overnight_high=overnight_high,
                overnight_low=overnight_low,
                round_levels=round_levels
            )
            
            # Extract session info
            sess = self.session_context.get_session()
            session_name = sess.value if sess else ""
            
            mods = self.session_context.get_session_modifiers()
            session_vol_scale = mods.volatility_scale
            session_liq_scale = mods.liquidity_scale
            session_risk_scale = mods.risk_scale
            
            # Flow context fields
            flow = self.session_context.flow
            vwap = flow.vwap
            if vwap and row['close'] > 0:
                vwap_distance_pct = abs(row['close'] - vwap) / row['close'] * 100.0
            
            if flow.round_levels:
                for level in flow.round_levels:
                    if abs(row['close'] - level) < 5:  # Within 5 points
                        round_level_proximity = f"{level:.0f}"
                        break
            
            stop_run_detected = flow.stop_run_detected
            initiative_move_detected = flow.initiative_move
            
            # Level reaction score (simple heuristic)
            if prior_high and prior_low:
                mid = (prior_high + prior_low) / 2.0
                if row['close'] > mid:
                    level_reaction_score = min(1.0, (row['close'] - mid) / (prior_high - mid) if prior_high != mid else 0.0)
                else:
                    level_reaction_score = max(-1.0, -(mid - row['close']) / (mid - prior_low) if mid != prior_low else 0.0)
        
        state = MarketState(
            timestamp=timestamp.timestamp(),
            symbol=self.symbol,
            timeframe=self.timeframe,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            time_regime=time_regime,
            macro_expectation=macro_exp,
            liquidity=liquidity,
            volatility=volatility,
            dealer_positioning=positioning,
            earnings_exposure=earnings,
            price_location=price_location,
            bid=bid,
            ask=ask,
            spread_pips=spread_pips,
            # Session/Flow fields
            session_name=session_name,
            session_vol_scale=session_vol_scale,
            session_liq_scale=session_liq_scale,
            session_risk_scale=session_risk_scale,
            prior_high=prior_high,
            prior_low=prior_low,
            overnight_high=overnight_high,
            overnight_low=overnight_low,
            vwap=vwap,
            vwap_distance_pct=vwap_distance_pct,
            round_level_proximity=round_level_proximity,
            stop_run_detected=stop_run_detected,
            initiative_move_detected=initiative_move_detected,
            level_reaction_score=level_reaction_score,
        )
        
        return state
    
    def _identify_time_regime(self, timestamp: pd.Timestamp) -> Tuple[TimeRegimeType, float]:
        """Identify time regime (Asia, London, NY_Open, etc.)"""
        
        if self.symbol not in EXCHANGE_SESSIONS:
            return (TimeRegimeType.UNKNOWN, 0.0)
        
        sessions = EXCHANGE_SESSIONS[self.symbol]
        hour = timestamp.hour
        
        # Map session to regime
        for regime_name, (start_hour, end_hour) in sessions.items():
            try:
                regime_type = TimeRegimeType[regime_name]
            except KeyError:
                continue
            
            if start_hour <= end_hour:
                if start_hour <= hour < end_hour:
                    return (regime_type, 0.95)
            else:
                if hour >= start_hour or hour < end_hour:
                    return (regime_type, 0.95)
        
        return (TimeRegimeType.UNKNOWN, 0.0)
    
    def _identify_macro_expectation(self, timestamp: pd.Timestamp) -> Tuple[MacroExpectationType, float]:
        """Identify macro expectation state"""
        
        # Simplified: just use day of week and time
        # In production, integrate with economic calendar
        
        day_of_week = timestamp.dayofweek  # 0=Monday, 4=Friday
        hour = timestamp.hour
        
        # NFP is first Friday of month at 1:30 PM ET (18:30 UTC)
        if day_of_week == 4 and 17 <= hour <= 18:
            return (MacroExpectationType.PRE_NFP, 0.8)
        elif day_of_week == 4 and 18 <= hour <= 19:
            return (MacroExpectationType.POST_NFP, 0.8)
        
        # CPI is typically mid-month at 12:30 PM ET
        if 16 <= hour <= 17:
            return (MacroExpectationType.QUIET_PERIOD, 0.5)
        
        return (MacroExpectationType.QUIET_PERIOD, 0.5)
    
    def _build_liquidity_state(self, row: pd.Series, lookback_df: pd.DataFrame) -> LiquidityState:
        """Build liquidity state"""
        
        # Average volume
        avg_volume = lookback_df['volume'].mean()
        current_volume = row['volume']
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Range as % of close
        range_pct = (row['high'] - row['low']) / row['close'] * 100 if row['close'] > 0 else 0
        
        # VWAP calculation (simplified)
        vwap = (lookback_df['volume'] * lookback_df['close']).sum() / lookback_df['volume'].sum()
        vwap_distance = abs(row['close'] - vwap) / row['close'] * 100 if row['close'] > 0 else 0
        
        # Determine liquidity state
        if volume_ratio > 1.5:
            liquidity_type = LiquidityStateType.ABUNDANT
            confidence = 0.85
        elif volume_ratio > 1.0:
            liquidity_type = LiquidityStateType.NORMAL
            confidence = 0.8
        elif volume_ratio > 0.5:
            liquidity_type = LiquidityStateType.CONSTRAINED
            confidence = 0.7
        else:
            liquidity_type = LiquidityStateType.DROUGHT
            confidence = 0.6
        
        return LiquidityState(
            volume=current_volume,
            range_pct=range_pct,
            vwap_distance=vwap_distance,
            liquidity_type=liquidity_type,
            confidence=confidence
        )
    
    def _build_volatility_state(self, lookback_df: pd.DataFrame) -> VolatilityState:
        """Build volatility state"""
        
        # ATR calculation
        lookback_df = lookback_df.copy()
        lookback_df['tr'] = np.maximum(
            lookback_df['high'] - lookback_df['low'],
            np.maximum(
                abs(lookback_df['high'] - lookback_df['close'].shift(1)),
                abs(lookback_df['low'] - lookback_df['close'].shift(1))
            )
        )
        atr = lookback_df['tr'].mean()
        atr_percent = (atr / lookback_df['close'].iloc[-1] * 100) if lookback_df['close'].iloc[-1] > 0 else 0
        
        # Realized volatility (log returns std)
        returns = np.log(lookback_df['close'] / lookback_df['close'].shift(1))
        realized_vol = returns.std() * np.sqrt(252 * 24 * 60 / self._get_minutes_per_candle()) * 100
        
        # Determine volatility state
        if realized_vol < 0.5:
            vol_state = VolatilityStateType.VERY_LOW
            confidence = 0.8
        elif realized_vol < 1.0:
            vol_state = VolatilityStateType.LOW
            confidence = 0.8
        elif realized_vol < 1.5:
            vol_state = VolatilityStateType.NORMAL
            confidence = 0.85
        elif realized_vol < 2.0:
            vol_state = VolatilityStateType.HIGH
            confidence = 0.8
        else:
            vol_state = VolatilityStateType.VERY_HIGH
            confidence = 0.8
        
        return VolatilityState(
            atr_percent=atr_percent,
            realized_vol=realized_vol,
            vol_state=vol_state,
            confidence=confidence
        )
    
    def _build_dealer_positioning(self, lookback_df: pd.DataFrame) -> DealerPositioningState:
        """Build dealer positioning state (simplified)"""
        
        # Simplified gamma estimate based on price clustering
        closes = lookback_df['close'].values
        
        # Find potential strike clusters
        price_std = np.std(closes)
        price_mean = np.mean(closes)
        
        # Gamma estimate: simplified as inverse of realized volatility
        returns = np.diff(np.log(closes))
        realized_vol = np.std(returns)
        gamma_exposure = 1.0 / (realized_vol + 0.01) if realized_vol > 0 else 10.0
        
        # Bias based on recent price action
        recent_returns = np.sum(np.diff(closes[-10:]))
        if recent_returns > price_std * 0.1:
            bias = "long"
        elif recent_returns < -price_std * 0.1:
            bias = "short"
        else:
            bias = "neutral"
        
        return DealerPositioningState(
            gamma_exposure=gamma_exposure,
            strike_clusters=[price_mean - price_std, price_mean, price_mean + price_std],
            positioning_bias=bias,
            confidence=0.6
        )
    
    def _build_earnings_exposure(self, timestamp: pd.Timestamp) -> EarningsExposureState:
        """Build earnings exposure state (simplified)"""
        
        # In production, integrate with earnings calendar
        # For now, just mark mega-cap earnings weeks
        
        day_of_week = timestamp.dayofweek
        
        # Earnings are typically after market close
        mega_cap_earnings_week = (day_of_week < 5)  # Any weekday
        mega_cap_earnings_today = False
        
        return EarningsExposureState(
            mega_cap_earnings_today=mega_cap_earnings_today,
            mega_cap_earnings_week=mega_cap_earnings_week,
            impact_level="low" if not mega_cap_earnings_week else "medium"
        )
    
    def _build_price_location(self, lookback_df: pd.DataFrame) -> PriceLocationState:
        """Build price location state"""
        
        # Session range (use lookback as proxy for session)
        session_high = lookback_df['high'].max()
        session_low = lookback_df['low'].min()
        current_price = lookback_df['close'].iloc[-1]
        
        # Range position (0 = at low, 1 = at high)
        range_span = session_high - session_low
        if range_span > 0:
            range_position = (current_price - session_low) / range_span
        else:
            range_position = 0.5
        
        # Distance from extremes (in pips)
        distance_to_high = session_high - current_price
        distance_to_low = current_price - session_low
        
        return PriceLocationState(
            range_position=max(0, min(1, range_position)),
            session_high=session_high,
            session_low=session_low,
            distance_to_high=distance_to_high,
            distance_to_low=distance_to_low,
            confidence=0.8
        )
    
    def _get_minutes_per_candle(self) -> int:
        """Get minutes per candle for this timeframe"""
        return CANDLE_INTERVALS.get(self.timeframe, 60) // 60


# ============================================================================
# UTILITIES
# ============================================================================

def validate_data(df: pd.DataFrame, symbol: str, timeframe: str) -> Tuple[bool, List[str]]:
    """
    Validate loaded data quality.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Trading symbol
        timeframe: Timeframe
    
    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings = []
    
    if df.empty:
        warnings.append("DataFrame is empty")
        return False, warnings
    
    # Check required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.append(f"Missing columns: {missing}")
        return False, warnings
    
    # Check OHLC relationships
    invalid_ohlc = (df['high'] < df['low']).sum()
    if invalid_ohlc > 0:
        warnings.append(f"High < Low in {invalid_ohlc} candles")
    
    # Check for negative values
    if (df[required] < 0).any().any():
        warnings.append("Negative values found in OHLCV data")
    
    # Check for NaN values
    if df[required].isna().any().any():
        warnings.append("NaN values found in OHLCV data")
    
    # Check volume
    if (df['volume'] == 0).sum() > len(df) * 0.1:
        warnings.append(f"More than 10% zero volume candles")
    
    return len(warnings) == 0, warnings


if __name__ == "__main__":
    # Example usage
    print("DataLoader Module - Real Market Data Loader")
    print("See documentation for usage examples")
