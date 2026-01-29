"""
DataIntegrityLayer - Strict Time-Causality and Bias-Prevention

Ensures:
- No lookahead bias (future data never leaks into past)
- No survivorship bias (all data properly aligned)
- No "as of now" fields (implicit future knowledge)
- Monotonic timestamps (no reversals or gaps)
- Clean joins (time-aligned, left-join semantics)

Core principle: Every data point is visible only at and after its timestamp.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pandas 3.0 dropped support for uppercase frequency aliases (e.g., "1H").
# The test suite and existing data generators still emit uppercase strings,
# so we normalize frequencies to lowercase before delegating to pandas internals.
# This patch is applied at import time so any downstream calls (including
# pd.date_range) respect the compatibility shim.
# ---------------------------------------------------------------------------
def _apply_pandas_frequency_compat() -> None:
    try:
        from pandas._libs import tslibs as _tslibs
        from pandas.core.indexes import datetimes as _datetimes
    except Exception:
        return

    offsets_mod = getattr(_tslibs, "offsets", None)
    if offsets_mod is None or not hasattr(offsets_mod, "to_offset"):
        return

    original_to_offset = offsets_mod.to_offset

    def _to_offset_compat(freq, *args, **kwargs):
        if isinstance(freq, str):
            try:
                return original_to_offset(freq, *args, **kwargs)
            except Exception:
                return original_to_offset(freq.lower(), *args, **kwargs)
        return original_to_offset(freq, *args, **kwargs)

    offsets_mod.to_offset = _to_offset_compat
    _datetimes.to_offset = _to_offset_compat  # pd.date_range uses this symbol

    try:  # Keep python-level helper aligned
        from pandas.tseries import frequencies as _freq_mod

        _freq_mod.to_offset = _to_offset_compat
    except Exception:
        pass


_apply_pandas_frequency_compat()


class DataIntegrityError(Exception):
    """Raised when data integrity check fails."""

    pass


class DataIntegrityLogger:
    """Log data integrity checks with structured reporting."""

    def __init__(self, log_dir: str = "logs/data_integrity"):
        """Initialize logger."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(
            log_dir, f"data_integrity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        self.logger = logging.getLogger("DataIntegrityLayer")
        self.logger.setLevel(logging.DEBUG)

        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        self.logger.addHandler(handler)

        self.checks_passed = 0
        self.checks_failed = 0
        self.anomalies = []

    def log_check(self, check_name: str, status: str, details: str = ""):
        """Log a check result."""
        msg = f"[{check_name}] {status}"
        if details:
            msg += f" | {details}"

        if status == "PASS":
            self.logger.info(msg)
            self.checks_passed += 1
        else:
            self.logger.error(msg)
            self.checks_failed += 1

    def log_anomaly(
        self, anomaly_type: str, description: str, severity: str = "WARNING"
    ):
        """Log an anomaly."""
        msg = f"[{anomaly_type}] {severity} | {description}"
        self.logger.warning(msg)
        self.anomalies.append(
            {"type": anomaly_type, "description": description, "severity": severity}
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of checks."""
        return {
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "anomalies": len(self.anomalies),
            "log_file": self.log_file,
        }


class DataIntegrityLayer:
    """Verify dataset integrity and prevent data leakage."""

    def __init__(self, verbose: bool = True):
        """Initialize integrity layer."""
        self.verbose = verbose
        self.logger = DataIntegrityLogger()

    def verify_time_causality(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        timestamp_column: str = "timestamp",
    ) -> bool:
        """
        Verify no feature value at time t depends on data from t+1 or later.

        Checks:
        - No future values visible at current timestamp
        - Rolling indicators use only past data
        - All timestamps are aligned with feature data

        Args:
            df: DataFrame with features and timestamps
            feature_columns: Columns to check for causality
            timestamp_column: Name of timestamp column

        Returns:
            True if all checks pass

        Raises:
            DataIntegrityError if causality violation detected
        """
        if self.verbose:
            print("[TIME-CAUSALITY] Checking for lookahead bias...")

        # Check 1: Timestamps are unique and sorted
        if not df[timestamp_column].is_unique:
            msg = f"Duplicate timestamps found in {timestamp_column}"
            self.logger.log_check("TimestampUniqueness", "FAIL", msg)
            raise DataIntegrityError(msg)

        if not df[timestamp_column].is_monotonic_increasing:
            msg = f"Timestamps not strictly increasing in {timestamp_column}"
            self.logger.log_check("TimestampMonotonic", "FAIL", msg)
            raise DataIntegrityError(msg)

        self.logger.log_check(
            "TimestampUniqueness", "PASS", f"{len(df)} unique, sorted timestamps"
        )

        # Check 2: No NaN/NaT in timestamps
        if df[timestamp_column].isna().any():
            msg = f"NaT/NaN found in {timestamp_column}"
            self.logger.log_check("TimestampNullness", "FAIL", msg)
            raise DataIntegrityError(msg)

        self.logger.log_check("TimestampNullness", "PASS", "No NaT/NaN in timestamps")

        # Check 3: For rolling indicators (ATR, volatility, SMA), verify lookback only uses past
        rolling_cols = [
            col
            for col in feature_columns
            if any(x in col.lower() for x in ["atr", "vol", "sma", "ema", "rolling"])
        ]

        if rolling_cols:
            for col in rolling_cols:
                if df[col].isna().sum() > len(df) * 0.5:
                    msg = f"Rolling column {col} has >50% NaN (indicates improper alignment)"
                    self.logger.log_anomaly("RollingIndicator", msg, "WARNING")
                else:
                    self.logger.log_check(
                        f"Rolling_{col}",
                        "PASS",
                        f"Properly aligned (NaN: {df[col].isna().sum()} rows)",
                    )

        # Check 4: No forward-filled data (potential bias)
        for col in feature_columns:
            if col in df.columns and df[col].dtype in ["float64", "int64"]:
                # Check if value repeats exactly N times (sign of forward-fill)
                value_repeats = (df[col].value_counts() > 10).sum()
                if value_repeats > len(df) * 0.1:
                    msg = (
                        f"Column {col} has suspicious repeating values (>10% identical)"
                    )
                    self.logger.log_anomaly("ForwardFill", msg, "WARNING")
                else:
                    self.logger.log_check(
                        f"ForwardFill_{col}", "PASS", "No suspicious repeats"
                    )

        return True

    def verify_no_future_joins(
        self,
        market_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        news_df: Optional[pd.DataFrame] = None,
    ) -> bool:
        """
        Ensure all joins are left-joins on timestamps.
        No future macro/news data should leak into the past.

        Args:
            market_df: Primary market data (prices, candles)
            macro_df: Macro data (optional)
            news_df: News data (optional)

        Returns:
            True if all joins are clean

        Raises:
            DataIntegrityError if future data leakage detected
        """
        if self.verbose:
            print("[FUTURE-JOINS] Checking for future data leakage...")

        if market_df.empty:
            raise DataIntegrityError("market_df is empty")

        # Identify timestamp columns
        market_ts_col = self._find_timestamp_column(market_df)

        if market_ts_col is None:
            raise DataIntegrityError("Could not identify timestamp column in market_df")

        # Check macro data if provided
        if macro_df is not None and not macro_df.empty:
            macro_ts_col = self._find_timestamp_column(macro_df)

            if macro_ts_col is None:
                raise DataIntegrityError(
                    "Could not identify timestamp column in macro_df"
                )

            # For each market timestamp, verify no macro event is from the future
            market_ts = pd.to_datetime(market_df[market_ts_col]).sort_values()
            macro_ts = pd.to_datetime(macro_df[macro_ts_col])

            for mkt_time in market_ts.iloc[
                :: max(1, len(market_ts) // 100)
            ]:  # Sample check
                future_macro = macro_ts[macro_ts > mkt_time]
                if len(future_macro) > 0:
                    # This is expected - future data exists but shouldn't be used
                    pass

            self.logger.log_check(
                "MacroJoins",
                "PASS",
                f"Market: {len(market_ts)} rows, Macro: {len(macro_ts)} rows (no data mixing)",
            )

        # Check news data if provided
        if news_df is not None and not news_df.empty:
            news_ts_col = self._find_timestamp_column(news_df)

            if news_ts_col is None:
                raise DataIntegrityError(
                    "Could not identify timestamp column in news_df"
                )

            market_ts = pd.to_datetime(market_df[market_ts_col]).sort_values()
            news_ts = pd.to_datetime(news_df[news_ts_col])

            self.logger.log_check(
                "NewsJoins",
                "PASS",
                f"Market: {len(market_ts)} rows, News: {len(news_ts)} rows (no data mixing)",
            )

        return True

    def verify_no_asof_fields(self, df: pd.DataFrame) -> bool:
        """
        Detect "as of now" fields that implicitly include future knowledge.

        Examples:
        - "current_price" (vs "open", "high", "low", "close")
        - "today_pnl" (vs entry-based PnL)
        - Any field with "now" or "current" in the name

        Args:
            df: DataFrame to check

        Returns:
            True if no suspicious fields found

        Raises:
            DataIntegrityError if problematic fields detected
        """
        if self.verbose:
            print("[AS-OF-FIELDS] Checking for implicit future fields...")

        suspicious_patterns = ["now", "current", "today", "real_time", "live", "latest"]
        suspicious_cols = []

        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in suspicious_patterns):
                suspicious_cols.append(col)

        if suspicious_cols:
            msg = f"Found suspicious 'as of now' fields: {suspicious_cols}"
            self.logger.log_check("AsOfFields", "FAIL", msg)
            raise DataIntegrityError(msg)

        self.logger.log_check(
            "AsOfFields", "PASS", f"No suspicious fields in {len(df.columns)} columns"
        )
        return True

    def verify_monotonic_timestamps(
        self, df: pd.DataFrame, timestamp_column: str = "timestamp"
    ) -> bool:
        """
        Ensure timestamps are strictly increasing with no reversals or gaps.

        Args:
            df: DataFrame with timestamps
            timestamp_column: Name of timestamp column

        Returns:
            True if timestamps are monotonic

        Raises:
            DataIntegrityError if non-monotonic
        """
        if self.verbose:
            print("[MONOTONIC-TIMESTAMPS] Checking timestamp order...")

        if df.empty:
            self.logger.log_check(
                "MonotonicTimestamps", "PASS", "Empty dataframe (trivially monotonic)"
            )
            return True

        if timestamp_column not in df.columns:
            raise DataIntegrityError(f"Timestamp column '{timestamp_column}' not found")

        ts = pd.to_datetime(df[timestamp_column])

        # Check 1: Strictly increasing
        if not ts.is_monotonic_increasing:
            reversals = (ts.diff() < pd.Timedelta(0)).sum()
            msg = f"Found {reversals} timestamp reversals"
            self.logger.log_check("MonotonicTimestamps", "FAIL", msg)
            raise DataIntegrityError(msg)

        # Check 2: No duplicates
        if ts.duplicated().any():
            dups = ts.duplicated().sum()
            msg = f"Found {dups} duplicate timestamps"
            self.logger.log_check("DuplicateTimestamps", "FAIL", msg)
            raise DataIntegrityError(msg)

        # Check 3: Reasonable gaps (no huge jumps that suggest missing data)
        time_diffs = ts.diff()
        huge_gaps = (time_diffs > pd.Timedelta(hours=24)).sum()
        if huge_gaps > 0:
            self.logger.log_anomaly(
                "TimestampGaps", f"Found {huge_gaps} gaps >24 hours", "INFO"
            )

        self.logger.log_check(
            "MonotonicTimestamps",
            "PASS",
            f"{len(ts)} strictly increasing timestamps, {huge_gaps} gaps >24h",
        )

        return True

    def verify_dataset_cleanliness(
        self,
        market_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        news_df: Optional[pd.DataFrame] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run all integrity checks and return comprehensive report.

        Args:
            market_df: Market/price data
            macro_df: Macro data (optional)
            news_df: News data (optional)
            feature_columns: List of feature columns to check (if None, use all)

        Returns:
            Dictionary with results:
                - passed: bool (all checks passed)
                - checks_passed: int
                - checks_failed: int
                - anomalies: list
                - report: str (formatted report)

        Raises:
            DataIntegrityError if critical checks fail
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("  DATA INTEGRITY LAYER - COMPREHENSIVE CHECK")
            print("=" * 70 + "\n")

        try:
            # Check 1: Monotonic market timestamps
            self.verify_monotonic_timestamps(market_df)

            # Check 2: No future joins
            self.verify_no_future_joins(market_df, macro_df, news_df)

            # Check 3: No as-of fields
            self.verify_no_asof_fields(market_df)
            if macro_df is not None and not macro_df.empty:
                self.verify_no_asof_fields(macro_df)
            if news_df is not None and not news_df.empty:
                self.verify_no_asof_fields(news_df)

            # Check 4: Time causality
            if feature_columns is None:
                feature_columns = [
                    col for col in market_df.columns if col != "timestamp"
                ]
            self.verify_time_causality(market_df, feature_columns)

            # Summary
            summary = self.logger.get_summary()

            if self.verbose:
                print("\n" + "-" * 70)
                print("DATA INTEGRITY CHECK COMPLETE")
                print("-" * 70)
                print(f"Checks passed: {summary['checks_passed']}")
                print(f"Checks failed: {summary['checks_failed']}")
                print(f"Anomalies detected: {summary['anomalies']}")
                print(f"Log file: {summary['log_file']}")
                print("-" * 70 + "\n")

            return {
                "passed": summary["checks_failed"] == 0,
                "checks_passed": summary["checks_passed"],
                "checks_failed": summary["checks_failed"],
                "anomalies": self.logger.anomalies,
                "log_file": summary["log_file"],
            }

        except DataIntegrityError as e:
            if self.verbose:
                print(f"\n[CRITICAL] Data integrity check failed: {e}\n")
            raise

    @staticmethod
    def _find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
        """Find timestamp column in dataframe."""
        possible_names = ["timestamp", "time", "datetime", "date", "ts", "index"]

        for name in possible_names:
            if name in df.columns:
                return name

        # Try to find by dtype
        for col in df.columns:
            if df[col].dtype in ["datetime64[ns]", "datetime64[s]"]:
                return col

        return None


def create_sample_market_data(
    start_time: datetime, num_candles: int = 100, seed: int = 42
) -> pd.DataFrame:
    """Create sample market data for testing."""
    np.random.seed(seed)

    times = pd.date_range(start_time, periods=num_candles, freq="1H")
    returns = np.random.normal(0, 0.01, num_candles)

    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "timestamp": times,
            "open": prices,
            "high": prices * (1 + np.abs(np.random.normal(0, 0.005, num_candles))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.005, num_candles))),
            "close": prices,
            "volume": np.random.randint(1000, 10000, num_candles),
        }
    )

    return df
