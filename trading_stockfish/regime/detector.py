"""
Regime detection via rolling volatility clustering.

Classifies each bar into one of three regimes:
  0 = LOW_VOL   (σ < low_threshold)
  1 = NORMAL    (low_threshold ≤ σ < high_threshold)
  2 = HIGH_VOL  (σ ≥ high_threshold)

Uses a configurable rolling window for realized volatility (annualized).
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Sequence

REGIME_LOW    = 0
REGIME_NORMAL = 1
REGIME_HIGH   = 2

@dataclass
class RegimeConfig:
    window: int = 20            # bars used for rolling vol
    low_threshold: float = 0.10 # annualized vol below which → LOW_VOL
    high_threshold: float = 0.25 # annualized vol above which → HIGH_VOL
    bars_per_year: int = 252     # trading bars per year (for annualisation)


class RegimeDetector:
    """Detects market regime from a price series."""

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()

    def rolling_volatility(self, prices: Sequence[float]) -> np.ndarray:
        """Return annualized rolling realized volatility for each bar."""
        px = np.asarray(prices, dtype=float)
        if len(px) < 2:
            return np.zeros(len(px))
        log_returns = np.diff(np.log(px))
        vol = np.full(len(px), np.nan)
        w = self.config.window
        scale = np.sqrt(self.config.bars_per_year)
        for i in range(w, len(log_returns) + 1):
            vol[i] = np.std(log_returns[i - w : i], ddof=1) * scale
        # fill leading NaN with the first valid value
        first_valid = next((v for v in vol if not np.isnan(v)), 0.0)
        vol = np.where(np.isnan(vol), first_valid, vol)
        return vol

    def detect(self, prices: Sequence[float]) -> np.ndarray:
        """
        Classify each bar into LOW_VOL / NORMAL / HIGH_VOL.

        Returns an integer array of length len(prices).
        """
        vol = self.rolling_volatility(prices)
        regimes = np.where(
            vol < self.config.low_threshold,
            REGIME_LOW,
            np.where(vol >= self.config.high_threshold, REGIME_HIGH, REGIME_NORMAL),
        ).astype(int)
        return regimes
