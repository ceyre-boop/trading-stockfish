"""
Signal generation: momentum and mean-reversion signals for intraday futures.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Sequence

SIGNAL_LONG  =  1
SIGNAL_FLAT  =  0
SIGNAL_SHORT = -1


@dataclass
class SignalConfig:
    # Momentum
    momentum_window: int = 10       # lookback for momentum signal
    momentum_threshold: float = 0.0 # minimum return to trigger signal

    # Mean-reversion (z-score based)
    mean_rev_window: int = 20
    mean_rev_z_entry: float = 1.5   # enter when |z| >= this
    mean_rev_z_exit: float = 0.5    # exit when |z| <= this


class SignalGenerator:
    """Generates trading signals from price data."""

    def __init__(self, config: SignalConfig | None = None) -> None:
        self.config = config or SignalConfig()

    def momentum(self, prices: Sequence[float]) -> np.ndarray:
        """
        Simple momentum: +1 if price > price[−window], −1 if below, 0 otherwise.
        Returns array of same length as prices.
        """
        px = np.asarray(prices, dtype=float)
        signals = np.zeros(len(px), dtype=int)
        w = self.config.momentum_window
        thr = self.config.momentum_threshold
        for i in range(w, len(px)):
            ret = (px[i] - px[i - w]) / px[i - w] if px[i - w] != 0 else 0.0
            if ret > thr:
                signals[i] = SIGNAL_LONG
            elif ret < -thr:
                signals[i] = SIGNAL_SHORT
        return signals

    def mean_reversion(self, prices: Sequence[float]) -> np.ndarray:
        """
        Z-score mean-reversion: short when z ≥ entry_z, long when z ≤ −entry_z.
        Flattens when |z| ≤ exit_z.
        """
        px = np.asarray(prices, dtype=float)
        signals = np.zeros(len(px), dtype=int)
        w = self.config.mean_rev_window
        z_entry = self.config.mean_rev_z_entry
        z_exit = self.config.mean_rev_z_exit
        current = SIGNAL_FLAT
        for i in range(w, len(px)):
            window = px[i - w : i]
            mean, std = np.mean(window), np.std(window, ddof=1)
            z = (px[i] - mean) / std if std > 0 else 0.0
            if abs(z) <= z_exit:
                current = SIGNAL_FLAT
            elif z >= z_entry:
                current = SIGNAL_SHORT   # price stretched high → expect reversion
            elif z <= -z_entry:
                current = SIGNAL_LONG    # price stretched low → expect reversion
            signals[i] = current
        return signals

    def combined(
        self, prices: Sequence[float], regime: int = 1
    ) -> np.ndarray:
        """
        Regime-aware combined signal.

        regime 0 (LOW_VOL)   → mean-reversion dominates
        regime 1 (NORMAL)    → equal-weight blend
        regime 2 (HIGH_VOL)  → momentum dominates
        """
        mom = self.momentum(prices)
        mr  = self.mean_reversion(prices)
        if regime == 0:
            weights = (0.25, 0.75)
        elif regime == 2:
            weights = (0.75, 0.25)
        else:
            weights = (0.50, 0.50)
        blended = weights[0] * mom + weights[1] * mr
        return np.sign(blended).astype(int)
