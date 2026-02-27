"""
Risk governance: position sizing, drawdown monitoring, and VaR estimation.
"""
import numpy as np
from dataclasses import dataclass
from typing import Sequence


@dataclass
class RiskConfig:
    max_position: float = 1.0       # maximum notional position (units)
    max_drawdown: float = 0.10      # halt trading above this drawdown fraction
    risk_per_trade: float = 0.01    # fraction of equity risked per trade
    var_confidence: float = 0.95    # confidence level for historical VaR
    var_window: int = 60            # lookback bars for VaR


class RiskGovernor:
    """Governs risk limits and computes position sizes."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self._equity_peak: float = 1.0

    def position_size(self, equity: float, stop_distance: float) -> float:
        """
        Kelly-inspired fixed-fractional sizing.

        equity        – current equity (e.g. 1.0 = 100 %)
        stop_distance – fractional distance to stop (e.g. 0.005 = 0.5 %)
        """
        if stop_distance <= 0:
            return 0.0
        risk_amount = equity * self.config.risk_per_trade
        size = risk_amount / stop_distance
        return float(np.clip(size, 0.0, self.config.max_position))

    def drawdown(self, equity: float) -> float:
        """Update peak equity and return current drawdown fraction (0–1)."""
        self._equity_peak = max(self._equity_peak, equity)
        return (self._equity_peak - equity) / self._equity_peak

    def is_halted(self, equity: float) -> bool:
        """Return True when drawdown breaches the configured maximum."""
        return self.drawdown(equity) >= self.config.max_drawdown

    def historical_var(self, pnl_series: Sequence[float]) -> float:
        """
        Estimate 1-bar historical VaR (loss expressed as a positive number)
        at the configured confidence level over the configured lookback window.
        """
        arr = np.asarray(pnl_series, dtype=float)
        window = arr[-self.config.var_window :]
        if len(window) == 0:
            return 0.0
        return float(-np.percentile(window, (1 - self.config.var_confidence) * 100))

    def reset(self) -> None:
        """Reset peak equity tracking (e.g. start of new session)."""
        self._equity_peak = 1.0
