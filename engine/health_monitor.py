"""
EngineHealthMonitor - Self-aware performance tracking with regime-specific thresholds.

Tracks rolling performance metrics (Sharpe, drawdown) and outputs a risk_multiplier
that scales the PolicyEngine's allowed position size based on engine health.

Deterministic, time-causal, production-ready.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from datetime import datetime


# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class HealthSnapshot:
    """Immutable snapshot of engine health state."""
    timestamp: datetime
    rolling_sharpe: float
    rolling_drawdown: float
    regime_label: str
    health_status: str  # "HEALTHY", "DEGRADED", "CRITICAL"
    risk_multiplier: float  # 1.0, 0.5, 0.0
    bars_in_window: int
    realized_pnl: float
    unrealized_pnl: float
    daily_pnl: float


@dataclass
class RegimeThresholds:
    """Regime-specific performance thresholds."""
    min_sharpe: float
    max_drawdown: float


class EngineHealthMonitor:
    """
    Tracks rolling engine performance and outputs risk multiplier for position sizing.
    
    Attributes:
        window_size: Number of bars for rolling window (e.g., 500)
        rolling_pnl: Deque of recent P&L values
        rolling_returns: Deque of recent returns (percentage changes)
        rolling_drawdown: Current max drawdown in rolling window
        rolling_sharpe: Current annualized Sharpe ratio
        regime_history: Deque tracking regime labels over time
        expected_bands: Dict of regime-specific thresholds
        health_status: Current status ("HEALTHY", "DEGRADED", "CRITICAL")
        risk_multiplier: Current risk multiplier (1.0, 0.5, 0.0)
    """
    
    def __init__(
        self,
        window_size: int = 500,
        annualization_factor: float = 252.0  # Trading days per year
    ):
        """
        Initialize EngineHealthMonitor.
        
        Args:
            window_size: Number of bars for rolling window
            annualization_factor: Days per year for Sharpe annualization (default: 252 trading days)
        """
        self.window_size = window_size
        self.annualization_factor = annualization_factor
        
        # Rolling metrics
        self.rolling_pnl: deque = deque(maxlen=window_size)
        self.rolling_returns: deque = deque(maxlen=window_size)
        self.rolling_drawdown: float = 0.0
        self.rolling_sharpe: float = 0.0
        
        # Regime tracking
        self.regime_history: deque = deque(maxlen=window_size)
        self.current_regime: Optional[str] = None
        
        # Regime-specific thresholds (can be customized)
        self.expected_bands: Dict[str, RegimeThresholds] = {
            "high_vol": RegimeThresholds(min_sharpe=0.2, max_drawdown=0.15),
            "low_vol": RegimeThresholds(min_sharpe=0.1, max_drawdown=0.10),
            "risk_on": RegimeThresholds(min_sharpe=0.15, max_drawdown=0.12),
            "risk_off": RegimeThresholds(min_sharpe=0.05, max_drawdown=0.20),
        }
        
        # Health state
        self.health_status: str = "HEALTHY"
        self.risk_multiplier: float = 1.0
        self.cumulative_pnl: float = 0.0
        self.peak_cumulative_pnl: float = 0.0
        
        # Transition tracking
        self.last_status: str = "HEALTHY"
        self.status_transition_count: int = 0
        
        # Bar counter
        self.bars_processed: int = 0
        
        logger.info(
            f"EngineHealthMonitor initialized: "
            f"window_size={window_size}, "
            f"annualization_factor={annualization_factor}"
        )
    
    def update(
        self,
        pnl: float,
        regime_label: str,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0
    ) -> None:
        """
        Update health monitor with new P&L and regime information.
        
        Args:
            pnl: P&L change for this bar (can be positive or negative)
            regime_label: Current market regime ("high_vol", "low_vol", "risk_on", "risk_off")
            realized_pnl: Cumulative realized P&L (for snapshot)
            unrealized_pnl: Current unrealized P&L (for snapshot)
        """
        if pnl == 0.0:
            logger.debug("Skipping update with zero P&L")
            return
        
        self.bars_processed += 1
        
        # Update cumulative P&L and track peak
        self.cumulative_pnl += pnl
        if self.cumulative_pnl > self.peak_cumulative_pnl:
            self.peak_cumulative_pnl = self.cumulative_pnl
        
        # Add to rolling deques
        self.rolling_pnl.append(pnl)
        
        # Calculate return as percentage change from previous cumulative
        if self.cumulative_pnl - pnl != 0:
            ret = pnl / abs(self.cumulative_pnl - pnl)
        else:
            ret = pnl / 1.0 if pnl != 0 else 0.0
        self.rolling_returns.append(ret)
        
        # Update regime
        self.current_regime = regime_label
        self.regime_history.append(regime_label)
        
        # Recompute rolling metrics
        self._compute_rolling_sharpe()
        self._compute_rolling_drawdown()
        
        # Evaluate health status
        self._evaluate_health()
        
        logger.debug(
            f"Health update: "
            f"pnl={pnl:.2f}, "
            f"cumulative={self.cumulative_pnl:.2f}, "
            f"regime={regime_label}, "
            f"sharpe={self.rolling_sharpe:.3f}, "
            f"drawdown={self.rolling_drawdown:.3f}, "
            f"status={self.health_status}, "
            f"multiplier={self.risk_multiplier}"
        )
    
    def _compute_rolling_sharpe(self) -> None:
        """
        Compute annualized Sharpe ratio from rolling returns.
        
        Sharpe = (mean_return / std_return) * sqrt(annualization_factor)
        """
        if len(self.rolling_returns) < 2:
            self.rolling_sharpe = 0.0
            return
        
        returns_list = list(self.rolling_returns)
        
        # Calculate mean return
        mean_return = sum(returns_list) / len(returns_list)
        
        # Calculate standard deviation
        variance = sum((r - mean_return) ** 2 for r in returns_list) / len(returns_list)
        std_return = variance ** 0.5
        
        # Avoid division by zero
        if std_return == 0.0:
            self.rolling_sharpe = 0.0
            return
        
        # Calculate annualized Sharpe
        self.rolling_sharpe = (mean_return / std_return) * (self.annualization_factor ** 0.5)
    
    def _compute_rolling_drawdown(self) -> None:
        """
        Compute maximum drawdown in rolling window.
        
        Max Drawdown = (Peak - Trough) / Peak
        """
        if len(self.rolling_pnl) < 2:
            self.rolling_drawdown = 0.0
            return
        
        # Reconstruct cumulative P&L for this window
        cumulative = 0.0
        running_cumulative_list = []
        
        for pnl in self.rolling_pnl:
            cumulative += pnl
            running_cumulative_list.append(cumulative)
        
        # Find peak and trough
        peak = max(running_cumulative_list)
        trough = min(running_cumulative_list)
        
        # Calculate max drawdown
        if peak == 0:
            self.rolling_drawdown = 0.0
        else:
            self.rolling_drawdown = abs((trough - peak) / peak)
    
    def _evaluate_health(self) -> None:
        """
        Evaluate engine health based on rolling metrics vs regime thresholds.
        
        Health states:
            HEALTHY: Metrics within or better than thresholds
            DEGRADED: Some metrics below thresholds
            CRITICAL: Multiple metrics severely below thresholds
        """
        # Get regime thresholds
        regime = self.current_regime or "low_vol"
        thresholds = self.expected_bands.get(regime, self.expected_bands["low_vol"])
        
        # Check if window is large enough
        if len(self.rolling_pnl) < max(10, self.window_size // 2):
            self.health_status = "HEALTHY"
            self.risk_multiplier = 1.0
            return
        
        # Evaluate each metric
        sharpe_ok = self.rolling_sharpe >= thresholds.min_sharpe
        drawdown_ok = self.rolling_drawdown <= thresholds.max_drawdown
        
        # Determine health status
        if sharpe_ok and drawdown_ok:
            self.health_status = "HEALTHY"
            self.risk_multiplier = 1.0
        elif sharpe_ok or drawdown_ok:
            # One metric is OK, one is not
            self.health_status = "DEGRADED"
            self.risk_multiplier = 0.5
        else:
            # Both metrics are below thresholds
            self.health_status = "CRITICAL"
            self.risk_multiplier = 0.0
        
        # Log status transitions
        if self.health_status != self.last_status:
            self.status_transition_count += 1
            logger.warning(
                f"Health status transition: {self.last_status} → {self.health_status} | "
                f"sharpe={self.rolling_sharpe:.3f} (threshold={thresholds.min_sharpe}), "
                f"drawdown={self.rolling_drawdown:.3f} (threshold={thresholds.max_drawdown}), "
                f"regime={regime}, "
                f"risk_multiplier={self.risk_multiplier}"
            )
            self.last_status = self.health_status
    
    def compute_sharpe(self) -> float:
        """
        Get current rolling Sharpe ratio (public accessor).
        
        Returns:
            Current annualized Sharpe ratio
        """
        return self.rolling_sharpe
    
    def compute_drawdown(self) -> float:
        """
        Get current rolling maximum drawdown (public accessor).
        
        Returns:
            Current max drawdown as proportion (e.g., 0.10 = 10%)
        """
        return self.rolling_drawdown
    
    def get_risk_multiplier(self) -> float:
        """
        Get current risk multiplier for position sizing.
        
        Returns:
            1.0 (full size), 0.5 (half size), or 0.0 (no new entries)
        """
        return self.risk_multiplier
    
    def get_health_status(self) -> str:
        """
        Get current health status.
        
        Returns:
            "HEALTHY", "DEGRADED", or "CRITICAL"
        """
        return self.health_status
    
    def get_regime(self) -> Optional[str]:
        """
        Get current market regime.
        
        Returns:
            Current regime label or None if not set
        """
        return self.current_regime
    
    def get_state_snapshot(
        self,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0
    ) -> HealthSnapshot:
        """
        Get immutable snapshot of current health state.
        
        Args:
            realized_pnl: Cumulative realized P&L (for reporting)
            unrealized_pnl: Current unrealized P&L (for reporting)
        
        Returns:
            HealthSnapshot with all current metrics
        """
        return HealthSnapshot(
            timestamp=datetime.now(),
            rolling_sharpe=self.rolling_sharpe,
            rolling_drawdown=self.rolling_drawdown,
            regime_label=self.current_regime or "unknown",
            health_status=self.health_status,
            risk_multiplier=self.risk_multiplier,
            bars_in_window=len(self.rolling_pnl),
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            daily_pnl=self.cumulative_pnl,
        )
    
    def get_report(self) -> Dict:
        """
        Get comprehensive health report.
        
        Returns:
            Dict with all health metrics and status
        """
        return {
            "health_status": self.health_status,
            "risk_multiplier": self.risk_multiplier,
            "rolling_sharpe": self.rolling_sharpe,
            "rolling_drawdown": self.rolling_drawdown,
            "current_regime": self.current_regime,
            "cumulative_pnl": self.cumulative_pnl,
            "peak_cumulative_pnl": self.peak_cumulative_pnl,
            "bars_processed": self.bars_processed,
            "bars_in_window": len(self.rolling_pnl),
            "window_size": self.window_size,
            "status_transitions": self.status_transition_count,
            "expected_bands": {
                k: {"min_sharpe": v.min_sharpe, "max_drawdown": v.max_drawdown}
                for k, v in self.expected_bands.items()
            }
        }
    
    def reset_for_session(self) -> None:
        """
        Reset rolling metrics for new session (e.g., new day).
        
        Preserves cumulative state but clears rolling window.
        """
        logger.info(
            f"Resetting health monitor for new session: "
            f"bars_in_window={len(self.rolling_pnl)}, "
            f"cumulative_pnl={self.cumulative_pnl:.2f}"
        )
        
        self.rolling_pnl.clear()
        self.rolling_returns.clear()
        self.rolling_sharpe = 0.0
        self.rolling_drawdown = 0.0
        self.regime_history.clear()
        self.health_status = "HEALTHY"
        self.risk_multiplier = 1.0
        self.peak_cumulative_pnl = 0.0
        self.cumulative_pnl = 0.0
    
    def set_regime_thresholds(self, regime: str, min_sharpe: float, max_drawdown: float) -> None:
        """
        Customize thresholds for a specific regime.
        
        Args:
            regime: Regime label ("high_vol", "low_vol", "risk_on", "risk_off")
            min_sharpe: Minimum acceptable Sharpe ratio
            max_drawdown: Maximum acceptable drawdown
        """
        if regime in self.expected_bands:
            self.expected_bands[regime] = RegimeThresholds(
                min_sharpe=min_sharpe,
                max_drawdown=max_drawdown
            )
            logger.info(
                f"Updated thresholds for regime '{regime}': "
                f"min_sharpe={min_sharpe}, max_drawdown={max_drawdown}"
            )
