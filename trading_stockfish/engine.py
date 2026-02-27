"""
BenchmarkEngine: orchestrates a full benchmarking run.

Usage
-----
    from trading_stockfish import BenchmarkEngine
    engine = BenchmarkEngine()
    result = engine.run(prices)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .execution.tracker import ExecutionTracker, Fill
from .regime.detector import RegimeDetector, RegimeConfig
from .reporting.logger import ExperimentLogger
from .risk.governor import RiskGovernor, RiskConfig
from .signals.generator import SignalGenerator, SignalConfig, SIGNAL_FLAT


@dataclass
class EngineConfig:
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    simulated_slippage_bps: float = 2.0    # per-fill slippage to simulate
    simulated_latency_ms: float = 5.0      # per-fill latency to simulate
    initial_equity: float = 100_000.0
    log_path: Optional[str] = None         # None → no file logging


@dataclass
class RunResult:
    pnl_series: List[float]
    equity_curve: List[float]
    regime_series: List[int]
    signal_series: List[int]
    execution_stats: Dict[str, Any]
    final_equity: float
    total_return: float     # fraction, e.g. 0.05 = +5 %
    max_drawdown: float
    sharpe_ratio: float
    regime_breakdown: Dict[str, int]  # LOW/NORMAL/HIGH counts


class BenchmarkEngine:
    """
    End-to-end benchmarking engine for intraday futures strategies.

    Parameters
    ----------
    config : EngineConfig, optional
    experiment_id : str, optional
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        experiment_id: str = "default",
    ) -> None:
        self.config = config or EngineConfig()
        self.experiment_id = experiment_id
        self._regime   = RegimeDetector(self.config.regime)
        self._risk     = RiskGovernor(self.config.risk)
        self._signals  = SignalGenerator(self.config.signal)
        self._exec     = ExecutionTracker()
        self._logger   = (
            ExperimentLogger(self.config.log_path)
            if self.config.log_path
            else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, prices: Sequence[float]) -> RunResult:
        """
        Simulate a strategy run over the provided price series.

        prices : sequence of bar close prices (length ≥ 2)
        """
        px = np.asarray(prices, dtype=float)
        if len(px) < 2:
            raise ValueError("prices must contain at least 2 bars")

        self._risk.reset()
        self._exec.reset()

        regimes = self._regime.detect(px)
        signals = self._signals.combined(px, regime=int(np.median(regimes)))

        equity = self.config.initial_equity
        equity_curve: List[float] = [equity]
        pnl_series:   List[float] = []
        max_dd = 0.0
        order_id = 0

        for i in range(1, len(px)):
            sig = signals[i - 1]   # signal generated on previous bar

            if sig == SIGNAL_FLAT or self._risk.is_halted(equity / self.config.initial_equity):
                pnl = 0.0
            else:
                # Simulate fill with slippage
                slip_frac = self.config.simulated_slippage_bps / 10_000
                filled_px = px[i] * (1 + slip_frac * sig)   # adverse slippage
                stop_dist = slip_frac * 10                    # illustrative stop
                size = self._risk.position_size(
                    equity / self.config.initial_equity, stop_dist
                )

                self._exec.record_attempt()
                fill = Fill(
                    order_id=str(order_id),
                    intended_price=px[i],
                    filled_price=filled_px,
                    quantity=size,
                    side="BUY" if sig > 0 else "SELL",
                    latency_ms=self.config.simulated_latency_ms,
                )
                self._exec.record_fill(fill)
                order_id += 1

                price_return = (px[i] - px[i - 1]) / px[i - 1] if px[i - 1] != 0 else 0.0
                pnl = sig * price_return * equity * size

            equity += pnl
            equity_curve.append(equity)
            pnl_series.append(pnl)
            dd = self._risk.drawdown(equity / self.config.initial_equity)
            max_dd = max(max_dd, dd)

        exec_stats_obj = self._exec.stats()
        exec_stats = {
            "total_fills":       exec_stats_obj.total_fills,
            "fill_rate":         exec_stats_obj.fill_rate,
            "mean_slippage_bps": exec_stats_obj.mean_slippage_bps,
            "mean_latency_ms":   exec_stats_obj.mean_latency_ms,
        }

        total_return = (equity - self.config.initial_equity) / self.config.initial_equity
        sharpe = self._sharpe(pnl_series)

        from .regime.detector import REGIME_LOW, REGIME_NORMAL, REGIME_HIGH
        regime_labels = {str(REGIME_LOW): "LOW_VOL", str(REGIME_NORMAL): "NORMAL", str(REGIME_HIGH): "HIGH_VOL"}
        unique, counts = np.unique(regimes, return_counts=True)
        regime_breakdown = {
            regime_labels.get(str(int(u)), str(int(u))): int(c)
            for u, c in zip(unique, counts)
        }

        result = RunResult(
            pnl_series=pnl_series,
            equity_curve=equity_curve,
            regime_series=regimes.tolist(),
            signal_series=signals.tolist(),
            execution_stats=exec_stats,
            final_equity=float(equity),
            total_return=float(total_return),
            max_drawdown=float(max_dd),
            sharpe_ratio=float(sharpe),
            regime_breakdown=regime_breakdown,
        )

        if self._logger is not None:
            from dataclasses import asdict
            self._logger.log(
                experiment_id=self.experiment_id,
                config={"engine": str(self.config)},
                metrics={
                    "total_return":  result.total_return,
                    "max_drawdown":  result.max_drawdown,
                    "sharpe_ratio":  result.sharpe_ratio,
                    "total_fills":   exec_stats["total_fills"],
                    "mean_slippage_bps": exec_stats["mean_slippage_bps"],
                },
            )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sharpe(pnl_series: List[float], bars_per_year: int = 252) -> float:
        arr = np.asarray(pnl_series, dtype=float)
        if len(arr) < 2 or arr.std() == 0:
            return 0.0
        return float(arr.mean() / arr.std() * np.sqrt(bars_per_year))
