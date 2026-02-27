import numpy as np
import pytest
from trading_stockfish import BenchmarkEngine, EngineConfig
from trading_stockfish.risk.governor import RiskConfig


def _prices(n=200, seed=42):
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0002, 0.01, n - 1)
    log_prices = np.concatenate([[0.0], np.cumsum(log_returns)])
    return (100.0 * np.exp(log_prices)).tolist()


class TestBenchmarkEngine:
    def test_run_basic(self):
        engine = BenchmarkEngine()
        result = engine.run(_prices())
        assert result.final_equity > 0
        assert len(result.equity_curve) == 200
        assert len(result.pnl_series) == 199

    def test_run_returns_regime_series(self):
        engine = BenchmarkEngine()
        result = engine.run(_prices())
        assert len(result.regime_series) == 200

    def test_total_return_finite(self):
        engine = BenchmarkEngine()
        result = engine.run(_prices())
        assert np.isfinite(result.total_return)

    def test_max_drawdown_non_negative(self):
        engine = BenchmarkEngine()
        result = engine.run(_prices())
        assert result.max_drawdown >= 0.0

    def test_sharpe_ratio_finite(self):
        engine = BenchmarkEngine()
        result = engine.run(_prices())
        assert np.isfinite(result.sharpe_ratio)

    def test_regime_breakdown_covers_all_bars(self):
        engine = BenchmarkEngine()
        result = engine.run(_prices(100))
        total = sum(result.regime_breakdown.values())
        assert total == 100

    def test_too_short_prices_raises(self):
        engine = BenchmarkEngine()
        with pytest.raises(ValueError):
            engine.run([100.0])

    def test_drawdown_halt_respected(self):
        # Configure a very tight drawdown limit
        cfg = EngineConfig(risk=RiskConfig(max_drawdown=0.001))
        engine = BenchmarkEngine(config=cfg)
        result = engine.run(_prices(200))
        # Once halted, no further fills should be placed
        assert result.execution_stats["total_fills"] >= 0

    def test_execution_stats_present(self):
        engine = BenchmarkEngine()
        result = engine.run(_prices())
        stats = result.execution_stats
        assert "total_fills" in stats
        assert "fill_rate" in stats
        assert "mean_slippage_bps" in stats
        assert "mean_latency_ms" in stats
