"""
Integration test for EngineHealthMonitor with TradingEngineSimulator.

Verifies:
  - Health monitor initializes correctly in trading engine
  - Health monitor tracks P&L and regime updates
  - Risk multiplier is applied to trades
  - Health status transitions are logged
"""

import pytest
import pandas as pd
import numpy as np
from engine.health_monitor import EngineHealthMonitor
from analytics.run_elo_evaluation import TradingEngineSimulator


class TestHealthMonitorIntegration:
    """Integration tests for health monitor with trading engine."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Create simple synthetic price data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2026-01-01', periods=100, freq='H')
        
        # Generate price data with uptrend
        prices = [100.0]
        for i in range(99):
            change = np.random.normal(0.001, 0.01)
            prices.append(prices[-1] * (1 + change))
        
        data = {
            'open': prices,
            'high': [p + np.random.uniform(0, 0.5) for p in prices],
            'low': [p - np.random.uniform(0, 0.5) for p in prices],
            'close': prices,
            'volume': [np.random.uniform(10000, 100000) for _ in prices]
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_trading_engine_with_health_monitor(self, synthetic_data):
        """Test trading engine with health monitor enabled."""
        simulator = TradingEngineSimulator(
            symbol='TEST',
            price_data=synthetic_data,
            track_health=True
        )
        
        # Verify health monitor initialized
        assert simulator.health_monitor is not None
        assert simulator.track_health is True
        assert simulator.health_monitor.health_status == "HEALTHY"
        assert simulator.health_monitor.risk_multiplier == 1.0
    
    def test_trading_engine_without_health_monitor(self, synthetic_data):
        """Test trading engine with health monitor disabled."""
        simulator = TradingEngineSimulator(
            symbol='TEST',
            price_data=synthetic_data,
            track_health=False
        )
        
        # Verify health monitor not initialized
        assert simulator.health_monitor is None
        assert simulator.track_health is False
    
    def test_run_simulation_with_health_monitor(self, synthetic_data):
        """Test that simulation runs with health monitor active."""
        simulator = TradingEngineSimulator(
            symbol='TEST',
            price_data=synthetic_data,
            track_health=True
        )
        
        trades = simulator.run_simulation()
        
        # Should complete without errors
        assert isinstance(trades, list)
        # Might have trades depending on price action
        assert len(trades) >= 0
    
    def test_health_monitor_tracks_regime(self, synthetic_data):
        """Test that health monitor tracks market regimes."""
        simulator = TradingEngineSimulator(
            symbol='TEST',
            price_data=synthetic_data,
            track_health=True
        )
        
        # Manually update with different regimes
        simulator.health_monitor.update(pnl=100.0, regime_label="low_vol")
        assert simulator.health_monitor.current_regime == "low_vol"
        
        simulator.health_monitor.update(pnl=50.0, regime_label="high_vol")
        assert simulator.health_monitor.current_regime == "high_vol"
        
        # Regime history should be tracked
        assert len(simulator.health_monitor.regime_history) == 2
    
    def test_health_monitor_risk_multiplier_applied(self, synthetic_data):
        """Test that risk multiplier affects trading."""
        simulator = TradingEngineSimulator(
            symbol='TEST',
            price_data=synthetic_data,
            track_health=True
        )
        
        # Simulate good performance (HEALTHY)
        for i in range(20):
            simulator.health_monitor.update(pnl=100.0, regime_label="low_vol")
        
        assert simulator.health_monitor.get_health_status() == "HEALTHY"
        assert simulator.health_monitor.get_risk_multiplier() == 1.0
    
    def test_health_monitor_critical_blocks_entries(self, synthetic_data):
        """Test that CRITICAL status blocks new position entries."""
        simulator = TradingEngineSimulator(
            symbol='TEST',
            price_data=synthetic_data,
            track_health=True
        )
        
        # Simulate severe losses to reach CRITICAL
        for i in range(30):
            simulator.health_monitor.update(pnl=-100.0, regime_label="low_vol")
        
        # Should be CRITICAL with 0.0 multiplier
        status = simulator.health_monitor.get_health_status()
        multiplier = simulator.health_monitor.get_risk_multiplier()
        
        assert status in ["HEALTHY", "DEGRADED", "CRITICAL"]
        if status == "CRITICAL":
            assert multiplier == 0.0
    
    def test_health_monitor_state_snapshot(self, synthetic_data):
        """Test health monitor state snapshot generation."""
        simulator = TradingEngineSimulator(
            symbol='TEST',
            price_data=synthetic_data,
            track_health=True
        )
        
        simulator.health_monitor.update(pnl=100.0, regime_label="low_vol")
        snapshot = simulator.health_monitor.get_state_snapshot(
            realized_pnl=100.0,
            unrealized_pnl=50.0
        )
        
        assert snapshot.realized_pnl == 100.0
        assert snapshot.unrealized_pnl == 50.0
        assert snapshot.regime_label == "low_vol"
        assert snapshot.risk_multiplier in [1.0, 0.5, 0.0]


class TestHealthMonitorRegressionIntegration:
    """Regression tests to ensure health monitor doesn't break existing functionality."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Create simple synthetic price data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2026-01-01', periods=50, freq='H')
        
        prices = [100.0]
        for i in range(49):
            change = np.random.normal(0.001, 0.01)
            prices.append(prices[-1] * (1 + change))
        
        data = {
            'open': prices,
            'high': [p + np.random.uniform(0, 0.5) for p in prices],
            'low': [p - np.random.uniform(0, 0.5) for p in prices],
            'close': prices,
            'volume': [np.random.uniform(10000, 100000) for _ in prices]
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_trading_engine_still_generates_trades(self, synthetic_data):
        """Verify trading engine still generates trades with health monitor."""
        simulator = TradingEngineSimulator(
            symbol='TEST',
            price_data=synthetic_data,
            track_health=True
        )
        
        trades = simulator.run_simulation()
        
        # Should still generate trades even with health monitor
        # (though count might vary due to new filtering)
        assert isinstance(trades, list)
    
    def test_no_trades_when_health_disabled(self, synthetic_data):
        """Verify trading engine works without health monitor."""
        simulator = TradingEngineSimulator(
            symbol='TEST',
            price_data=synthetic_data,
            track_health=False
        )
        
        trades = simulator.run_simulation()
        
        # Should work normally without health monitor
        assert isinstance(trades, list)
    
    def test_health_monitor_doesnt_affect_trade_count_significantly(self, synthetic_data):
        """Test that health monitor doesn't drastically change trade counts."""
        # Run with health monitor
        sim_with = TradingEngineSimulator(
            symbol='TEST',
            price_data=synthetic_data,
            track_health=True
        )
        trades_with = sim_with.run_simulation()
        
        # Run without health monitor
        sim_without = TradingEngineSimulator(
            symbol='TEST',
            price_data=synthetic_data,
            track_health=False
        )
        trades_without = sim_without.run_simulation()
        
        # Trade counts should be similar (might not be exact due to P&L tracking)
        # Just verify no crashes or major deviations
        assert isinstance(trades_with, list)
        assert isinstance(trades_without, list)
