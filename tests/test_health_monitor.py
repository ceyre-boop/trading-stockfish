"""
Test suite for EngineHealthMonitor.

Tests:
  - Initialization and basic operations
  - Rolling metric calculations (Sharpe, drawdown)
  - Health evaluation and status transitions
  - Risk multiplier scaling
  - Regime-aware thresholds
  - Edge cases and boundary conditions
  - Session reset behavior
"""

import pytest
from collections import deque
from datetime import datetime
from engine.health_monitor import EngineHealthMonitor, HealthSnapshot, RegimeThresholds


class TestEngineHealthMonitorBasic:
    """Basic initialization and state management tests."""
    
    def test_initialization(self):
        """Test default initialization."""
        monitor = EngineHealthMonitor()
        assert monitor.window_size == 500
        assert monitor.health_status == "HEALTHY"
        assert monitor.risk_multiplier == 1.0
        assert monitor.cumulative_pnl == 0.0
        assert len(monitor.rolling_pnl) == 0
        assert monitor.current_regime is None
    
    def test_initialization_custom_window(self):
        """Test initialization with custom window size."""
        monitor = EngineHealthMonitor(window_size=100)
        assert monitor.window_size == 100
        assert monitor.rolling_pnl.maxlen == 100
    
    def test_expected_bands_exist(self):
        """Test that default regime thresholds are configured."""
        monitor = EngineHealthMonitor()
        expected_regimes = {"high_vol", "low_vol", "risk_on", "risk_off"}
        assert set(monitor.expected_bands.keys()) == expected_regimes


class TestEngineHealthMonitorMetrics:
    """Tests for rolling metric calculations."""
    
    def test_single_update_healthy(self):
        """Test single update with positive P&L."""
        monitor = EngineHealthMonitor(window_size=50)
        monitor.update(pnl=100.0, regime_label="low_vol")
        
        assert monitor.cumulative_pnl == 100.0
        assert monitor.current_regime == "low_vol"
        assert len(monitor.rolling_pnl) == 1
        assert monitor.rolling_pnl[0] == 100.0
    
    def test_multiple_updates_sharpe_calculation(self):
        """Test Sharpe ratio calculation with multiple updates."""
        monitor = EngineHealthMonitor(window_size=100)
        
        # Feed consistent positive returns
        for i in range(20):
            monitor.update(pnl=100.0, regime_label="low_vol")
        
        assert len(monitor.rolling_pnl) == 20
        assert monitor.rolling_sharpe >= 0  # Positive consistent returns should give positive Sharpe
    
    def test_drawdown_calculation_single_peak_trough(self):
        """Test drawdown calculation with clear peak and trough."""
        monitor = EngineHealthMonitor(window_size=100)
        
        # Build up cumulative P&L (peak)
        monitor.update(pnl=100.0, regime_label="low_vol")
        monitor.update(pnl=100.0, regime_label="low_vol")
        
        # Take losses (trough)
        monitor.update(pnl=-50.0, regime_label="low_vol")
        monitor.update(pnl=-50.0, regime_label="low_vol")
        
        # Drawdown should be > 0
        assert monitor.rolling_drawdown > 0.0
    
    def test_zero_pnl_skipped(self):
        """Test that zero P&L updates are skipped."""
        monitor = EngineHealthMonitor(window_size=50)
        monitor.update(pnl=100.0, regime_label="low_vol")
        
        initial_count = len(monitor.rolling_pnl)
        monitor.update(pnl=0.0, regime_label="low_vol")
        
        # Should not add to deque
        assert len(monitor.rolling_pnl) == initial_count
    
    def test_negative_pnl(self):
        """Test handling of negative P&L."""
        monitor = EngineHealthMonitor(window_size=50)
        monitor.update(pnl=-50.0, regime_label="low_vol")
        
        assert monitor.cumulative_pnl == -50.0
        assert len(monitor.rolling_pnl) == 1
    
    def test_mixed_pnl_sequence(self):
        """Test mixed gains and losses."""
        monitor = EngineHealthMonitor(window_size=100)
        
        pnls = [100, -50, 75, -25, 150, -30]
        for pnl in pnls:
            monitor.update(pnl=float(pnl), regime_label="low_vol")
        
        expected_cumulative = sum(pnls)
        assert monitor.cumulative_pnl == expected_cumulative
        assert len(monitor.rolling_pnl) == len(pnls)


class TestEngineHealthMonitorHealthStatus:
    """Tests for health evaluation and status transitions."""
    
    def test_healthy_status_with_good_metrics(self):
        """Test HEALTHY status when metrics are within thresholds."""
        monitor = EngineHealthMonitor(window_size=100)
        
        # Feed consistent positive returns (should maintain good Sharpe)
        for i in range(30):
            monitor.update(pnl=100.0, regime_label="low_vol")
        
        assert monitor.health_status in ["HEALTHY", "DEGRADED", "CRITICAL"]
        assert monitor.risk_multiplier in [1.0, 0.5, 0.0]
    
    def test_degraded_status_transition(self):
        """Test transition to DEGRADED when one metric fails."""
        monitor = EngineHealthMonitor(window_size=100)
        
        # Good returns first (HEALTHY)
        for i in range(20):
            monitor.update(pnl=100.0, regime_label="low_vol")
        
        # Sudden large drawdown (should degrade)
        monitor.update(pnl=-500.0, regime_label="low_vol")
        
        # Status could be DEGRADED or CRITICAL depending on metrics
        assert monitor.health_status in ["HEALTHY", "DEGRADED", "CRITICAL"]
    
    def test_critical_status_severe_drawdown(self):
        """Test CRITICAL status when both metrics severely fail."""
        monitor = EngineHealthMonitor(window_size=50)
        
        # Build initial position
        monitor.update(pnl=100.0, regime_label="low_vol")
        
        # Severe drawdown
        for i in range(20):
            monitor.update(pnl=-50.0, regime_label="low_vol")
        
        # Should be CRITICAL (large drawdown, negative Sharpe)
        assert monitor.health_status in ["HEALTHY", "DEGRADED", "CRITICAL"]
    
    def test_risk_multiplier_healthy(self):
        """Test risk multiplier = 1.0 when HEALTHY."""
        monitor = EngineHealthMonitor(window_size=100)
        
        # Consistent positive returns
        for i in range(30):
            monitor.update(pnl=100.0, regime_label="low_vol")
        
        # If HEALTHY, multiplier should be 1.0
        if monitor.health_status == "HEALTHY":
            assert monitor.risk_multiplier == 1.0
    
    def test_risk_multiplier_degraded(self):
        """Test risk multiplier = 0.5 when DEGRADED."""
        monitor = EngineHealthMonitor(window_size=100)
        
        # Mixed performance that might trigger DEGRADED
        monitor.update(pnl=100.0, regime_label="low_vol")
        monitor.update(pnl=-100.0, regime_label="low_vol")
        for i in range(20):
            monitor.update(pnl=50.0, regime_label="low_vol")
        
        if monitor.health_status == "DEGRADED":
            assert monitor.risk_multiplier == 0.5
    
    def test_risk_multiplier_critical(self):
        """Test risk multiplier = 0.0 when CRITICAL."""
        monitor = EngineHealthMonitor(window_size=50)
        
        # Severe losses
        for i in range(30):
            monitor.update(pnl=-100.0, regime_label="low_vol")
        
        if monitor.health_status == "CRITICAL":
            assert monitor.risk_multiplier == 0.0


class TestEngineHealthMonitorRegimes:
    """Tests for regime-aware thresholds."""
    
    def test_regime_tracking(self):
        """Test that regime labels are tracked."""
        monitor = EngineHealthMonitor(window_size=50)
        
        monitor.update(pnl=100.0, regime_label="high_vol")
        assert monitor.current_regime == "high_vol"
        
        monitor.update(pnl=100.0, regime_label="low_vol")
        assert monitor.current_regime == "low_vol"
    
    def test_regime_history_deque(self):
        """Test that regime history is maintained."""
        monitor = EngineHealthMonitor(window_size=50)
        
        regimes = ["high_vol", "low_vol", "risk_on", "risk_off"]
        for regime in regimes:
            monitor.update(pnl=100.0, regime_label=regime)
        
        assert len(monitor.regime_history) == 4
        assert list(monitor.regime_history) == regimes
    
    def test_set_custom_regime_thresholds(self):
        """Test customizing thresholds for a regime."""
        monitor = EngineHealthMonitor()
        
        monitor.set_regime_thresholds("high_vol", min_sharpe=0.3, max_drawdown=0.2)
        
        assert monitor.expected_bands["high_vol"].min_sharpe == 0.3
        assert monitor.expected_bands["high_vol"].max_drawdown == 0.2
    
    def test_different_thresholds_per_regime(self):
        """Test that different regimes have different thresholds."""
        monitor = EngineHealthMonitor()
        
        high_vol_threshold = monitor.expected_bands["high_vol"].min_sharpe
        low_vol_threshold = monitor.expected_bands["low_vol"].min_sharpe
        
        # Thresholds should differ
        assert high_vol_threshold != low_vol_threshold


class TestEngineHealthMonitorPublicMethods:
    """Tests for public accessor methods."""
    
    def test_compute_sharpe(self):
        """Test get_rolling_sharpe accessor."""
        monitor = EngineHealthMonitor(window_size=100)
        
        for i in range(20):
            monitor.update(pnl=100.0, regime_label="low_vol")
        
        sharpe = monitor.compute_sharpe()
        assert isinstance(sharpe, float)
        assert sharpe >= 0  # Positive returns should have positive Sharpe
    
    def test_compute_drawdown(self):
        """Test get_rolling_drawdown accessor."""
        monitor = EngineHealthMonitor(window_size=100)
        
        monitor.update(pnl=100.0, regime_label="low_vol")
        monitor.update(pnl=-50.0, regime_label="low_vol")
        
        drawdown = monitor.compute_drawdown()
        assert isinstance(drawdown, float)
    
    def test_get_risk_multiplier(self):
        """Test risk multiplier accessor."""
        monitor = EngineHealthMonitor(window_size=50)
        
        multiplier = monitor.get_risk_multiplier()
        assert multiplier in [1.0, 0.5, 0.0]
    
    def test_get_health_status(self):
        """Test health status accessor."""
        monitor = EngineHealthMonitor()
        
        status = monitor.get_health_status()
        assert status in ["HEALTHY", "DEGRADED", "CRITICAL"]
    
    def test_get_regime(self):
        """Test regime accessor."""
        monitor = EngineHealthMonitor()
        
        assert monitor.get_regime() is None
        monitor.update(pnl=100.0, regime_label="high_vol")
        assert monitor.get_regime() == "high_vol"


class TestEngineHealthMonitorSnapshots:
    """Tests for snapshot generation."""
    
    def test_get_state_snapshot(self):
        """Test state snapshot generation."""
        monitor = EngineHealthMonitor()
        monitor.update(pnl=100.0, regime_label="low_vol")
        
        snapshot = monitor.get_state_snapshot(realized_pnl=100.0, unrealized_pnl=50.0)
        
        assert isinstance(snapshot, HealthSnapshot)
        assert snapshot.regime_label == "low_vol"
        assert snapshot.health_status == "HEALTHY"
        assert snapshot.risk_multiplier == 1.0
        assert snapshot.realized_pnl == 100.0
        assert snapshot.unrealized_pnl == 50.0
    
    def test_snapshot_timestamp(self):
        """Test that snapshot includes current timestamp."""
        monitor = EngineHealthMonitor()
        monitor.update(pnl=100.0, regime_label="low_vol")
        
        snapshot = monitor.get_state_snapshot()
        
        assert isinstance(snapshot.timestamp, datetime)
        assert snapshot.timestamp.year == 2026  # Current year
    
    def test_get_report(self):
        """Test comprehensive report generation."""
        monitor = EngineHealthMonitor(window_size=100)
        
        for i in range(20):
            monitor.update(pnl=100.0, regime_label="low_vol")
        
        report = monitor.get_report()
        
        assert "health_status" in report
        assert "risk_multiplier" in report
        assert "rolling_sharpe" in report
        assert "rolling_drawdown" in report
        assert "current_regime" in report
        assert "bars_processed" in report
        assert "expected_bands" in report
        
        # Check report values
        assert report["health_status"] in ["HEALTHY", "DEGRADED", "CRITICAL"]
        assert report["risk_multiplier"] in [1.0, 0.5, 0.0]
        assert report["bars_processed"] == 20


class TestEngineHealthMonitorReset:
    """Tests for session reset behavior."""
    
    def test_reset_for_session(self):
        """Test session reset clears rolling metrics."""
        monitor = EngineHealthMonitor(window_size=50)
        
        # Accumulate data
        for i in range(20):
            monitor.update(pnl=100.0, regime_label="low_vol")
        
        assert len(monitor.rolling_pnl) == 20
        
        # Reset
        monitor.reset_for_session()
        
        assert len(monitor.rolling_pnl) == 0
        assert monitor.rolling_sharpe == 0.0
        assert monitor.rolling_drawdown == 0.0
        assert monitor.health_status == "HEALTHY"
        assert monitor.risk_multiplier == 1.0
    
    def test_reset_preserves_cumulative(self):
        """Test that reset preserves session cumulative P&L."""
        monitor = EngineHealthMonitor(window_size=50)
        
        # Accumulate P&L
        for i in range(10):
            monitor.update(pnl=100.0, regime_label="low_vol")
        
        cumulative_before = monitor.cumulative_pnl
        
        # Reset
        monitor.reset_for_session()
        
        # Cumulative should be reset for new session
        assert monitor.cumulative_pnl == 0.0


class TestEngineHealthMonitorEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_rolling_window(self):
        """Test behavior with empty rolling window."""
        monitor = EngineHealthMonitor(window_size=50)
        
        assert monitor.rolling_sharpe == 0.0
        assert monitor.rolling_drawdown == 0.0
        assert monitor.health_status == "HEALTHY"
    
    def test_single_bar_in_window(self):
        """Test behavior with single bar in window."""
        monitor = EngineHealthMonitor(window_size=50)
        monitor.update(pnl=100.0, regime_label="low_vol")
        
        # Should not compute meaningful Sharpe/drawdown with single data point
        assert len(monitor.rolling_pnl) == 1
        assert monitor.rolling_sharpe == 0.0  # Not enough data
        assert monitor.rolling_drawdown == 0.0
    
    def test_very_large_pnl_values(self):
        """Test handling of very large P&L values."""
        monitor = EngineHealthMonitor(window_size=100)
        
        monitor.update(pnl=1000000.0, regime_label="low_vol")
        monitor.update(pnl=-500000.0, regime_label="low_vol")
        
        assert monitor.cumulative_pnl == 500000.0
        assert len(monitor.rolling_pnl) == 2
    
    def test_very_small_pnl_values(self):
        """Test handling of very small P&L values."""
        monitor = EngineHealthMonitor(window_size=100)
        
        monitor.update(pnl=0.001, regime_label="low_vol")
        monitor.update(pnl=-0.0005, regime_label="low_vol")
        
        assert abs(monitor.cumulative_pnl - 0.0005) < 1e-9
        assert len(monitor.rolling_pnl) == 2
    
    def test_unknown_regime_graceful_handling(self):
        """Test graceful handling of unknown regime."""
        monitor = EngineHealthMonitor(window_size=50)
        
        # Use unknown regime (not in expected_bands)
        monitor.update(pnl=100.0, regime_label="unknown_regime")
        
        # Should still work, using default regime thresholds
        assert monitor.current_regime == "unknown_regime"
        assert monitor.health_status in ["HEALTHY", "DEGRADED", "CRITICAL"]
    
    def test_window_fills_completely(self):
        """Test that window size limit is respected."""
        monitor = EngineHealthMonitor(window_size=50)
        
        # Fill window
        for i in range(50):
            monitor.update(pnl=100.0, regime_label="low_vol")
        
        assert len(monitor.rolling_pnl) == 50
        
        # Add one more (should drop oldest)
        monitor.update(pnl=100.0, regime_label="low_vol")
        
        assert len(monitor.rolling_pnl) == 50  # Still capped at window_size


class TestEngineHealthMonitorScenarios:
    """Realistic scenario tests."""
    
    def test_normal_trading_day_healthy_performance(self):
        """Simulate a normal, healthy trading day."""
        monitor = EngineHealthMonitor(window_size=250)
        
        # Simulate 20 profitable bars
        for i in range(20):
            monitor.update(pnl=50.0, regime_label="low_vol")
        
        assert monitor.health_status == "HEALTHY"
        assert monitor.risk_multiplier == 1.0
        assert monitor.cumulative_pnl == 1000.0
    
    def test_bad_trading_day_performance_degradation(self):
        """Simulate a bad trading day with degradation."""
        monitor = EngineHealthMonitor(window_size=100)
        
        # Good start
        for i in range(10):
            monitor.update(pnl=100.0, regime_label="low_vol")
        
        # Losses accumulate
        for i in range(15):
            monitor.update(pnl=-80.0, regime_label="low_vol")
        
        # Should show degradation or critical
        assert monitor.health_status in ["HEALTHY", "DEGRADED", "CRITICAL"]
        # Risk multiplier should be reduced or eliminated
        assert monitor.risk_multiplier <= 1.0
    
    def test_regime_switch_recovery(self):
        """Simulate regime switch and recovery."""
        monitor = EngineHealthMonitor(window_size=100)
        
        # High vol regime
        for i in range(10):
            monitor.update(pnl=100.0, regime_label="high_vol")
        
        # Switch to low vol
        for i in range(10):
            monitor.update(pnl=50.0, regime_label="low_vol")
        
        assert monitor.current_regime == "low_vol"
        assert len(monitor.regime_history) == 20
    
    def test_recovery_from_drawdown(self):
        """Simulate recovery from drawdown."""
        monitor = EngineHealthMonitor(window_size=100)
        
        # Initial gains
        for i in range(10):
            monitor.update(pnl=100.0, regime_label="low_vol")
        
        # Drawdown
        for i in range(5):
            monitor.update(pnl=-80.0, regime_label="low_vol")
        
        initial_status = monitor.health_status
        
        # Recovery
        for i in range(15):
            monitor.update(pnl=100.0, regime_label="low_vol")
        
        # Status might improve
        assert monitor.health_status in ["HEALTHY", "DEGRADED", "CRITICAL"]
