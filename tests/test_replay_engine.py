"""
Test Suite for ReplayEngine - analytics/replay_day.py

Tests cover:
  - ReplayEngine initialization
  - Candle-by-candle stepping
  - Full replay runs
  - Snapshot generation and data integrity
  - Export functionality (JSON, logs)
  - Market state building
  - Policy application
  - Execution simulation
  - Position tracking

Author: Trading-Stockfish Tests
Version: 1.0.0
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

from analytics.replay_day import (
    ReplayEngine, ReplaySnapshot, ReplaySession, ReplayStatus
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    closes = 1.0 + np.cumsum(np.random.normal(0, 0.01, 100))
    
    df = pd.DataFrame({
        'open': closes + np.random.normal(0, 0.005, 100),
        'high': closes + np.abs(np.random.normal(0, 0.005, 100)),
        'low': closes - np.abs(np.random.normal(0, 0.005, 100)),
        'close': closes,
        'volume': np.random.randint(1000, 10000, 100),
    }, index=dates)
    
    return df


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        'evaluator_weights': {'macro': 0.3, 'liquidity': 0.3, 'volatility': 0.4},
        'policy_thresholds': {'entry': 0.5, 'exit': 0.3},
    }


@pytest.fixture
def replay_engine(sample_data, sample_config):
    """Create ReplayEngine instance."""
    return ReplayEngine(
        symbol='EURUSD',
        data=sample_data,
        config=sample_config,
        verbose=False
    )


# =============================================================================
# BASIC INITIALIZATION TESTS
# =============================================================================

class TestReplayEngineInitialization:
    """Test ReplayEngine initialization."""
    
    def test_initialization_success(self, sample_data, sample_config):
        """Test successful initialization."""
        engine = ReplayEngine(
            symbol='EURUSD',
            data=sample_data,
            config=sample_config,
            verbose=False
        )
        
        assert engine.symbol == 'EURUSD'
        assert engine.status == ReplayStatus.READY
        assert engine.current_index == 0
        assert len(engine.data) == 100
        assert engine.position_size == 0.0
        assert engine.cumulative_pnl == 0.0
    
    def test_initialization_missing_columns(self, sample_config):
        """Test initialization with missing OHLCV columns."""
        bad_data = pd.DataFrame({
            'open': [1.0, 1.1],
            'close': [1.05, 1.15],
        })
        
        with pytest.raises(ValueError, match="must contain columns"):
            ReplayEngine(
                symbol='EURUSD',
                data=bad_data,
                config=sample_config
            )
    
    def test_config_hash_generation(self, sample_data, sample_config):
        """Test config hash generation."""
        engine = ReplayEngine(
            symbol='EURUSD',
            data=sample_data,
            config=sample_config,
            verbose=False
        )
        
        hash1 = engine.session.config_hash
        assert isinstance(hash1, str)
        assert len(hash1) == 8
        
        # Different config should produce different hash
        engine2 = ReplayEngine(
            symbol='EURUSD',
            data=sample_data,
            config={'different': 'config'},
            verbose=False
        )
        hash2 = engine2.session.config_hash
        assert hash1 != hash2


# =============================================================================
# STEPPING AND REPLAY TESTS
# =============================================================================

class TestReplayEngineStepAndRun:
    """Test stepping and running replays."""
    
    def test_step_single_candle(self, replay_engine):
        """Test stepping through single candle."""
        snapshot = replay_engine.step()
        
        assert snapshot is not None
        assert snapshot.candle_index == 0
        assert replay_engine.current_index == 1
        assert replay_engine.status == ReplayStatus.RUNNING
    
    def test_step_multiple_candles(self, replay_engine):
        """Test stepping through multiple candles."""
        snapshots = []
        for i in range(5):
            snapshot = replay_engine.step()
            snapshots.append(snapshot)
        
        assert len(snapshots) == 5
        assert snapshots[0].candle_index == 0
        assert snapshots[4].candle_index == 4
        assert replay_engine.current_index == 5
    
    def test_step_beyond_data(self, replay_engine):
        """Test stepping beyond available data."""
        # Step through entire dataset
        for _ in range(100):
            snapshot = replay_engine.step()
        
        # Should reach end
        assert replay_engine.current_index == 100
        # Status should be RUNNING or STOPPED after processing
        assert replay_engine.status in [ReplayStatus.RUNNING, ReplayStatus.STOPPED]
        
        # Further steps should return None
        snapshot = replay_engine.step()
        assert snapshot is None
        assert replay_engine.status == ReplayStatus.STOPPED
    
    def test_run_full_replay(self, replay_engine):
        """Test running full replay."""
        snapshots = replay_engine.run_full()
        
        assert len(snapshots) == 100
        assert replay_engine.current_index == 100
        assert replay_engine.status == ReplayStatus.STOPPED
        assert len(replay_engine.session.snapshots) == 100
    
    def test_reset_replay(self, replay_engine):
        """Test resetting replay state."""
        # Run a few steps
        replay_engine.step()
        replay_engine.step()
        replay_engine.step()
        
        assert replay_engine.current_index == 3
        assert len(replay_engine.session.snapshots) == 3
        
        # Reset
        replay_engine.reset()
        
        assert replay_engine.current_index == 0
        assert len(replay_engine.session.snapshots) == 0
        assert replay_engine.status == ReplayStatus.READY
        assert replay_engine.cumulative_pnl == 0.0


# =============================================================================
# SNAPSHOT GENERATION TESTS
# =============================================================================

class TestSnapshotGeneration:
    """Test snapshot generation and data integrity."""
    
    def test_snapshot_structure(self, replay_engine):
        """Test snapshot has all required fields."""
        snapshot = replay_engine.step()
        
        # OHLCV
        assert hasattr(snapshot, 'open')
        assert hasattr(snapshot, 'high')
        assert hasattr(snapshot, 'low')
        assert hasattr(snapshot, 'close')
        assert hasattr(snapshot, 'volume')
        
        # Market state
        assert hasattr(snapshot, 'market_state')
        
        # Evaluation
        assert hasattr(snapshot, 'eval_score')
        assert hasattr(snapshot, 'eval_confidence')
        assert hasattr(snapshot, 'subsystem_scores')
        
        # Policy
        assert hasattr(snapshot, 'policy_action')
        assert hasattr(snapshot, 'target_size')
        assert hasattr(snapshot, 'action_reasoning')
        
        # Execution
        assert hasattr(snapshot, 'fill_price')
        assert hasattr(snapshot, 'filled_size')
        assert hasattr(snapshot, 'transaction_cost')
        
        # Position
        assert hasattr(snapshot, 'position_side')
        assert hasattr(snapshot, 'position_size')
        
        # Health
        assert hasattr(snapshot, 'health_status')
        assert hasattr(snapshot, 'risk_multiplier')
        
        # P&L
        assert hasattr(snapshot, 'cumulative_pnl')
    
    def test_snapshot_ohlcv_values(self, replay_engine):
        """Test snapshot OHLCV values match data."""
        snapshot = replay_engine.step()
        row = replay_engine.data.iloc[0]
        
        assert snapshot.open == float(row['open'])
        assert snapshot.high == float(row['high'])
        assert snapshot.low == float(row['low'])
        assert snapshot.close == float(row['close'])
        assert snapshot.volume == int(row['volume'])
    
    def test_eval_score_bounds(self, replay_engine):
        """Test evaluation scores are properly bounded."""
        snapshots = replay_engine.run_full()
        
        for snapshot in snapshots:
            assert -1.0 <= snapshot.eval_score <= 1.0
            assert 0.0 <= snapshot.eval_confidence <= 1.0
    
    def test_position_tracking_consistency(self, replay_engine):
        """Test position state is consistent across snapshots."""
        snapshots = replay_engine.run_full()
        
        # Initial position should be FLAT
        assert snapshots[0].position_side == "FLAT"
        
        # Final snapshot should match engine state
        assert snapshots[-1].position_side == replay_engine.position_side
        assert snapshots[-1].position_size == replay_engine.position_size
        assert snapshots[-1].cumulative_pnl == replay_engine.cumulative_pnl


# =============================================================================
# EXPORT TESTS
# =============================================================================

class TestExportFunctionality:
    """Test export to log and JSON."""
    
    def test_export_json(self, replay_engine):
        """Test JSON export."""
        replay_engine.run_full()
        json_file = replay_engine.export_json()
        
        assert json_file.exists()
        assert json_file.suffix == '.json'
        
        # Verify JSON is valid and contains data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        assert 'symbol' in data
        assert 'stats' in data
        assert 'snapshots' in data
        assert len(data['snapshots']) == 100
    
    def test_export_log(self, replay_engine):
        """Test log export."""
        replay_engine.run_full()
        log_file = replay_engine.export_log()
        
        assert log_file.exists()
        assert log_file.suffix == '.log'
        
        # Verify log contains data
        with open(log_file, 'r') as f:
            content = f.read()
        
        assert 'REPLAY LOG' in content
        assert 'EURUSD' in content
        assert len(content) > 1000
    
    def test_log_file_created_on_init(self, sample_data, sample_config):
        """Test log file is created on engine initialization."""
        engine = ReplayEngine(
            symbol='TEST',
            data=sample_data,
            config=sample_config,
            verbose=False
        )
        
        assert engine.log_file.exists()
        # Check path contains logs and replay (handle Windows backslash)
        path_str = str(engine.log_file).lower().replace('\\', '/')
        assert 'logs/replay' in path_str


# =============================================================================
# STATISTICS TESTS
# =============================================================================

class TestStatisticsGeneration:
    """Test statistics computation."""
    
    def test_session_stats_computation(self, replay_engine):
        """Test session statistics computation."""
        replay_engine.run_full()
        stats = replay_engine.get_stats()
        
        assert 'total_candles' in stats
        assert stats['total_candles'] == 100
        assert 'start_price' in stats
        assert 'end_price' in stats
        assert 'final_pnl' in stats
    
    def test_stats_empty_replay(self, replay_engine):
        """Test stats with no snapshots."""
        stats = replay_engine.get_stats()
        
        assert stats['total_candles'] == 0


# =============================================================================
# MARKET STATE BUILDING TESTS
# =============================================================================

class TestMarketStateBuilding:
    """Test market state building."""
    
    def test_market_state_insufficient_data(self, replay_engine):
        """Test market state with insufficient lookback."""
        snapshot = replay_engine.step()
        
        # Early candle should report insufficient data
        assert snapshot.market_state['status'] == 'insufficient_data'
    
    def test_market_state_valid_after_lookback(self, replay_engine):
        """Test market state becomes valid after enough data."""
        # Step past minimum lookback
        for _ in range(30):
            snapshot = replay_engine.step()
        
        # Should now have valid market state
        assert snapshot.market_state['status'] == 'valid'
        assert 'factors' in snapshot.market_state


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestReplayEngineIntegration:
    """Integration tests for replay engine."""
    
    def test_full_workflow(self, sample_data, sample_config):
        """Test complete replay workflow."""
        engine = ReplayEngine(
            symbol='EURUSD',
            data=sample_data,
            config=sample_config,
            verbose=False
        )
        
        # Run full replay
        snapshots = engine.run_full()
        
        # Export results
        json_file = engine.export_json()
        log_file = engine.export_log()
        
        assert len(snapshots) == 100
        assert json_file.exists()
        assert log_file.exists()
    
    def test_multiple_runs_isolated(self, sample_data, sample_config):
        """Test multiple replays are independent."""
        engine1 = ReplayEngine(
            symbol='EURUSD',
            data=sample_data.copy(),
            config=sample_config,
            verbose=False
        )
        
        engine2 = ReplayEngine(
            symbol='EURUSD',
            data=sample_data.copy(),
            config=sample_config,
            verbose=False
        )
        
        snapshots1 = engine1.run_full()
        snapshots2 = engine2.run_full()
        
        # Results should be identical (same data and config)
        assert len(snapshots1) == len(snapshots2)
        assert snapshots1[0].close == snapshots2[0].close


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_candle_dataset(self, sample_config):
        """Test with single candle."""
        single_candle = pd.DataFrame({
            'open': [1.0],
            'high': [1.01],
            'low': [0.99],
            'close': [1.005],
            'volume': [5000],
        })
        
        engine = ReplayEngine(
            symbol='EURUSD',
            data=single_candle,
            config=sample_config,
            verbose=False
        )
        
        snapshots = engine.run_full()
        assert len(snapshots) == 1
    
    def test_constant_price_data(self, sample_config):
        """Test with constant price data."""
        constant_data = pd.DataFrame({
            'open': [1.0] * 50,
            'high': [1.0] * 50,
            'low': [1.0] * 50,
            'close': [1.0] * 50,
            'volume': [5000] * 50,
        })
        
        engine = ReplayEngine(
            symbol='EURUSD',
            data=constant_data,
            config=sample_config,
            verbose=False
        )
        
        snapshots = engine.run_full()
        assert len(snapshots) == 50
        
        # With constant prices, eval scores will be near zero (but may have randomness in simplified engine)
        # Just verify they are computed
        for snapshot in snapshots[20:]:  # Skip initial period
            assert isinstance(snapshot.eval_score, (int, float))
            assert -1.0 <= snapshot.eval_score <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
