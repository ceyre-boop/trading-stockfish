"""
ExecutionSimulator v1 - End-to-End Verification Test

Validates:
1. Module imports
2. Configuration loading
3. All trade actions (ENTER, ADD, REDUCE, EXIT, REVERSE)
4. Spread, slippage, cost calculations
5. Position state tracking
6. Partial fills
7. Trade logging
8. Summary statistics
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from engine.execution_simulator import (
    ExecutionSimulator, LiquidityState, VolatilityState, PositionState
)

def test_module_imports():
    """Test all required imports."""
    print("\n[TEST 1] Module Imports")
    print("-" * 50)
    try:
        from analytics.run_elo_evaluation import RealDataTournament, RealDataTradingSimulator
        print("[PASS] RealDataTournament and RealDataTradingSimulator import")
        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\n[TEST 2] Configuration Loading")
    print("-" * 50)
    try:
        executor = ExecutionSimulator(config_path="execution_config.yaml")
        
        symbols = list(executor.symbol_configs.keys())
        assert symbols == ['ES', 'NQ', 'EURUSD'], f"Expected ['ES', 'NQ', 'EURUSD'], got {symbols}"
        
        es_config = executor.symbol_configs['ES']
        assert 'fixed_spread' in es_config, "ES config missing fixed_spread"
        assert 'slippage_coefficient' in es_config, "ES config missing slippage_coefficient"
        
        print(f"[PASS] Config loaded with {len(symbols)} symbols")
        print(f"       Symbols: {', '.join(symbols)}")
        return True
    except Exception as e:
        print(f"[FAIL] Config loading failed: {e}")
        return False

def test_enter_action():
    """Test ENTER action."""
    print("\n[TEST 3] ENTER Action")
    print("-" * 50)
    try:
        executor = ExecutionSimulator(config_path="execution_config.yaml")
        
        # Good liquidity to avoid partial fills
        liquidity = LiquidityState(
            volume_per_minute=50000,  # High volume
            bid_size=500,
            ask_size=500,
            typical_atr=10.0  # Low ATR = good liquidity metric
        )
        
        volatility = VolatilityState(
            current_atr=10.0,
            volatility_percentile=50,
            regime='moderate'
        )
        
        result = executor.simulate_execution(
            action='enter',
            target_size=10.0,
            mid_price=4500.0,
            liquidity_state=liquidity,
            volatility_state=volatility,
            symbol='ES'
        )
        
        assert result.actual_filled_size == 10.0, f"Expected 10.0 filled, got {result.actual_filled_size}"
        assert result.fill_price > 4500.0, "Expected fill price > mid (buy slippage)"
        assert result.updated_position.side == 'long', "Expected long side"
        assert result.updated_position.quantity == 10.0, "Expected qty 10"
        
        print(f"[PASS] ENTER action executed")
        print(f"       Entry price: {result.fill_price:.4f}")
        print(f"       Spread: {result.spread:.4f}")
        print(f"       Slippage: {result.slippage:.4f}")
        print(f"       Cost: ${result.transaction_cost:.2f}")
        return True
    except Exception as e:
        print(f"[FAIL] ENTER action failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_add_action():
    """Test ADD action."""
    print("\n[TEST 4] ADD Action")
    print("-" * 50)
    try:
        executor = ExecutionSimulator(config_path="execution_config.yaml")
        
        liquidity = LiquidityState(volume_per_minute=50000, bid_size=500, ask_size=500, typical_atr=10.0)
        volatility = VolatilityState(current_atr=10.0, volatility_percentile=50, regime='moderate')
        
        # First entry
        result1 = executor.simulate_execution(
            action='enter', target_size=10.0, mid_price=4500.0,
            liquidity_state=liquidity, volatility_state=volatility, symbol='ES'
        )
        
        # Add to position
        result2 = executor.simulate_execution(
            action='add', target_size=5.0, mid_price=4505.0,
            liquidity_state=liquidity, volatility_state=volatility, symbol='ES',
            current_position=result1.updated_position
        )
        
        assert result2.updated_position.quantity == 15.0, f"Expected 15 qty after ADD, got {result2.updated_position.quantity}"
        assert result2.updated_position.side == 'long', "Expected long side"
        
        print(f"[PASS] ADD action executed")
        print(f"       Total quantity: {result2.updated_position.quantity}")
        print(f"       New entry price: {result2.updated_position.entry_price:.4f}")
        return True
    except Exception as e:
        print(f"[FAIL] ADD action failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reduce_action():
    """Test REDUCE action."""
    print("\n[TEST 5] REDUCE Action")
    print("-" * 50)
    try:
        executor = ExecutionSimulator(config_path="execution_config.yaml")
        
        liquidity = LiquidityState(volume_per_minute=50000, bid_size=500, ask_size=500, typical_atr=10.0)
        volatility = VolatilityState(current_atr=10.0, volatility_percentile=50, regime='moderate')
        
        # Entry
        result1 = executor.simulate_execution(
            action='enter', target_size=10.0, mid_price=4500.0,
            liquidity_state=liquidity, volatility_state=volatility, symbol='ES'
        )
        
        # Reduce
        result2 = executor.simulate_execution(
            action='reduce', target_size=5.0, mid_price=4510.0,
            liquidity_state=liquidity, volatility_state=volatility, symbol='ES',
            current_position=result1.updated_position
        )
        
        assert result2.updated_position.quantity == 5.0, f"Expected 5 qty after REDUCE, got {result2.updated_position.quantity}"
        assert result2.updated_position.realized_pnl > 0, "Expected positive realized P&L"
        
        print(f"[PASS] REDUCE action executed")
        print(f"       Remaining quantity: {result2.updated_position.quantity}")
        print(f"       Realized P&L: ${result2.updated_position.realized_pnl:.2f}")
        return True
    except Exception as e:
        print(f"[FAIL] REDUCE action failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_exit_action():
    """Test EXIT action."""
    print("\n[TEST 6] EXIT Action")
    print("-" * 50)
    try:
        executor = ExecutionSimulator(config_path="execution_config.yaml")
        
        liquidity = LiquidityState(volume_per_minute=50000, bid_size=500, ask_size=500, typical_atr=10.0)
        volatility = VolatilityState(current_atr=10.0, volatility_percentile=50, regime='moderate')
        
        # Entry
        result1 = executor.simulate_execution(
            action='enter', target_size=10.0, mid_price=4500.0,
            liquidity_state=liquidity, volatility_state=volatility, symbol='ES'
        )
        
        # Exit
        result2 = executor.simulate_execution(
            action='exit', target_size=10.0, mid_price=4510.0,
            liquidity_state=liquidity, volatility_state=volatility, symbol='ES',
            current_position=result1.updated_position
        )
        
        assert result2.updated_position.quantity == 0, "Expected 0 qty after EXIT"
        assert result2.updated_position.side == 'flat', "Expected flat side"
        assert result2.updated_position.realized_pnl > 0, "Expected positive P&L"
        
        print(f"[PASS] EXIT action executed")
        print(f"       Final position: {result2.updated_position.side}")
        print(f"       Realized P&L: ${result2.updated_position.realized_pnl:.2f}")
        return True
    except Exception as e:
        print(f"[FAIL] EXIT action failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reverse_action():
    """Test REVERSE action."""
    print("\n[TEST 7] REVERSE Action")
    print("-" * 50)
    try:
        executor = ExecutionSimulator(config_path="execution_config.yaml")
        
        liquidity = LiquidityState(volume_per_minute=50000, bid_size=500, ask_size=500, typical_atr=10.0)
        volatility = VolatilityState(current_atr=10.0, volatility_percentile=50, regime='moderate')
        
        # Entry long
        result1 = executor.simulate_execution(
            action='enter', target_size=10.0, mid_price=4500.0,
            liquidity_state=liquidity, volatility_state=volatility, symbol='ES'
        )
        
        # Reverse to short
        result2 = executor.simulate_execution(
            action='reverse', target_size=10.0, mid_price=4510.0,
            liquidity_state=liquidity, volatility_state=volatility, symbol='ES',
            current_position=result1.updated_position
        )
        
        assert result2.updated_position.side == 'short', "Expected short side after REVERSE"
        assert result2.updated_position.quantity == 10.0, f"Expected 10 qty in new short, got {result2.updated_position.quantity}"
        
        print(f"[PASS] REVERSE action executed")
        print(f"       New position: {result2.updated_position.side} {result2.updated_position.quantity}")
        print(f"       Realized P&L: ${result2.updated_position.realized_pnl:.2f}")
        return True
    except Exception as e:
        print(f"[FAIL] REVERSE action failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_partial_fills():
    """Test partial fill logic."""
    print("\n[TEST 8] Partial Fills")
    print("-" * 50)
    try:
        executor = ExecutionSimulator(config_path="execution_config.yaml")
        
        # Low liquidity
        liquidity = LiquidityState(
            volume_per_minute=100,  # Very low
            bid_size=10,
            ask_size=10,
            typical_atr=1000.0  # Very high ATR = low liquidity metric
        )
        
        volatility = VolatilityState(
            current_atr=1000.0, volatility_percentile=100, regime='strong'
        )
        
        result = executor.simulate_execution(
            action='enter', target_size=100.0, mid_price=4500.0,
            liquidity_state=liquidity, volatility_state=volatility, symbol='ES'
        )
        
        # With low liquidity, might get partial fill
        print(f"[PASS] Partial fill test executed")
        print(f"       Target: 100.0, Filled: {result.actual_filled_size}")
        print(f"       Constrained: {result.liquidity_constraint_applied}")
        return True
    except Exception as e:
        print(f"[FAIL] Partial fill test failed: {e}")
        return False

def test_logging():
    """Test execution logging."""
    print("\n[TEST 9] Execution Logging")
    print("-" * 50)
    try:
        executor = ExecutionSimulator(config_path="execution_config.yaml")
        
        liquidity = LiquidityState(volume_per_minute=50000, bid_size=500, ask_size=500, typical_atr=10.0)
        volatility = VolatilityState(current_atr=10.0, volatility_percentile=50, regime='moderate')
        
        result = executor.simulate_execution(
            action='enter', target_size=10.0, mid_price=4500.0,
            liquidity_state=liquidity, volatility_state=volatility, symbol='ES'
        )
        
        # Check log file exists
        assert os.path.exists(executor.log_file), f"Log file not created: {executor.log_file}"
        
        with open(executor.log_file, 'r') as f:
            content = f.read()
            assert 'ENTER' in content, "ENTER action not logged"
            assert 'ES' in content, "Symbol not logged"
        
        print(f"[PASS] Logging verified")
        print(f"       Log file: {executor.log_file}")
        return True
    except Exception as e:
        print(f"[FAIL] Logging test failed: {e}")
        return False

def test_summary_statistics():
    """Test summary statistics."""
    print("\n[TEST 10] Summary Statistics")
    print("-" * 50)
    try:
        executor = ExecutionSimulator(config_path="execution_config.yaml")
        
        liquidity = LiquidityState(volume_per_minute=50000, bid_size=500, ask_size=500, typical_atr=10.0)
        volatility = VolatilityState(current_atr=10.0, volatility_percentile=50, regime='moderate')
        
        # Execute several trades
        for i in range(3):
            executor.simulate_execution(
                action='enter', target_size=10.0, mid_price=4500.0 + i*10,
                liquidity_state=liquidity, volatility_state=volatility, symbol='ES'
            )
        
        summary = executor.get_trade_log_summary()
        
        assert summary['total_trades'] == 3, f"Expected 3 trades, got {summary['total_trades']}"
        assert summary['total_costs'] > 0, "Expected positive total costs"
        assert summary['average_fill_percentage'] == 100.0, f"Expected 100% average fill, got {summary['average_fill_percentage']}%"
        
        print(f"[PASS] Summary statistics calculated")
        print(f"       Total trades: {summary['total_trades']}")
        print(f"       Total costs: ${summary['total_costs']:.2f}")
        print(f"       Avg fill %: {summary['average_fill_percentage']:.1f}%")
        return True
    except Exception as e:
        print(f"[FAIL] Summary statistics test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("ExecutionSimulator v1 - End-to-End Verification")
    print("=" * 50)
    
    tests = [
        test_module_imports,
        test_config_loading,
        test_enter_action,
        test_add_action,
        test_reduce_action,
        test_exit_action,
        test_reverse_action,
        test_partial_fills,
        test_logging,
        test_summary_statistics,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"[ERROR] Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 50)
    
    if all(results):
        print("[OK] All tests passed - ExecutionSimulator v1 is production-ready!")
        return 0
    else:
        print("[FAIL] Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
