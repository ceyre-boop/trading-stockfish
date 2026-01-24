"""Test ExecutionSimulator v1 functionality."""

from engine.execution_simulator import (
    ExecutionSimulator, LiquidityState, VolatilityState, PositionState
)

# Initialize simulator
try:
    executor = ExecutionSimulator(config_path="execution_config.yaml")
    print("[OK] ExecutionSimulator initialized")
    print(f"[OK] Config loaded for symbols: {list(executor.symbol_configs.keys())}")
except Exception as e:
    print(f"[ERROR] Failed to initialize: {e}")
    exit(1)

# Test a simple execution
liquidity = LiquidityState(
    volume_per_minute=1000,
    bid_size=500,
    ask_size=500,
    typical_atr=1.0
)

volatility = VolatilityState(
    current_atr=1.0,
    volatility_percentile=50,
    regime='moderate'
)

try:
    result = executor.simulate_execution(
        action='enter',
        target_size=10.0,
        mid_price=4500.0,
        liquidity_state=liquidity,
        volatility_state=volatility,
        symbol='ES'
    )
    
    print(f"\n[OK] Execution simulated for ES")
    print(f"    Target size: {result.target_size}")
    print(f"    Actual filled: {result.actual_filled_size}")
    print(f"    Fill price: {result.fill_price:.2f}")
    print(f"    Spread: {result.spread:.4f}")
    print(f"    Slippage: {result.slippage:.4f}")
    print(f"    Transaction cost: ${result.transaction_cost:.2f}")
    print(f"    Total cost (slippage + commission): ${result.total_cost:.2f}")
    print(f"    Filled %: {result.filled_percentage*100:.1f}%")
    print(f"    Position side: {result.updated_position.side}")
    print(f"    Position quantity: {result.updated_position.quantity}")
    print(f"    Position entry price: {result.updated_position.entry_price:.2f}")

except Exception as e:
    print(f"[ERROR] Execution simulation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test reduction
try:
    result2 = executor.simulate_execution(
        action='reduce',
        target_size=5.0,
        mid_price=4510.0,
        liquidity_state=liquidity,
        volatility_state=volatility,
        symbol='ES',
        current_position=result.updated_position
    )
    
    print(f"\n[OK] Position reduction executed")
    print(f"    Action: {result2.action}")
    print(f"    Target size: {result2.target_size}")
    print(f"    Actual filled: {result2.actual_filled_size}")
    print(f"    Fill price: {result2.fill_price:.2f}")
    print(f"    Realized P&L: ${result2.updated_position.realized_pnl:.2f}")
    print(f"    Remaining position: {result2.updated_position.quantity}")

except Exception as e:
    print(f"[ERROR] Reduction simulation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Get log summary
summary = executor.get_trade_log_summary()
print(f"\n[OK] Trade log summary:")
for key, value in summary.items():
    print(f"    {key}: {value}")

print("\n[OK] ExecutionSimulator v1 is fully functional")
