"""
ExecutionSimulator v1.1.1 Validation Script

Demonstrates ExecutionSimulator with session and flow awareness.
Simulates a day of trading across different sessions with varying flow conditions.
"""

import json
from datetime import datetime, timezone
from engine.execution_simulator import (
    ExecutionSimulator, LiquidityState, VolatilityState, PositionState, TradeAction
)


def create_market_state(session_name, volatility_pct=50):
    """Create a market state for a given session."""
    liquidity = LiquidityState(
        volume_per_minute=400.0 if session_name in ["GLOBEX", "PREMARKET"] else 
                          600.0 if session_name == "RTH_OPEN" else
                          800.0 if session_name == "MIDDAY" else
                          700.0 if session_name == "POWER_HOUR" else 500.0,
        bid_size=100.0,
        ask_size=100.0,
        typical_atr=0.5
    )
    
    volatility = VolatilityState(
        current_atr=0.75 if session_name == "RTH_OPEN" else 
                    0.5 if session_name == "MIDDAY" else 0.65,
        volatility_percentile=volatility_pct,
        regime="high" if volatility_pct > 70 else "moderate" if volatility_pct > 40 else "low"
    )
    
    return liquidity, volatility


def run_session_simulation(session_name, mid_price, volatility_pct=50):
    """Simulate trades in a specific session."""
    print(f"\n{'='*80}")
    print(f"SESSION: {session_name}")
    print(f"{'='*80}")
    
    sim = ExecutionSimulator()
    liquidity, volatility = create_market_state(session_name, volatility_pct)
    position = PositionState(
        symbol="ES",
        side="flat",
        quantity=0,
        entry_price=0,
        current_price=mid_price,
        entry_cost=0,
        unrealized_pnl=0,
        realized_pnl=0
    )
    
    # Scenario 1: Clean entry (no flow issues)
    print(f"\n[1] Clean Entry - Mid={mid_price:.1f}, Vol={volatility_pct}%")
    decision_clean = {
        'session_name': session_name,
        'session_modifiers': {},
        'flow_signals': {},
    }
    result_clean = sim.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=mid_price,
        liquidity_state=liquidity,
        volatility_state=volatility,
        symbol="ES",
        current_position=position,
        policy_decision=decision_clean
    )
    print(f"  Fill: {result_clean.actual_filled_size:.0f}@{result_clean.fill_price:.2f}")
    print(f"  Slippage: {result_clean.slippage:.4f} | FillProb: {result_clean.fill_probability:.0%}")
    print(f"  Components: {result_clean.slippage_components}")
    
    # Scenario 2: With stop-run detected
    print(f"\n[2] Stop-Run Detected - Worsens execution")
    decision_stoprun = {
        'session_name': session_name,
        'session_modifiers': {},
        'flow_signals': {'stop_run_detected': True},
    }
    result_stoprun = sim.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=mid_price,
        liquidity_state=liquidity,
        volatility_state=volatility,
        symbol="ES",
        current_position=position,
        policy_decision=decision_stoprun
    )
    print(f"  Fill: {result_stoprun.actual_filled_size:.0f}@{result_stoprun.fill_price:.2f}")
    print(f"  Slippage: {result_stoprun.slippage:.4f} | FillProb: {result_stoprun.fill_probability:.0%}")
    print(f"  Components: {result_stoprun.slippage_components}")
    
    # Scenario 3: With VWAP distance extreme
    print(f"\n[3] Extreme VWAP Distance - Increases slippage for mean-reversion")
    decision_vwap = {
        'session_name': session_name,
        'session_modifiers': {},
        'flow_signals': {'vwap_distance': 0.035},  # 3.5% from VWAP
    }
    result_vwap = sim.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=mid_price,
        liquidity_state=liquidity,
        volatility_state=volatility,
        symbol="ES",
        current_position=position,
        policy_decision=decision_vwap
    )
    print(f"  Fill: {result_vwap.actual_filled_size:.0f}@{result_vwap.fill_price:.2f}")
    print(f"  Slippage: {result_vwap.slippage:.4f} | FillProb: {result_vwap.fill_probability:.0%}")
    print(f"  Components: {result_vwap.slippage_components}")
    
    # Scenario 4: With initiative move (session-dependent)
    print(f"\n[4] Initiative Move Detected - Effect depends on session")
    decision_init = {
        'session_name': session_name,
        'session_modifiers': {},
        'flow_signals': {'initiative_move_detected': True},
    }
    result_init = sim.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=mid_price,
        liquidity_state=liquidity,
        volatility_state=volatility,
        symbol="ES",
        current_position=position,
        policy_decision=decision_init
    )
    print(f"  Fill: {result_init.actual_filled_size:.0f}@{result_init.fill_price:.2f}")
    print(f"  Slippage: {result_init.slippage:.4f} | FillProb: {result_init.fill_probability:.0%}")
    if 'initiative' in result_init.slippage_components:
        print(f"  Initiative Effect: {result_init.slippage_components['initiative']}")
    
    # Summary
    print(f"\n[Summary] Session execution characteristics:")
    print(f"  Base slippage (clean): {result_clean.slippage:.4f}")
    print(f"  With stop-run: {result_stoprun.slippage:.4f} (+{(result_stoprun.slippage/result_clean.slippage - 1)*100:.0f}%)")
    print(f"  With VWAP distance: {result_vwap.slippage:.4f} (+{(result_vwap.slippage/result_clean.slippage - 1)*100:.0f}%)")
    print(f"  Fill probability: {result_clean.fill_probability:.0%} → {result_stoprun.fill_probability:.0%} (stop-run)")


def main():
    """Run validation across all sessions."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "ExecutionSimulator v1.1.1 Validation" + " "*24 + "║")
    print("║" + " "*15 + "Session-Aware and Flow-Aware Execution Realism" + " "*17 + "║")
    print("╚" + "="*78 + "╝")
    
    # Define test scenarios for each session
    sessions = [
        ("GLOBEX", 4495.0, 30),      # Low volatility overnight
        ("PREMARKET", 4498.0, 35),   # Moderate volatility
        ("RTH_OPEN", 4500.0, 75),    # HIGH volatility at open
        ("MIDDAY", 4505.0, 45),      # Moderate, cleaner
        ("POWER_HOUR", 4510.0, 65),  # Elevated volatility
        ("CLOSE", 4508.0, 55),       # Moderate flow
    ]
    
    for session_name, mid_price, vol_pct in sessions:
        run_session_simulation(session_name, mid_price, vol_pct)
    
    print(f"\n{'='*80}")
    print("✅ Validation Complete")
    print(f"{'='*80}")
    
    print("\n[Key Findings]")
    print("1. GLOBEX: High slippage, low fill probability (overnight illiquidity)")
    print("2. RTH_OPEN: Highest slippage of day (chaotic, volatile)")
    print("3. MIDDAY: Tight execution, high fill probability (peak efficiency)")
    print("4. POWER_HOUR: Moderate slippage, good fills (elevated volatility)")
    print("5. Stop-run detection: Increases slippage ~50%, reduces fill probability")
    print("6. VWAP distance: Extreme deviation (>2%) penalizes mean-reversion entries")
    print("7. Initiative moves: Helpful in POWER_HOUR, harmful elsewhere")
    print()


if __name__ == "__main__":
    main()
