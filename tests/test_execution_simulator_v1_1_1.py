"""
Tests for ExecutionSimulator v1.1.1 session/flow integration
"""

import pytest
from datetime import datetime
from engine.execution_simulator import (
    ExecutionSimulator, ExecutionResult, LiquidityState, 
    VolatilityState, PositionState, TradeAction
)


@pytest.fixture
def execution_simulator():
    """Create ExecutionSimulator instance for testing."""
    sim = ExecutionSimulator(config_path="execution_config.yaml")
    return sim


@pytest.fixture
def liquidity_state():
    """Create typical liquidity state."""
    return LiquidityState(
        volume_per_minute=500.0,
        bid_size=100.0,
        ask_size=100.0,
        typical_atr=0.5
    )


@pytest.fixture
def volatility_state():
    """Create typical volatility state."""
    return VolatilityState(
        current_atr=0.75,
        volatility_percentile=50.0,
        regime="moderate"
    )


@pytest.fixture
def flat_position():
    """Create flat position state."""
    return PositionState(
        symbol="ES",
        side="flat",
        quantity=0,
        entry_price=0,
        current_price=4500.0,
        entry_cost=0,
        unrealized_pnl=0,
        realized_pnl=0
    )


def create_policy_decision(session_name, flow_signals=None, session_modifiers=None):
    """Create a mock policy decision dict."""
    return {
        'session_name': session_name,
        'session_modifiers': session_modifiers or {},
        'flow_signals': flow_signals or {},
    }


# ============================================================================
# Test 1: Session-aware slippage (GLOBEX)
# ============================================================================
def test_globex_increases_slippage_and_reduces_fill_probability(execution_simulator, liquidity_state, volatility_state, flat_position):
    """GLOBEX should have high slippage and low fill probability."""
    decision = create_policy_decision("GLOBEX", {}, {})
    
    result = execution_simulator.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=4500.0,
        liquidity_state=liquidity_state,
        volatility_state=volatility_state,
        symbol="ES",
        current_position=flat_position,
        policy_decision=decision
    )
    
    assert result.session_name == "GLOBEX"
    assert result.fill_probability < 0.85  # GLOBEX should have fill_prob ~0.70
    assert result.slippage > 0  # Slippage should be increased
    assert 'session_factor' in result.slippage_components


# ============================================================================
# Test 2: Session-aware slippage (RTH_OPEN)
# ============================================================================
def test_rth_open_highest_slippage(execution_simulator, liquidity_state, volatility_state, flat_position):
    """RTH_OPEN should have the highest slippage of the day."""
    decision = create_policy_decision("RTH_OPEN", {}, {})
    
    result = execution_simulator.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=4500.0,
        liquidity_state=liquidity_state,
        volatility_state=volatility_state,
        symbol="ES",
        current_position=flat_position,
        policy_decision=decision
    )
    
    assert result.session_name == "RTH_OPEN"
    assert result.fill_probability < 0.75  # RTH_OPEN should have fill_prob ~0.65
    assert 'session_factor' in result.slippage_components
    assert result.slippage_components['session_factor'] == 2.0


# ============================================================================
# Test 3: Session-aware slippage (MIDDAY)
# ============================================================================
def test_midday_tight_spreads_and_low_slippage(execution_simulator, liquidity_state, volatility_state, flat_position):
    """MIDDAY should have tight spreads, low slippage, high fills."""
    decision = create_policy_decision("MIDDAY", {}, {})
    
    result = execution_simulator.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=4500.0,
        liquidity_state=liquidity_state,
        volatility_state=volatility_state,
        symbol="ES",
        current_position=flat_position,
        policy_decision=decision
    )
    
    assert result.session_name == "MIDDAY"
    assert result.fill_probability > 0.90  # MIDDAY should have fill_prob ~0.95
    assert result.slippage_components['session_factor'] == 0.6  # 60% of base slippage


# ============================================================================
# Test 4: Session-aware slippage (POWER_HOUR)
# ============================================================================
def test_power_hour_moderate_slippage(execution_simulator, liquidity_state, volatility_state, flat_position):
    """POWER_HOUR should have moderate slippage with strong fills."""
    decision = create_policy_decision("POWER_HOUR", {}, {})
    
    result = execution_simulator.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=4500.0,
        liquidity_state=liquidity_state,
        volatility_state=volatility_state,
        symbol="ES",
        current_position=flat_position,
        policy_decision=decision
    )
    
    assert result.session_name == "POWER_HOUR"
    assert result.fill_probability > 0.85  # POWER_HOUR ~0.88
    assert result.slippage_components['session_factor'] == 1.2


# ============================================================================
# Test 5: Flow-aware: Stop-run detection increases slippage
# ============================================================================
def test_stop_run_increases_slippage_reduces_fill_probability(execution_simulator, liquidity_state, volatility_state, flat_position):
    """Stop-run detected should increase slippage and reduce fill probability."""
    flow_signals = {
        'stop_run_detected': True,
    }
    decision = create_policy_decision("MIDDAY", flow_signals, {})
    
    result = execution_simulator.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=4500.0,
        liquidity_state=liquidity_state,
        volatility_state=volatility_state,
        symbol="ES",
        current_position=flat_position,
        policy_decision=decision
    )
    
    assert result.session_name == "MIDDAY"
    assert 'stop_run' in result.slippage_components
    assert result.slippage_components['stop_run']['slippage_mult'] == 1.5
    assert result.slippage_components['stop_run']['fill_prob_mult'] == 0.80


# ============================================================================
# Test 6: Flow-aware: Initiative move in POWER_HOUR allows faster fills
# ============================================================================
def test_initiative_in_power_hour_improves_fills(execution_simulator, liquidity_state, volatility_state, flat_position):
    """Initiative move detected in POWER_HOUR should allow faster fills."""
    flow_signals = {
        'initiative_move_detected': True,
    }
    decision = create_policy_decision("POWER_HOUR", flow_signals, {})
    
    result = execution_simulator.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=4500.0,
        liquidity_state=liquidity_state,
        volatility_state=volatility_state,
        symbol="ES",
        current_position=flat_position,
        policy_decision=decision
    )
    
    assert result.session_name == "POWER_HOUR"
    assert 'initiative' in result.slippage_components
    assert result.slippage_components['initiative']['slippage_mult'] == 0.85
    assert result.slippage_components['initiative']['fill_prob_mult'] == 1.1


# ============================================================================
# Test 7: Flow-aware: Initiative move in non-POWER_HOUR increases slippage
# ============================================================================
def test_initiative_in_midday_worsens_fills(execution_simulator, liquidity_state, volatility_state, flat_position):
    """Initiative move in MIDDAY (not POWER_HOUR) should worsen fills."""
    flow_signals = {
        'initiative_move_detected': True,
    }
    decision = create_policy_decision("MIDDAY", flow_signals, {})
    
    result = execution_simulator.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=4500.0,
        liquidity_state=liquidity_state,
        volatility_state=volatility_state,
        symbol="ES",
        current_position=flat_position,
        policy_decision=decision
    )
    
    assert result.session_name == "MIDDAY"
    assert 'initiative' in result.slippage_components
    assert result.slippage_components['initiative']['slippage_mult'] == 1.1
    assert result.slippage_components['initiative']['fill_prob_mult'] == 0.90


# ============================================================================
# Test 8: Flow-aware: Strong level reaction improves fills
# ============================================================================
def test_strong_level_reaction_improves_fills(execution_simulator, liquidity_state, volatility_state, flat_position):
    """Strong positive level reaction should improve fill probability."""
    flow_signals = {
        'level_reaction_score': 0.7,
    }
    decision = create_policy_decision("MIDDAY", flow_signals, {})
    
    result = execution_simulator.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=4500.0,
        liquidity_state=liquidity_state,
        volatility_state=volatility_state,
        symbol="ES",
        current_position=flat_position,
        policy_decision=decision
    )
    
    assert 'level_reaction' in result.slippage_components
    assert result.slippage_components['level_reaction']['fill_prob_mult'] == 1.05


# ============================================================================
# Test 9: Flow-aware: Extreme VWAP distance increases slippage
# ============================================================================
def test_extreme_vwap_distance_increases_slippage(execution_simulator, liquidity_state, volatility_state, flat_position):
    """Extreme VWAP distance should increase slippage."""
    flow_signals = {
        'vwap_distance': 0.03,  # 3% from VWAP
    }
    decision = create_policy_decision("MIDDAY", flow_signals, {})
    
    result = execution_simulator.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=4500.0,
        liquidity_state=liquidity_state,
        volatility_state=volatility_state,
        symbol="ES",
        current_position=flat_position,
        policy_decision=decision
    )
    
    assert 'vwap_distance' in result.slippage_components
    assert result.slippage_components['vwap_distance']['slippage_mult'] > 1.0


# ============================================================================
# Test 10: Flow-aware: Round-level proximity widens spreads
# ============================================================================
def test_round_level_proximity_widens_spreads(execution_simulator, liquidity_state, volatility_state, flat_position):
    """High round-level proximity should slightly widen spreads."""
    flow_signals = {
        'round_level_proximity': 0.95,
    }
    decision = create_policy_decision("MIDDAY", flow_signals, {})
    
    result = execution_simulator.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=4500.0,
        liquidity_state=liquidity_state,
        volatility_state=volatility_state,
        symbol="ES",
        current_position=flat_position,
        policy_decision=decision
    )
    
    assert 'round_level' in result.slippage_components
    assert result.slippage_components['round_level']['slippage_mult'] == 1.1


# ============================================================================
# Test 11: Session modifiers affect slippage
# ============================================================================
def test_session_modifiers_affect_slippage(execution_simulator, liquidity_state, volatility_state, flat_position):
    """Session modifiers (volatility/liquidity scale) should affect slippage."""
    session_modifiers = {
        'volatility_scale': 1.5,
        'liquidity_scale': 0.8,
    }
    decision = create_policy_decision("MIDDAY", {}, session_modifiers)
    
    result = execution_simulator.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=4500.0,
        liquidity_state=liquidity_state,
        volatility_state=volatility_state,
        symbol="ES",
        current_position=flat_position,
        policy_decision=decision
    )
    
    assert result.session_modifiers == session_modifiers
    assert 'modifier_factor' in result.slippage_components


# ============================================================================
# Test 12: ExecutionResult is JSON serializable
# ============================================================================
def test_execution_result_json_serializable(execution_simulator, liquidity_state, volatility_state, flat_position):
    """ExecutionResult with session/flow fields should be JSON serializable."""
    import json
    
    flow_signals = {
        'stop_run_detected': True,
        'vwap_distance': 0.01,
    }
    decision = create_policy_decision("GLOBEX", flow_signals, {'liquidity_scale': 0.5})
    
    result = execution_simulator.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=4500.0,
        liquidity_state=liquidity_state,
        volatility_state=volatility_state,
        symbol="ES",
        current_position=flat_position,
        policy_decision=decision
    )
    
    # Try to serialize result's components
    result_dict = {
        'action': result.action,
        'target_size': result.target_size,
        'actual_filled_size': result.actual_filled_size,
        'fill_price': result.fill_price,
        'session_name': result.session_name,
        'session_modifiers': result.session_modifiers,
        'flow_signals': result.flow_signals,
        'slippage_components': result.slippage_components,
        'fill_probability': result.fill_probability,
        'partial_fill_ratio': result.partial_fill_ratio,
    }
    
    json_str = json.dumps(result_dict, default=str)
    assert json_str is not None
    restored = json.loads(json_str)
    assert restored['session_name'] == "GLOBEX"


# ============================================================================
# Test 13: Multiple flows combined (stop-run + VWAP + round-level)
# ============================================================================
def test_multiple_flow_signals_combined(execution_simulator, liquidity_state, volatility_state, flat_position):
    """Multiple flow signals should stack adjustments."""
    flow_signals = {
        'stop_run_detected': True,
        'vwap_distance': 0.025,
        'round_level_proximity': 0.9,
    }
    decision = create_policy_decision("RTH_OPEN", flow_signals, {})
    
    result = execution_simulator.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=4500.0,
        liquidity_state=liquidity_state,
        volatility_state=volatility_state,
        symbol="ES",
        current_position=flat_position,
        policy_decision=decision
    )
    
    assert result.session_name == "RTH_OPEN"
    assert 'stop_run' in result.slippage_components
    assert 'vwap_distance' in result.slippage_components
    assert 'round_level' in result.slippage_components


# ============================================================================
# Test 14: Trade log includes session/flow context
# ============================================================================
def test_trade_log_includes_session_flow_context(execution_simulator, liquidity_state, volatility_state, flat_position):
    """Trade log should include session/flow context for all trades."""
    flow_signals = {'stop_run_detected': True}
    decision = create_policy_decision("MIDDAY", flow_signals, {'liquidity_scale': 1.2})
    
    result = execution_simulator.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=4500.0,
        liquidity_state=liquidity_state,
        volatility_state=volatility_state,
        symbol="ES",
        current_position=flat_position,
        policy_decision=decision
    )
    
    assert len(execution_simulator.trade_log) == 1
    logged_result = execution_simulator.trade_log[0]
    assert logged_result.session_name == "MIDDAY"
    assert logged_result.flow_signals == flow_signals
    assert logged_result.session_modifiers == {'liquidity_scale': 1.2}


# ============================================================================
# Test 15: CLOSE session allows flow trades
# ============================================================================
def test_close_session_allows_flow_trades(execution_simulator, liquidity_state, volatility_state, flat_position):
    """CLOSE session should have moderate slippage, good fill probability."""
    flow_signals = {}
    decision = create_policy_decision("CLOSE", flow_signals, {})
    
    result = execution_simulator.execute_order(
        action=TradeAction.ENTER.value,
        target_size=10,
        mid_price=4500.0,
        liquidity_state=liquidity_state,
        volatility_state=volatility_state,
        symbol="ES",
        current_position=flat_position,
        policy_decision=decision
    )
    
    assert result.session_name == "CLOSE"
    assert result.fill_probability > 0.80
    assert result.slippage_components['session_factor'] == 1.4
