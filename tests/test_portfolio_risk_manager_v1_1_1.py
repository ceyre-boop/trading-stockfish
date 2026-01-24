"""
Tests for PortfolioRiskManager v1.1.1 session/flow/capacity integration
"""

import pytest
from datetime import datetime
from engine.portfolio_risk_manager import PortfolioRiskManager, RiskDecision


@pytest.fixture
def portfolio_manager():
    """Create PortfolioRiskManager instance for testing."""
    mgr = PortfolioRiskManager(
        total_capital=1000000,
        max_symbol_exposure=500000,
        max_total_exposure=1000000,
        max_daily_loss=50000
    )
    return mgr


@pytest.fixture
def volume_state():
    """Create typical volume state."""
    return {
        'volume_1min': 5000.0,
        'volume_5min': 25000.0,
    }


def create_policy_decision(session_name, flow_signals=None, session_modifiers=None):
    """Create a mock policy decision dict."""
    return {
        'session_name': session_name,
        'session_modifiers': session_modifiers or {},
        'flow_signals': flow_signals or {},
    }


# ============================================================================
# Test 1: Session-aware risk scaling (GLOBEX)
# ============================================================================
def test_globex_reduces_position_size(portfolio_manager, volume_state):
    """GLOBEX should reduce position size (0.5x)."""
    decision = create_policy_decision("GLOBEX", {}, {})
    
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=100,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert result.session_name == "GLOBEX"
    assert result.action == "ALLOW"
    # GLOBEX multiplier is 0.5
    assert result.approved_size < 1.0
    assert 'session_factor' in result.risk_scaling_factors


# ============================================================================
# Test 2: Session-aware risk scaling (RTH_OPEN)
# ============================================================================
def test_rth_open_strictest_limits(portfolio_manager, volume_state):
    """RTH_OPEN should have strictest limits (0.4x multiplier)."""
    decision = create_policy_decision("RTH_OPEN", {}, {})
    
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=100,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert result.session_name == "RTH_OPEN"
    assert result.action == "ALLOW"
    assert result.risk_scaling_factors['session_factor'] == 0.4


# ============================================================================
# Test 3: Session-aware risk scaling (MIDDAY)
# ============================================================================
def test_midday_normal_scaling(portfolio_manager, volume_state):
    """MIDDAY should have normal 1.0x scaling."""
    decision = create_policy_decision("MIDDAY", {}, {})
    
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=100,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert result.session_name == "MIDDAY"
    assert result.action == "ALLOW"
    assert result.risk_scaling_factors['session_factor'] == 1.0


# ============================================================================
# Test 4: Session-aware risk scaling (POWER_HOUR)
# ============================================================================
def test_power_hour_allows_scaling(portfolio_manager, volume_state):
    """POWER_HOUR should allow scaling (1.1x multiplier)."""
    decision = create_policy_decision("POWER_HOUR", {}, {})
    
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=100,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert result.session_name == "POWER_HOUR"
    assert result.action == "ALLOW"
    assert result.risk_scaling_factors['session_factor'] == 1.1


# ============================================================================
# Test 5: Flow-aware: Stop-run detection reduces size
# ============================================================================
def test_stop_run_reduces_size(portfolio_manager, volume_state):
    """Stop-run detected should reduce size (0.6x)."""
    flow_signals = {
        'stop_run_detected': True,
    }
    decision = create_policy_decision("MIDDAY", flow_signals, {})
    
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=100,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert result.action == "ALLOW"
    assert 'stop_run' in result.risk_scaling_factors
    assert result.risk_scaling_factors['stop_run'] == 0.6


# ============================================================================
# Test 6: Flow-aware: Initiative in POWER_HOUR allows scaling
# ============================================================================
def test_initiative_in_power_hour_scales_up(portfolio_manager, volume_state):
    """Initiative in POWER_HOUR should allow scaling (1.15x)."""
    flow_signals = {
        'initiative_move_detected': True,
    }
    decision = create_policy_decision("POWER_HOUR", flow_signals, {})
    
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=100,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert result.action == "ALLOW"
    assert 'initiative' in result.risk_scaling_factors
    assert result.risk_scaling_factors['initiative'] == 1.15


# ============================================================================
# Test 7: Flow-aware: Initiative outside POWER_HOUR reduces size
# ============================================================================
def test_initiative_outside_power_hour_reduces(portfolio_manager, volume_state):
    """Initiative outside POWER_HOUR should reduce size (0.75x)."""
    flow_signals = {
        'initiative_move_detected': True,
    }
    decision = create_policy_decision("MIDDAY", flow_signals, {})
    
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=100,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert result.action == "ALLOW"
    assert 'initiative' in result.risk_scaling_factors
    assert result.risk_scaling_factors['initiative'] == 0.75


# ============================================================================
# Test 8: Flow-aware: Strong level reaction adjusts size
# ============================================================================
def test_strong_positive_level_reaction_scales_up(portfolio_manager, volume_state):
    """Strong positive level reaction should scale up (1.1x)."""
    flow_signals = {
        'level_reaction_score': 0.75,
    }
    decision = create_policy_decision("MIDDAY", flow_signals, {})
    
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=100,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert 'level_reaction' in result.risk_scaling_factors
    assert result.risk_scaling_factors['level_reaction'] == 1.1


# ============================================================================
# Test 9: Flow-aware: Strong negative level reaction reduces size
# ============================================================================
def test_strong_negative_level_reaction_reduces(portfolio_manager, volume_state):
    """Strong negative level reaction should reduce size (0.7x)."""
    flow_signals = {
        'level_reaction_score': -0.75,
    }
    decision = create_policy_decision("MIDDAY", flow_signals, {})
    
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=100,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert 'level_reaction' in result.risk_scaling_factors
    assert result.risk_scaling_factors['level_reaction'] == 0.7


# ============================================================================
# Test 10: Flow-aware: Extreme VWAP distance reduces size
# ============================================================================
def test_extreme_vwap_distance_reduces_size(portfolio_manager, volume_state):
    """Extreme VWAP distance (>3%) should reduce size (0.5x)."""
    flow_signals = {
        'vwap_distance': 0.035,  # 3.5% from VWAP
    }
    decision = create_policy_decision("MIDDAY", flow_signals, {})
    
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=100,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert 'vwap_distance' in result.risk_scaling_factors
    assert result.risk_scaling_factors['vwap_distance'] == 0.5


# ============================================================================
# Test 11: Capacity enforcement: notional limit
# ============================================================================
def test_notional_limit_blocks_oversized_order(portfolio_manager, volume_state):
    """Order exceeding notional limit should be BLOCKED."""
    decision = create_policy_decision("MIDDAY", {}, {})
    
    # Try to trade 2000 ES contracts at 4500 = 9,000,000 notional (exceeds 5M ES limit)
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=2000,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert result.action == "BLOCK"
    assert result.capacity_flags.get('notional_exceeded', False)


# ============================================================================
# Test 12: Capacity enforcement: volume limit
# ============================================================================
def test_volume_limit_blocks_oversized_order(portfolio_manager, volume_state):
    """Order exceeding volume limit should be BLOCKED."""
    decision = create_policy_decision("MIDDAY", {}, {})
    
    # Try to trade 600 contracts (600 > 5% of 1000 volume)
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=600,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert result.action == "BLOCK"
    assert result.capacity_flags.get('volume_1min_exceeded', False)


# ============================================================================
# Test 13: Session modifiers affect position sizing
# ============================================================================
def test_session_modifiers_affect_size(portfolio_manager, volume_state):
    """Session modifiers (risk_scale) should affect approved size."""
    session_modifiers = {
        'liquidity_scale': 1.2,
        'risk_scale': 0.5,
    }
    decision = create_policy_decision("MIDDAY", {}, session_modifiers)
    
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=100,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert result.session_modifiers == session_modifiers
    assert 'modifier_factor' in result.risk_scaling_factors


# ============================================================================
# Test 14: Daily loss limit triggers FORCE_EXIT
# ============================================================================
def test_daily_loss_limit_triggers_force_exit(portfolio_manager, volume_state):
    """Daily loss exceeding limit should trigger FORCE_EXIT."""
    portfolio_manager.daily_pnl = -60000  # Exceeds 50k limit
    
    decision = create_policy_decision("MIDDAY", {}, {})
    
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=100,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert result.action == "FORCE_EXIT"
    assert result.approved_size == 0.0


# ============================================================================
# Test 15: RiskDecision is JSON serializable
# ============================================================================
def test_risk_decision_json_serializable(portfolio_manager, volume_state):
    """RiskDecision with all context fields should be JSON serializable."""
    import json
    
    flow_signals = {
        'stop_run_detected': True,
        'vwap_distance': 0.02,
    }
    decision = create_policy_decision("GLOBEX", flow_signals, {'liquidity_scale': 0.8})
    
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=50,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    result_dict = result.to_dict()
    json_str = json.dumps(result_dict, default=str)
    assert json_str is not None
    restored = json.loads(json_str)
    assert restored['session_name'] == "GLOBEX"
    assert restored['action'] in ["ALLOW", "BLOCK", "REDUCE_SIZE", "FORCE_EXIT"]


# ============================================================================
# Test 16: Multiple flows combined (stop-run + initiative + level-reaction)
# ============================================================================
def test_multiple_flows_combined(portfolio_manager, volume_state):
    """Multiple flow signals should combine adjustments."""
    flow_signals = {
        'stop_run_detected': True,
        'initiative_move_detected': True,
        'level_reaction_score': 0.7,
    }
    decision = create_policy_decision("POWER_HOUR", flow_signals, {})
    
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=100,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert result.action == "ALLOW"
    assert 'stop_run' in result.risk_scaling_factors
    assert 'initiative' in result.risk_scaling_factors
    assert 'level_reaction' in result.risk_scaling_factors


# ============================================================================
# Test 17: Close session tightens limits
# ============================================================================
def test_close_session_tightens_limits(portfolio_manager, volume_state):
    """CLOSE session should tighten position limits (0.6x)."""
    decision = create_policy_decision("CLOSE", {}, {})
    
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=100,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert result.session_name == "CLOSE"
    assert result.action == "ALLOW"
    assert result.risk_scaling_factors['session_factor'] == 0.6


# ============================================================================
# Test 18: PREMARKET reduces size before macro events
# ============================================================================
def test_premarket_reduces_size(portfolio_manager, volume_state):
    """PREMARKET session should reduce size (0.7x)."""
    decision = create_policy_decision("PREMARKET", {}, {})
    
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=100,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert result.session_name == "PREMARKET"
    assert result.action == "ALLOW"
    assert result.risk_scaling_factors['session_factor'] == 0.7


# ============================================================================
# Test 19: Exposure limit triggers REDUCE_SIZE
# ============================================================================
def test_symbol_exposure_limit_reduces_size(portfolio_manager, volume_state):
    """Order exceeding symbol exposure limit should trigger REDUCE_SIZE."""
    # Set portfolio to near symbol limit already
    portfolio_manager.current_exposure_per_symbol["ES"] = 450000  
    portfolio_manager.current_total_exposure = 450000
    
    decision = create_policy_decision("MIDDAY", {}, {})
    
    # Try to add 50 more (50 * 4500 = 225k, would be 675k total for ES, exceeds 500k limit)
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=50,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    assert result.action == "REDUCE_SIZE"
    assert result.approved_size < 1.0


# ============================================================================
# Test 20: Portfolio exposure limit triggers REDUCE_SIZE
# ============================================================================
def test_portfolio_exposure_limit_reduces_size(portfolio_manager, volume_state):
    """Order exceeding total portfolio limit should trigger REDUCE_SIZE."""
    # Set portfolio to near total limit
    portfolio_manager.current_exposure_per_symbol["ES"] = 600000
    portfolio_manager.current_exposure_per_symbol["NQ"] = 350000
    portfolio_manager.current_total_exposure = 950000  # Near 1M limit
    
    decision = create_policy_decision("MIDDAY", {}, {})
    
    # Try to add 20 more ES (20 * 4500 = 90k, would be 1,040k total, exceeds 1M limit)
    result = portfolio_manager.evaluate_risk_with_context(
        symbol="ES",
        target_size=20,
        price=4500.0,
        policy_decision=decision,
        volume_state=volume_state
    )
    
    # Should reduce size to fit portfolio limit
    assert result.action == "REDUCE_SIZE"
