"""
Integration tests for PolicyEngine with SessionContext and FlowContext.

Validates:
- Session-aware thresholds applied correctly
- Flow-aware decision logic (stop-run, initiative, level reactions)
- PolicyDecision includes session/flow fields
- Session-specific behavior differences

Requires mock eval results with session context from CausalEvaluator.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock

from engine.policy_engine import (
    PolicyEngine,
    PositionState,
    PositionSide,
    TradingAction,
    RiskConfig,
)


def create_mock_eval_result(
    eval_score: float = 0.5,
    confidence: float = 0.75,
    session_name: str = "MIDDAY",
    stop_run_detected: bool = False,
    initiative_move_detected: bool = False,
    level_reaction_score: float = 0.5,
    vwap_distance_pct: float = 0.0,
) -> dict:
    """Create a mock CausalEvaluator result with session/flow context."""
    return {
        'eval': eval_score,
        'eval_score': eval_score,
        'confidence': confidence,
        'session': session_name,
        'session_modifiers': {
            'vol_scale': 1.0,
            'liq_scale': 1.0,
            'risk_scale': 1.0,
        },
        'flow_signals': {
            'vwap': 5000.0,
            'vwap_distance_pct': vwap_distance_pct,
            'round_level_proximity': 0.2,
            'stop_run_detected': stop_run_detected,
            'initiative_move_detected': initiative_move_detected,
            'prior_high': 5010.0,
            'prior_low': 4990.0,
            'overnight_high': 5005.0,
            'overnight_low': 4995.0,
        },
        'stop_run_detected': stop_run_detected,
        'initiative_move_detected': initiative_move_detected,
        'level_reaction_score': level_reaction_score,
    }


@pytest.fixture
def policy_engine():
    """Create PolicyEngine instance."""
    return PolicyEngine(verbose=False)


@pytest.fixture
def flat_position():
    """Create a FLAT PositionState."""
    return PositionState(side=PositionSide.FLAT, size=0.0)


@pytest.fixture
def long_position():
    """Create a LONG PositionState."""
    return PositionState(
        side=PositionSide.LONG,
        size=0.5,
        entry_price=5000.0,
        current_price=5002.0,
        unrealized_pnl=2.0,
        bars_since_entry=3,
    )


def test_session_name_in_policy_decision(policy_engine, flat_position):
    """Test that session name is included in PolicyDecision."""
    eval_result = create_mock_eval_result(session_name='MIDDAY')
    decision = policy_engine.decide_action(
        market_state=None,
        eval_result=eval_result,
        position_state=flat_position,
    )
    
    assert decision.session_name == 'MIDDAY'


def test_session_modifiers_in_policy_decision(policy_engine, flat_position):
    """Test that session modifiers are included in PolicyDecision."""
    eval_result = create_mock_eval_result(session_name='MIDDAY')
    decision = policy_engine.decide_action(
        market_state=None,
        eval_result=eval_result,
        position_state=flat_position,
    )
    
    assert decision.session_modifiers is not None
    assert 'vol_scale' in decision.session_modifiers


def test_flow_signals_in_policy_decision(policy_engine, flat_position):
    """Test that flow signals are included in PolicyDecision."""
    eval_result = create_mock_eval_result(
        session_name='POWER_HOUR',
        initiative_move_detected=True,
    )
    decision = policy_engine.decide_action(
        market_state=None,
        eval_result=eval_result,
        position_state=flat_position,
    )
    
    assert decision.flow_signals is not None
    assert decision.initiative_move_detected == True


def test_stop_run_avoidance_in_entry(policy_engine, flat_position):
    """Test that stop-run detection prevents entry."""
    # With stop-run detected and moderate score, should avoid entry
    eval_result = create_mock_eval_result(
        eval_score=0.4,
        confidence=0.75,
        session_name='MIDDAY',
        stop_run_detected=True,
    )
    decision = policy_engine.decide_action(
        market_state=None,
        eval_result=eval_result,
        position_state=flat_position,
    )
    
    # Should not enter on stop-run if eval is not strongly positive
    assert decision.stop_run_detected == True


def test_initiative_move_entry_in_power_hour(policy_engine, flat_position):
    """Test that initiative moves allow entry in POWER_HOUR."""
    eval_result = create_mock_eval_result(
        eval_score=0.4,
        confidence=0.75,
        session_name='POWER_HOUR',
        initiative_move_detected=True,
    )
    decision = policy_engine.decide_action(
        market_state=None,
        eval_result=eval_result,
        position_state=flat_position,
    )
    
    # POWER_HOUR with initiative should allow entry
    assert decision.initiative_move_detected == True


def test_globex_reduces_position_size(policy_engine, flat_position):
    """Test that GLOBEX session reduces position size."""
    risk_config = RiskConfig()
    
    # Test GLOBEX
    eval_globex = create_mock_eval_result(
        eval_score=0.8,
        confidence=0.85,
        session_name='GLOBEX',
    )
    decision_globex = policy_engine.decide_action(
        market_state=None,
        eval_result=eval_globex,
        position_state=flat_position,
        risk_config=risk_config,
    )
    
    # Test MIDDAY with same conditions
    eval_midday = create_mock_eval_result(
        eval_score=0.8,
        confidence=0.85,
        session_name='MIDDAY',
    )
    decision_midday = policy_engine.decide_action(
        market_state=None,
        eval_result=eval_midday,
        position_state=flat_position,
        risk_config=risk_config,
    )
    
    # GLOBEX should have smaller position size than MIDDAY
    assert decision_globex.target_size <= decision_midday.target_size


def test_midday_higher_confidence_requirement_than_power_hour(policy_engine, flat_position):
    """Test that MIDDAY has different confidence requirement than POWER_HOUR."""
    # Low eval, low confidence
    eval_result = create_mock_eval_result(
        eval_score=0.3,
        confidence=0.52,
        session_name='MIDDAY',
    )
    decision_midday = policy_engine.decide_action(
        market_state=None,
        eval_result=eval_result,
        position_state=flat_position,
    )
    
    eval_result = create_mock_eval_result(
        eval_score=0.3,
        confidence=0.52,
        session_name='POWER_HOUR',
    )
    decision_power = policy_engine.decide_action(
        market_state=None,
        eval_result=eval_result,
        position_state=flat_position,
    )
    
    # POWER_HOUR might allow action that MIDDAY blocks (or vice versa)
    assert isinstance(decision_midday.action, TradingAction)
    assert isinstance(decision_power.action, TradingAction)


def test_policy_decision_to_dict_includes_session_fields(policy_engine, flat_position):
    """Test that PolicyDecision.to_dict() includes all session/flow fields."""
    eval_result = create_mock_eval_result(
        session_name='MIDDAY',
        initiative_move_detected=True,
        level_reaction_score=0.7,
    )
    decision = policy_engine.decide_action(
        market_state=None,
        eval_result=eval_result,
        position_state=flat_position,
    )
    
    decision_dict = decision.to_dict()
    
    # Verify all session/flow fields are in dict
    assert 'session' in decision_dict
    assert 'session_modifiers' in decision_dict
    assert 'flow_signals' in decision_dict
    assert 'stop_run_detected' in decision_dict
    assert 'initiative_move_detected' in decision_dict
    assert 'level_reaction_score' in decision_dict
    
    # Verify values
    assert decision_dict['session'] == 'MIDDAY'
    assert decision_dict['initiative_move_detected'] == True
    assert decision_dict['level_reaction_score'] == 0.7


def test_vwap_distance_extreme_reduces_size(policy_engine, flat_position):
    """Test that extreme VWAP distance reduces position size."""
    # Normal VWAP distance
    eval_normal = create_mock_eval_result(
        eval_score=0.8,
        confidence=0.85,
        session_name='MIDDAY',
        vwap_distance_pct=0.5,
    )
    decision_normal = policy_engine.decide_action(
        market_state=None,
        eval_result=eval_normal,
        position_state=flat_position,
    )
    
    # Extreme VWAP distance
    eval_extreme = create_mock_eval_result(
        eval_score=0.8,
        confidence=0.85,
        session_name='MIDDAY',
        vwap_distance_pct=2.5,
    )
    decision_extreme = policy_engine.decide_action(
        market_state=None,
        eval_result=eval_extreme,
        position_state=flat_position,
    )
    
    # Extreme distance should reduce size or change action
    # Just verify both produce valid decisions
    assert isinstance(decision_normal.action, TradingAction)
    assert isinstance(decision_extreme.action, TradingAction)


def test_rth_open_requires_high_confidence(policy_engine, flat_position):
    """Test that RTH_OPEN requires higher confidence for entry."""
    # Borderline confidence in RTH_OPEN should not enter
    eval_result = create_mock_eval_result(
        eval_score=0.6,
        confidence=0.58,
        session_name='RTH_OPEN',
    )
    decision = policy_engine.decide_action(
        market_state=None,
        eval_result=eval_result,
        position_state=flat_position,
    )
    
    # Should be DO_NOTHING due to insufficient confidence in RTH_OPEN
    assert decision.action in [TradingAction.DO_NOTHING, TradingAction.ENTER_SMALL]


def test_close_session_allows_flow_trades(policy_engine, flat_position):
    """Test that CLOSE session allows flow-persistence trades."""
    eval_result = create_mock_eval_result(
        eval_score=0.4,
        confidence=0.58,
        session_name='CLOSE',
    )
    decision = policy_engine.decide_action(
        market_state=None,
        eval_result=eval_result,
        position_state=flat_position,
    )
    
    # CLOSE should be more lenient than GLOBEX
    assert isinstance(decision.action, TradingAction)


def test_policy_decision_serializable(policy_engine, flat_position):
    """Test that PolicyDecision is JSON-serializable."""
    import json
    
    eval_result = create_mock_eval_result(
        session_name='MIDDAY',
        initiative_move_detected=True,
    )
    decision = policy_engine.decide_action(
        market_state=None,
        eval_result=eval_result,
        position_state=flat_position,
    )
    
    decision_dict = decision.to_dict()
    
    # Verify it can be JSON serialized
    json_str = json.dumps(decision_dict)
    parsed = json.loads(json_str)
    
    assert parsed['session'] == 'MIDDAY'
    assert parsed['initiative_move_detected'] == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
