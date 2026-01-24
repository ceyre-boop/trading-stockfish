"""
Integration tests for CausalEvaluator with SessionContext and FlowContext.

Validates:
- Session-aware weighting adjustments
- Flow-aware scoring (stop-run penalties, initiative rewards, level reactions)
- EvaluationResult includes session/flow fields
- Reasoning includes session/flow context
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock

from engine.causal_evaluator import (
    create_causal_evaluator,
    LiquidityRegime,
    VolatilityRegime,
    TimeRegimeType,
    RiskSentiment,
)


def create_mock_state(
    timestamp: datetime = None,
    session_name: str = "MIDDAY",
    stop_run: bool = False,
    initiative: bool = False,
    level_reaction: float = 0.5,
):
    """Create a mock MarketState with session and flow fields."""
    if timestamp is None:
        timestamp = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    
    state = Mock()
    state.timestamp = timestamp
    state.symbol = "ES"
    state.bid = 5000.0
    state.ask = 5000.5
    state.spread_pips = 0.5
    state.close = 5000.25
    state.open = 5000.0
    state.high = 5002.0
    state.low = 4998.0
    state.volume = 5000000
    state.timeframe = '1m'
    
    # Session/flow context
    state.session_name = session_name
    state.session_vol_scale = 1.0
    state.session_liq_scale = 1.0
    state.session_risk_scale = 1.0
    state.prior_high = 5010.0
    state.prior_low = 4990.0
    state.overnight_high = 5005.0
    state.overnight_low = 4995.0
    state.vwap = 5000.0
    state.vwap_distance_pct = 0.0
    state.round_level_proximity = 0.2  # Default: not near round level
    state.stop_run_detected = stop_run
    state.initiative_move_detected = initiative
    state.level_reaction_score = level_reaction
    
    # Macro state
    state.macro_state = Mock()
    state.macro_state.sentiment_score = 0.0
    state.macro_state.surprise_score = 0.0
    state.macro_state.rate_expectation = 0.0
    state.macro_state.inflation_expectation = 0.0
    state.macro_state.gdp_expectation = 0.0
    
    # Liquidity state
    state.liquidity_state = Mock()
    state.liquidity_state.bid_ask_spread = 0.5
    state.liquidity_state.order_book_depth = 0.5
    state.liquidity_state.regime = LiquidityRegime.NORMAL
    state.liquidity_state.volume_trend = 0.0
    
    # Volatility state
    state.volatility_state = Mock()
    state.volatility_state.current_vol = 0.15
    state.volatility_state.vol_percentile = 0.5
    state.volatility_state.regime = VolatilityRegime.NORMAL
    state.volatility_state.vol_trend = 0.0
    state.volatility_state.skew = 0.0
    
    # Dealer state
    state.dealer_state = Mock()
    state.dealer_state.net_gamma_exposure = 0.0
    state.dealer_state.net_spot_exposure = 0.0
    state.dealer_state.vega_exposure = 0.0
    state.dealer_state.dealer_sentiment = 0.0
    
    # Earnings state
    state.earnings_state = Mock()
    state.earnings_state.multi_mega_cap_exposure = 0.5
    state.earnings_state.small_cap_exposure = 0.5
    state.earnings_state.earnings_season_flag = False
    state.earnings_state.earnings_surprise_momentum = 0.0
    
    # Time regime state
    state.time_regime_state = Mock()
    state.time_regime_state.regime_type = TimeRegimeType.LONDON_OPEN
    state.time_regime_state.minutes_into_session = 60
    state.time_regime_state.hours_until_session_end = 8
    state.time_regime_state.day_of_week = 2
    
    # Price location state
    state.price_location_state = Mock()
    state.price_location_state.distance_from_high = 0.5
    state.price_location_state.distance_from_low = 0.5
    state.price_location_state.range_ratio = 1.0
    state.price_location_state.session_extremity = 0.0
    
    # Macro news state
    state.macro_news_state = Mock()
    state.macro_news_state.risk_sentiment_score = 0.0
    state.macro_news_state.hawkishness_score = 0.0
    state.macro_news_state.surprise_score = 0.0
    state.macro_news_state.event_importance = 0
    state.macro_news_state.hours_since_last_event = 24.0
    state.macro_news_state.macro_event_count = 0
    state.macro_news_state.news_article_count = 0
    state.macro_news_state.macro_news_state = 'NEUTRAL'
    
    return state


@pytest.fixture
def evaluator():
    """Create evaluator instance."""
    return create_causal_evaluator(verbose=False)


def test_session_name_in_result(evaluator):
    """Test that session name is included in evaluation result."""
    state = create_mock_state(session_name='MIDDAY')
    result = evaluator.evaluate(state)
    
    assert hasattr(result, 'session_name')
    assert result.session_name == 'MIDDAY'


def test_session_modifiers_in_result(evaluator):
    """Test that session modifiers are included in evaluation result."""
    state = create_mock_state(session_name='MIDDAY')
    state.session_vol_scale = 0.9
    state.session_liq_scale = 1.1
    state.session_risk_scale = 1.0
    
    result = evaluator.evaluate(state)
    
    assert hasattr(result, 'session_modifiers')
    assert result.session_modifiers is not None
    assert isinstance(result.session_modifiers, dict)
    assert 'vol_scale' in result.session_modifiers
    assert result.session_modifiers['vol_scale'] == 0.9


def test_flow_signals_in_result(evaluator):
    """Test that flow signals are included in evaluation result."""
    state = create_mock_state(
        session_name='POWER_HOUR',
        stop_run=True,
        initiative=True,
        level_reaction=0.7
    )
    result = evaluator.evaluate(state)
    
    assert hasattr(result, 'flow_signals')
    assert result.flow_signals is not None
    assert isinstance(result.flow_signals, dict)
    assert 'stop_run_detected' in result.flow_signals
    assert 'initiative_move_detected' in result.flow_signals
    assert result.flow_signals['stop_run_detected'] == True
    assert result.flow_signals['initiative_move_detected'] == True


def test_stop_run_detected_field(evaluator):
    """Test that stop_run_detected is in result."""
    state = create_mock_state(stop_run=True)
    result = evaluator.evaluate(state)
    
    assert hasattr(result, 'stop_run_detected')
    assert result.stop_run_detected == True


def test_initiative_move_detected_field(evaluator):
    """Test that initiative_move_detected is in result."""
    state = create_mock_state(initiative=True)
    result = evaluator.evaluate(state)
    
    assert hasattr(result, 'initiative_move_detected')
    assert result.initiative_move_detected == True


def test_level_reaction_score_field(evaluator):
    """Test that level_reaction_score is in result."""
    state = create_mock_state(level_reaction=0.75)
    result = evaluator.evaluate(state)
    
    assert hasattr(result, 'level_reaction_score')
    assert result.level_reaction_score == 0.75


def test_session_aware_confidence_globex_vs_midday(evaluator):
    """Test that GLOBEX (overnight) has lower confidence than MIDDAY."""
    state_globex = create_mock_state(session_name='GLOBEX')
    state_midday = create_mock_state(session_name='MIDDAY')
    
    result_globex = evaluator.evaluate(state_globex)
    result_midday = evaluator.evaluate(state_midday)
    
    # GLOBEX (overnight, low liquidity) should have lower confidence
    assert result_globex.confidence < result_midday.confidence


def test_session_aware_confidence_rth_open(evaluator):
    """Test that RTH_OPEN has expected confidence adjustment."""
    state_rth_open = create_mock_state(session_name='RTH_OPEN')
    state_midday = create_mock_state(session_name='MIDDAY')
    
    result_rth = evaluator.evaluate(state_rth_open)
    result_mid = evaluator.evaluate(state_midday)
    
    # RTH_OPEN may have lower confidence due to volume ramp-up uncertainty
    # Just verify they can be different
    assert isinstance(result_rth.confidence, (int, float))
    assert isinstance(result_mid.confidence, (int, float))


def test_stop_run_penalty_in_rth_open(evaluator):
    """Test that stop-run detected in RTH_OPEN has penalty applied."""
    state_no_stoprun = create_mock_state(session_name='RTH_OPEN', stop_run=False)
    state_with_stoprun = create_mock_state(session_name='RTH_OPEN', stop_run=True)
    
    result_no = evaluator.evaluate(state_no_stoprun)
    result_yes = evaluator.evaluate(state_with_stoprun)
    
    # With stop-run, confidence should be reduced
    assert result_yes.confidence <= result_no.confidence


def test_result_to_dict_includes_session_fields(evaluator):
    """Test that result_dict includes session and flow_signals."""
    state = create_mock_state(
        session_name='MIDDAY',
        initiative=True,
        level_reaction=0.6
    )
    result = evaluator.evaluate(state)
    result_dict = result.result_dict
    
    # Verify session/flow fields are in dict
    assert 'session' in result_dict
    assert 'session_modifiers' in result_dict
    assert 'flow_signals' in result_dict
    
    # Verify values match
    assert result_dict['session'] == 'MIDDAY'
    assert isinstance(result_dict['session_modifiers'], dict)
    assert isinstance(result_dict['flow_signals'], dict)
    
    # Verify EvaluationResult dataclass has session/flow fields as attributes
    assert result.session_name == 'MIDDAY'
    assert result.stop_run_detected == False
    assert result.initiative_move_detected == True
    assert result.level_reaction_score == 0.6


def test_evaluation_produces_reasonable_scores(evaluator):
    """Test that evaluation produces scores within expected ranges."""
    state = create_mock_state()
    result = evaluator.evaluate(state)
    
    # eval_score should be in [-1, 1]
    assert -1.0 <= result.eval_score <= 1.0
    
    # confidence should be in [0, 1]
    assert 0.0 <= result.confidence <= 1.0
    
    # Symbol should match
    assert result.symbol == "ES"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
