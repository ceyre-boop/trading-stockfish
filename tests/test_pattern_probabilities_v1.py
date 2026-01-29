from pattern_stats import PatternProbabilities, compute_pattern_probabilities


def test_probability_computation_from_synthetic_history():
    history = [
        {
            "current_price": 100.0,
            "last_sweep_direction": "UP",
            "swept_bsl": True,
            "current_bullish_ob_low": 99.0,
            "current_bullish_ob_high": 101.0,
            "has_fvg": True,
            "fvg_lower": 99.0,
            "fvg_upper": 101.0,
        },
        {"current_price": 99.0},  # reversal after sweep
        {
            "current_price": 100.0,
            "current_bullish_ob_low": 99.0,
            "current_bullish_ob_high": 101.0,
        },
        {"current_price": 100.5},  # OB hold
        {
            "current_price": 102.0,
            "has_fvg": True,
            "fvg_lower": 101.0,
            "fvg_upper": 103.0,
        },
        {"current_price": 102.0},  # FVG fill (inside band)
    ]

    probs = compute_pattern_probabilities(history, lookahead=1)
    assert isinstance(probs, PatternProbabilities)
    assert 0.5 < probs.p_sweep_reversal < 1.0
    assert 0.0 < probs.p_sweep_continuation < 0.6
    assert 0.5 < probs.p_ob_hold <= 1.0
    assert probs.p_ob_fail > 0.0
    assert probs.p_fvg_fill > 0.5


def test_laplace_smoothing_behavior():
    empty_probs = compute_pattern_probabilities([])
    assert empty_probs.p_sweep_reversal == 0.5
    assert empty_probs.p_fvg_fill == 0.5

    # No outcomes but one condition -> still smoothed away from 0/1
    history = [
        {
            "current_price": 100.0,
            "last_sweep_direction": "UP",
            "swept_bsl": True,
        },
        {"current_price": 100.0},
    ]
    probs = compute_pattern_probabilities(history, lookahead=1)
    assert 0.3 < probs.p_sweep_reversal < 0.7
    assert 0.3 < probs.p_sweep_continuation < 0.7


def test_expectancy_computation_and_clipping():
    history = [
        {
            "current_price": 100.0,
            "last_sweep_direction": "UP",
            "swept_bsl": True,
        },
        {"current_price": 80.0},  # large move should be clipped
    ]
    probs = compute_pattern_probabilities(history, lookahead=1)
    assert probs.expected_move_after_sweep >= -5.0
    assert probs.expected_move_after_sweep <= 0.0

    # Add large positive move for fvg expectancy
    history = [
        {
            "current_price": 100.0,
            "has_fvg": True,
            "fvg_lower": 95.0,
            "fvg_upper": 105.0,
        },
        {"current_price": 120.0},
    ]
    probs = compute_pattern_probabilities(history, lookahead=1)
    assert probs.expected_move_after_fvg_fill <= 5.0
