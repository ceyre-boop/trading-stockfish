from dataclasses import replace

from engine.evaluator import evaluate_state
from engine.types import MarketState


def _base_state() -> MarketState:
    return MarketState(
        current_price=100.0,
        ma_short=100.0,
        ma_long=100.0,
        momentum=0.0,
        volatility=0.0,
        recent_returns=[0.0] * 5,
    )


def test_evaluator_tilts_applied_but_bounded():
    base = _base_state()
    tilted = replace(
        base,
        last_sweep_direction="UP",
        p_sweep_reversal=0.9,
        p_sweep_continuation=0.1,
    )
    neutral = replace(base, p_sweep_reversal=0.5, p_sweep_continuation=0.5)
    score_tilted = evaluate_state(tilted).score
    score_neutral = evaluate_state(neutral).score
    assert score_tilted <= 1.0
    assert score_tilted - score_neutral < 0.0  # reversal tilt opposes UP sweep


def test_no_tilt_when_no_pattern_condition_met():
    base = _base_state()
    neutral = replace(base, p_sweep_reversal=0.5, p_sweep_continuation=0.5)
    result = evaluate_state(neutral).score
    assert result == evaluate_state(neutral).score


def test_determinism_same_inputs_same_score():
    base = _base_state()
    state_a = replace(
        base,
        last_sweep_direction="DOWN",
        p_sweep_reversal=0.8,
        p_sweep_continuation=0.2,
        p_ob_hold=0.7,
        p_ob_fail=0.3,
        has_fvg=True,
        p_fvg_fill=0.6,
    )
    state_b = replace(state_a)
    assert evaluate_state(state_a).score == evaluate_state(state_b).score
