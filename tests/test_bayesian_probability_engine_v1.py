import pytest

from bayesian_probability_engine import compute_bayesian_probabilities

EXPECTED_KEYS = {
    "bayes_trend_continuation",
    "bayes_trend_continuation_confidence",
    "bayes_trend_reversal",
    "bayes_trend_reversal_confidence",
    "bayes_sweep_reversal",
    "bayes_sweep_reversal_confidence",
    "bayes_sweep_continuation",
    "bayes_sweep_continuation_confidence",
    "bayes_ob_respect",
    "bayes_ob_respect_confidence",
    "bayes_ob_violation",
    "bayes_ob_violation_confidence",
    "bayes_fvg_fill",
    "bayes_fvg_fill_confidence",
    "bayes_fvg_reject",
    "bayes_fvg_reject_confidence",
    "bayesian_update_strength",
}


def test_prior_initialization():
    res = compute_bayesian_probabilities({})
    missing = EXPECTED_KEYS - set(res.keys())
    assert not missing, f"Missing keys: {missing}"
    assert 0.4 < res["bayes_trend_continuation"] < 0.7
    for key in EXPECTED_KEYS:
        assert 0.0 <= res[key] <= 1.0


def test_update_behavior_for_strong_evidence():
    state = {
        "trend_strength": 0.9,
        "momentum_regime": "STRONG_TREND",
        "p_sweep_reversal": 0.55,
        "p_sweep_continuation": 0.6,
    }
    res = compute_bayesian_probabilities(state)
    assert res["bayes_trend_continuation"] > 0.7
    assert res["bayes_trend_reversal"] < 0.4
    assert res["bayes_trend_continuation_confidence"] > 0.3


def test_update_behavior_for_conflicting_evidence():
    state = {
        "trend_strength": 0.2,
        "momentum_regime": "CHOP",
        "rsi_bearish_divergence": True,
        "p_sweep_reversal": 0.45,
        "p_sweep_continuation": 0.55,
    }
    res = compute_bayesian_probabilities(state)
    assert res["bayes_trend_reversal"] > res["bayes_trend_continuation"]
    assert res["bayes_trend_reversal_confidence"] >= 0.1


def test_clamping_bounds():
    high_state = {
        "p_fvg_fill": 0.99,
        "has_fvg": True,
        "expected_volatility_state": "EXTREME",
    }
    res_high = compute_bayesian_probabilities(high_state)
    assert res_high["bayes_fvg_fill"] <= 0.99
    low_state = {
        "p_fvg_fill": 0.0,
        "has_fvg": False,
        "expected_volatility_state": "LOW",
    }
    res_low = compute_bayesian_probabilities(low_state)
    assert res_low["bayes_fvg_fill"] >= 0.01


def test_confidence_score_behavior():
    strong = compute_bayesian_probabilities(
        {"trend_strength": 0.8, "momentum_regime": "TREND"}
    )
    weak = compute_bayesian_probabilities(
        {"trend_strength": 0.05, "momentum_regime": "CHOP"}
    )
    assert (
        strong["bayes_trend_continuation_confidence"]
        > weak["bayes_trend_continuation_confidence"]
    )


def test_replay_live_parity():
    state = {
        "trend_strength": 0.6,
        "momentum_regime": "TREND",
        "last_sweep_direction": "UP",
        "has_absorption": True,
        "has_exhaustion": False,
        "p_sweep_reversal": 0.55,
    }
    res_a = compute_bayesian_probabilities(state)
    res_b = compute_bayesian_probabilities(state)
    assert res_a == res_b
