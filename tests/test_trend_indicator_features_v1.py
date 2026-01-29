import math

from trend_indicator_features import compute_trend_indicator_features


def test_ma_computation_against_known_series():
    prices = list(range(1, 11))
    features = compute_trend_indicator_features(prices)
    expected_sma = sum(prices) / len(prices)
    assert math.isclose(features["sma_20"], expected_sma, rel_tol=1e-9)
    expected_ema9 = 1.0
    alpha = 2.0 / (9 + 1)
    for price in prices[1:]:
        expected_ema9 = (price - expected_ema9) * alpha + expected_ema9
    assert math.isclose(features["ema_9"], expected_ema9, rel_tol=1e-6)


def test_ma_stacking_classification():
    bullish_series = list(range(1, 60))
    bearish_series = list(range(100, 40, -1))
    chop_series = [1, 2, 1, 2, 1, 2, 1, 2]

    assert (
        compute_trend_indicator_features(bullish_series)["ma_stack_state"] == "BULLISH"
    )
    assert (
        compute_trend_indicator_features(bearish_series)["ma_stack_state"] == "BEARISH"
    )
    assert compute_trend_indicator_features(chop_series)["ma_stack_state"] == "NEUTRAL"


def test_distance_from_ma_bounds():
    prices = [1000.0] * 10 + [1.0]
    features = compute_trend_indicator_features(prices)
    assert -5.0 <= features["distance_from_ema_20"] <= 5.0
    assert -5.0 <= features["distance_from_ema_50"] <= 5.0


def test_trend_strength_monotonicity():
    strong_trend = [float(x) for x in range(1, 51)]
    chop = [1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.7, 2.3, 1.9, 2.1]

    strong_features = compute_trend_indicator_features(strong_trend)
    chop_features = compute_trend_indicator_features(chop)

    assert strong_features["trend_strength"] > chop_features["trend_strength"]
    assert strong_features["trend_strength_state"] in {"MEDIUM", "STRONG"}
    assert chop_features["trend_strength_state"] in {"WEAK", "MEDIUM"}


def test_deterministic_replay_parity():
    prices = [10.0, 10.5, 10.2, 10.8, 11.0]
    first_pass = compute_trend_indicator_features(prices)
    second_pass = compute_trend_indicator_features(list(prices))
    assert first_pass == second_pass
