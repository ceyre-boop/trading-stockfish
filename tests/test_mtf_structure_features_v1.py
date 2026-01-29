import pytest

from mtf_structure_features import compute_mtf_structure_features


def _series(
    start: float, step: float, count: int, interval: float
) -> tuple[list[float], list[float]]:
    prices = [start + step * i for i in range(count)]
    timestamps = [i * interval for i in range(count)]
    return prices, timestamps


def test_htf_swings_and_bos_choch_on_synthetic_trend():
    prices, timestamps = _series(100.0, 1.5, 40, 900)  # 15m bars trending up
    features = compute_mtf_structure_features(prices, timestamps, timeframes=("4H",))
    assert features["htf_4h_trend_direction"] == "UP"
    assert features["htf_4h_current_leg_type"] == "IMPULSE"
    assert features["htf_4h_trend_strength"] > 10.0


def test_htf_trend_direction_and_strength():
    prices, timestamps = _series(200.0, -2.0, 60, 1800)  # 30m bars trending down
    features = compute_mtf_structure_features(prices, timestamps, timeframes=("D",))
    assert features["htf_d_trend_direction"] == "DOWN"
    assert features["htf_d_trend_strength"] > 20.0


def test_fractal_compression_expansion_behavior():
    prices, timestamps = _series(100.0, 1.0, 50, 900)
    compressed = compute_mtf_structure_features(
        prices,
        timestamps,
        timeframes=("4H",),
        ltf_trend_direction="UP",
        ltf_trend_strength=70.0,
        ltf_volatility=0.1,
    )
    assert compressed["fractal_state"] == "COMPRESSED"
    expanded = compute_mtf_structure_features(
        prices,
        timestamps,
        timeframes=("4H",),
        ltf_trend_direction="UP",
        ltf_trend_strength=70.0,
        ltf_volatility=0.8,
    )
    assert expanded["fractal_state"] == "EXPANDING"


def test_alignment_score_sign_and_magnitude():
    prices, timestamps = _series(50.0, 0.8, 48, 1800)
    aligned = compute_mtf_structure_features(
        prices,
        timestamps,
        timeframes=("1H",),
        ltf_trend_direction="UP",
        ltf_trend_strength=80.0,
    )
    assert aligned["htf_ltf_alignment_score"] > 0.1
    counter = compute_mtf_structure_features(
        prices,
        timestamps,
        timeframes=("1H",),
        ltf_trend_direction="DOWN",
        ltf_trend_strength=80.0,
    )
    assert counter["htf_ltf_alignment_score"] < -0.05


def test_htf_bias_classification():
    prices, timestamps = _series(300.0, -3.5, 72, 1800)
    features = compute_mtf_structure_features(
        prices, timestamps, timeframes=("4H", "D")
    )
    assert features["htf_bias"] == "BEARISH"


def test_replay_live_parity_for_mtf_structure():
    prices, timestamps = _series(100.0, 0.5, 45, 900)
    live_prices = prices + [prices[-1] + 0.25]
    live_ts = timestamps + [timestamps[-1] + 300]  # partial bucket should be ignored
    snapshot_replay = compute_mtf_structure_features(
        prices, timestamps, timeframes=("4H",)
    )
    snapshot_live = compute_mtf_structure_features(
        live_prices, live_ts, timeframes=("4H",)
    )
    assert snapshot_replay == snapshot_live
