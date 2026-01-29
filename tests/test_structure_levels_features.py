from levels_features import compute_level_features
from structure_features import (
    SWING_HIGH,
    SWING_LOW,
    SWING_NONE,
    compute_structure_features,
    detect_swings,
)


def test_swing_detection_basic_patterns():
    prices = [1, 2, 3, 2, 1, 2]
    tags = detect_swings(prices, lookback=1, lookforward=1)
    assert SWING_HIGH in tags
    assert SWING_LOW in tags
    assert tags[-1] in {SWING_NONE, SWING_LOW, SWING_HIGH}


def test_bos_choch_detection():
    prices = [10, 9, 8, 9, 7, 6, 8, 11]
    feats = compute_structure_features(prices, lookback=1, lookforward=1)
    assert feats["last_bos_direction"] == "UP"
    assert feats["last_choch_direction"] in {"UP", "NONE"}
    assert feats["current_leg_type"] in {"IMPULSE", "CORRECTION"}


def test_session_and_day_levels():
    prices = [100, 101, 102, 98, 99, 105]
    timestamps = [0, 3600, 7200, 90000, 93600, 97200]
    sessions = ["ASIA", "ASIA", "ASIA", "LONDON", "LONDON", "LONDON"]
    feats = compute_level_features(
        prices, timestamps=timestamps, session_labels=sessions
    )
    assert feats["previous_day_high"] == 102
    assert feats["previous_day_low"] == 100
    assert feats["previous_day_close"] == 102
    assert feats["day_high"] == 105
    assert feats["day_low"] == 98
    assert feats["session_high"] == 105
    assert feats["session_low"] == 98


def test_vwap_stability():
    prices = [100, 101, 102]
    volumes = [1, 1, 2]
    feats = compute_level_features(prices, volumes=volumes)
    assert feats["vwap_price"] > 0
    assert feats["distance_from_vwap"] == feats["distance_from_vwap"]  # not NaN


def test_replay_live_parity_for_structure_levels():
    prices = [1, 2, 1.5, 2.5, 2.0]
    first = compute_structure_features(prices)
    second = compute_structure_features(list(prices))
    assert first == second
    lvl1 = compute_level_features(prices)
    lvl2 = compute_level_features(list(prices))
    assert lvl1 == lvl2
