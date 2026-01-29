from datetime import datetime, timezone

import pytest

from ict_smc_features import compute_ict_smc_features


def test_empty_series_returns_defaults():
    res = compute_ict_smc_features([])
    assert res["premium_discount_state"] == "EQ"
    assert res["current_bullish_ob_low"] == 0.0
    assert res["in_london_killzone"] is False
    assert res["has_fvg"] is False


def test_bullish_orderblock_sets_mitigation():
    prices = [100.0, 99.0, 98.0, 101.0, 98.5]
    res = compute_ict_smc_features(prices)
    assert res["current_bullish_ob_low"] == 98.0
    assert res["current_bullish_ob_high"] == 99.0
    assert res["has_mitigation"] is True
    assert res["last_touched_ob_type"] == "BULLISH"
    assert res["mitigation_low"] == 98.0
    assert res["mitigation_high"] == 99.0
    assert res["has_flip_zone"] is False


def test_bearish_orderblock_sets_flip_zone():
    prices = [100.0, 101.0, 102.0, 99.0, 103.0]
    res = compute_ict_smc_features(prices)
    assert res["current_bearish_ob_low"] == 101.0
    assert res["current_bearish_ob_high"] == 102.0
    assert res["has_flip_zone"] is True
    assert res["flip_low"] == 101.0
    assert res["flip_high"] == 102.0
    assert res["has_mitigation"] is False


def test_fvg_and_ifvg_detection():
    prices = [110.0, 90.0, 80.0, 70.0, 60.0, 50.0]
    res = compute_ict_smc_features(prices)
    assert res["has_fvg"] is True
    assert res["has_ifvg"] is True
    assert res["fvg_upper"] == pytest.approx(110.0)
    assert res["fvg_lower"] == pytest.approx(90.0)
    assert res["ifvg_upper"] == pytest.approx(110.0)
    assert res["ifvg_lower"] == pytest.approx(90.0)


def test_premium_discount_and_killzones():
    ts = datetime(2023, 1, 2, 8, 30, tzinfo=timezone.utc).timestamp()
    prices = [100.0, 120.0, 80.0, 120.0]
    res = compute_ict_smc_features(prices, timestamps=[ts - 3, ts - 2, ts - 1, ts])
    assert res["premium_discount_state"] == "PREMIUM"
    assert res["equilibrium_level"] == pytest.approx(100.0)
    assert res["in_london_killzone"] is True
    assert res["in_ny_killzone"] is False
