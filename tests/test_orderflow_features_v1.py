import pytest

from orderflow_features import compute_orderflow_features


def test_bar_and_cumulative_delta():
    prices = [100.0, 101.0, 102.0]
    buys = [1.0, 2.0, 3.0]
    sells = [0.5, 1.0, 1.0]
    res = compute_orderflow_features(prices, buys, sells)
    assert res["bar_delta"] == pytest.approx(2.0)
    assert res["cumulative_delta"] == pytest.approx(3.5)


def test_footprint_imbalance_bounds_and_symmetry():
    prices = [100.0, 100.5]
    buys = [10.0, 5.0]
    sells = [0.0, 5.0]
    res_buy = compute_orderflow_features(prices, buys, sells)
    assert -1.0 <= res_buy["footprint_imbalance"] <= 1.0
    assert res_buy["footprint_imbalance"] == pytest.approx(0.0)

    res_sell = compute_orderflow_features(prices, sells, buys)
    assert res_sell["footprint_imbalance"] == pytest.approx(0.0)

    res_all_buy = compute_orderflow_features(prices, [10.0, 10.0], [0.0, 0.0])
    assert res_all_buy["footprint_imbalance"] == pytest.approx(1.0)

    res_all_sell = compute_orderflow_features(prices, [0.0, 0.0], [10.0, 10.0])
    assert res_all_sell["footprint_imbalance"] == pytest.approx(-1.0)


def test_absorption_detection():
    prices = [100.0, 100.01]
    buys = [5.0, 200.0]
    sells = [5.0, 20.0]
    res = compute_orderflow_features(prices, buys, sells)
    assert res["has_absorption"] is True
    assert res["absorption_side"] == "BUY"


def test_exhaustion_detection():
    prices = [100.0, 102.0, 102.1]
    buys = [0.0, 120.0, 10.0]
    sells = [0.0, 5.0, 2.0]
    res = compute_orderflow_features(prices, buys, sells)
    assert res["has_exhaustion"] is True
    assert res["exhaustion_side"] == "BUY"


def test_replay_live_parity_for_orderflow():
    prices = [99.0, 100.0, 101.0]
    buys = [3.0, 4.0, 5.0]
    sells = [1.0, 1.5, 2.0]
    first = compute_orderflow_features(prices, buys, sells)
    second = compute_orderflow_features(prices, buys, sells)
    assert first == second
