from engine.evaluator import evaluate_state
from engine.market_state_builder import build_market_state
from engine.types import MarketState
from liquidity_depth_features import compute_depth_features
from momentum_features import (
    close_to_close_roc,
    compute_momentum_features,
    rolling_momentum,
)


def test_momentum_roc_windows_no_lookahead():
    prices = [100 + i for i in range(25)]
    feats = compute_momentum_features(prices)
    assert feats["roc_5"] > 0
    assert feats["roc_10"] > 0
    assert feats["roc_20"] > 0
    # Ensure no lookahead: dropping last price changes output
    feats_truncated = compute_momentum_features(prices[:-1])
    assert feats_truncated["roc_5"] != feats["roc_5"]


def test_early_history_neutrality():
    prices = [100.0, 100.1]
    assert close_to_close_roc(prices, 5) == 0.0
    assert rolling_momentum(prices, 5) == 0.0


def test_liquidity_depth_imbalance():
    ob = {
        "bids": [{"price": 99.0, "volume": 10.0}],
        "asks": [{"price": 101.0, "volume": 5.0}],
    }
    feats = compute_depth_features(ob)
    assert feats["bid_depth"] == 10.0
    assert feats["ask_depth"] == 5.0
    assert feats["depth_imbalance"] > 0
    assert feats["top_of_book_spread"] == 2.0


def test_replay_live_feature_parity():
    prices = [100.0, 100.5, 101.0, 101.5]
    a = compute_momentum_features(prices)
    b = compute_momentum_features(list(prices))
    assert a == b


def test_evaluator_momentum_liquidity_tilts():
    base = MarketState(
        current_price=101.0,
        ma_short=100.0,
        ma_long=99.0,
        momentum=0.2,
        recent_returns=[0.01] * 5,
        volatility=0.1,
        liquidity=0.5,
        rsi=55.0,
        trend_direction="UP",
    )
    tilted = MarketState(
        **{
            **base.__dict__,
            "momentum_20": 0.02,
            "roc_20": 0.02,
        }
    )
    depth_opposite = MarketState(
        **{
            **tilted.__dict__,
            "depth_imbalance": -0.5,
        }
    )

    conf_base = evaluate_state(base).confidence
    conf_tilt = evaluate_state(tilted).confidence
    conf_depth = evaluate_state(depth_opposite).confidence

    assert conf_tilt > conf_base
    assert conf_depth < conf_tilt


def test_market_state_builder_depth_and_momentum():
    events = [
        {
            "type": "book",
            "bids": [{"price": 100.0, "volume": 8.0}],
            "asks": [{"price": 100.2, "volume": 4.0}],
        }
    ]
    state = build_market_state(symbol="TEST", order_book_events=events, timestamp=0.0)
    assert state["bid_depth"] == 8.0
    assert state["ask_depth"] == 4.0
    assert state["depth_imbalance"] > 0
    assert "momentum_5" in state
    assert "roc_5" in state
