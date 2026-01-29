from orderbook_features import (
    OrderBookFeaturesConfig,
    build_l2_snapshot,
    compute_orderbook_features,
)


def test_l2_aggregation_and_sorting():
    raw = {
        "bids": [{"price": 100.0, "volume": 5.0}, {"price": 99.5, "volume": 3.0}],
        "asks": [(100.5, 2.0), (101.0, 4.0)],
    }
    cfg = OrderBookFeaturesConfig(max_levels_per_side=2, aggregation_mode="RAW")
    snap = build_l2_snapshot(raw, cfg)
    assert snap.bids[0].price > snap.bids[-1].price
    assert snap.asks[0].price < snap.asks[-1].price
    assert len(snap.bids) == 2 and len(snap.asks) == 2

    raw_tick = {
        "bids": [{"price": 100.01, "volume": 2.0}, {"price": 100.02, "volume": 3.0}],
        "asks": [{"price": 100.49, "volume": 1.0}, {"price": 100.51, "volume": 1.5}],
    }
    cfg_tick = OrderBookFeaturesConfig(
        max_levels_per_side=2, aggregation_mode="AGGREGATED_BY_TICK", tick_size=0.05
    )
    snap_tick = build_l2_snapshot(raw_tick, cfg_tick)
    # Aggregated bids collapse into one level around 100.0
    assert len(snap_tick.bids) == 1
    assert snap_tick.bids[0].size == 5.0


def test_top_level_and_multi_level_imbalance():
    raw = {
        "bids": [(100.0, 10.0), (99.5, 5.0)],
        "asks": [(100.5, 5.0), (101.0, 5.0)],
    }
    feats = compute_orderbook_features(raw)
    assert round(feats["top_level_imbalance"], 3) == round((10 - 5) / (10 + 5), 3)
    assert round(feats["multi_level_imbalance"], 3) == round((15 - 10) / (25), 3)


def test_spread_dynamics_classification():
    raw = {"bids": [(100.0, 5.0)], "asks": [(100.4, 5.0)]}
    cfg = OrderBookFeaturesConfig(tick_size=0.1, spread_shift_threshold=0.1)
    feats = compute_orderbook_features(
        raw, config=cfg, recent_spreads=[0.2, 0.22, 0.21]
    )
    assert feats["microstructure_shift"] == "WIDENING"
    assert feats["spread_widening"] is True


def test_hidden_liquidity_detection_on_synthetic_sequences():
    prev_raw = {"bids": [(100.0, 10.0)], "asks": [(100.5, 5.0)]}
    prev_snap = build_l2_snapshot(prev_raw)
    curr_raw = {"bids": [(100.0, 10.0)], "asks": [(100.5, 5.0)]}
    feats = compute_orderbook_features(
        curr_raw,
        aggressive_sell=50.0,
        prev_snapshot=prev_snap,
        config=OrderBookFeaturesConfig(hidden_aggressive_threshold=20.0),
    )
    assert feats["hidden_bid_liquidity"] is True
    assert feats["hidden_ask_liquidity"] is False


def test_queue_position_estimate_bounds():
    raw = {"bids": [(100.0, 20.0)], "asks": [(100.5, 10.0)]}
    feats = compute_orderbook_features(
        raw, config=OrderBookFeaturesConfig(queue_fraction=0.3)
    )
    assert 0.0 <= feats["queue_position_estimate"] <= 1.0


def test_replay_live_parity_for_orderbook():
    raw = {"bids": [(100.0, 10.0), (99.5, 0.05)], "asks": [(100.5, 8.0)]}
    live_raw = {
        "bids": [(100.0, 10.0), (99.5, 0.05), (99.0, 0.001)],
        "asks": [(100.5, 8.0)],
    }
    feats_replay = compute_orderbook_features(raw)
    feats_live = compute_orderbook_features(live_raw)
    assert feats_replay == feats_live
