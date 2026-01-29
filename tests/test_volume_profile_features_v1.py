from volume_profile_features import (
    VolumeProfileConfig,
    build_volume_profile,
    compute_volume_profile_features,
)


def _bars(prices, vols):
    return list(zip(prices, vols))


def test_profile_construction_on_synthetic_data():
    prices = [100.0, 101.0, 102.0]
    vols = [10.0, 20.0, 30.0]
    cfg = VolumeProfileConfig(price_bin_size=1.0)
    profile = build_volume_profile(_bars(prices, vols), cfg)
    assert len(profile.bins) == 3
    assert profile.poc_price == 102.0
    assert profile.value_area_low <= profile.poc_price <= profile.value_area_high


def test_poc_detection():
    prices = [100.0, 101.0, 99.5]
    vols = [5.0, 50.0, 10.0]
    cfg = VolumeProfileConfig(price_bin_size=0.5)
    profile = build_volume_profile(_bars(prices, vols), cfg)
    assert profile.poc_price == 101.0


def test_hvn_lvn_classification():
    prices = [100.0, 101.0, 102.0, 103.0]
    vols = [100.0, 80.0, 10.0, 5.0]
    cfg = VolumeProfileConfig(
        price_bin_size=1.0, hvn_threshold_ratio=0.7, lvn_threshold_ratio=0.3
    )
    profile = build_volume_profile(_bars(prices, vols), cfg)
    assert set(profile.hvn_levels) == {100.0, 101.0}
    assert set(profile.lvn_levels) == {102.0, 103.0}


def test_value_area_coverage_and_bounds():
    prices = [100.0, 101.0, 102.0]
    vols = [50.0, 40.0, 10.0]
    cfg = VolumeProfileConfig(price_bin_size=1.0, value_area_target=0.7)
    profile = build_volume_profile(_bars(prices, vols), cfg)
    assert profile.value_area_low == 100.0
    assert profile.value_area_high == 101.0
    assert profile.value_area_coverage >= 0.7


def test_price_vs_value_area_state():
    prices = [100.0, 101.0, 102.0, 103.0]
    vols = [50.0, 40.0, 30.0, 5.0]
    cfg = VolumeProfileConfig(price_bin_size=1.0)
    feats = compute_volume_profile_features(prices, vols, config=cfg)
    assert feats["price_vs_value_area_state"] == "ABOVE"


def test_near_hvn_lvn_flags():
    prices = [100.0, 104.0, 100.6]
    vols = [100.0, 1.0, 40.0]
    cfg = VolumeProfileConfig(price_bin_size=1.0, proximity_band=1.5)
    feats = compute_volume_profile_features(prices, vols, config=cfg)
    assert feats["near_hvn"] is True
    assert feats["near_lvn"] is False


def test_replay_live_parity_for_volume_profile():
    prices = [100.0, 101.0, 102.0]
    vols = [10.0, 10.0, 10.0]
    live_prices = prices + [102.05]
    live_vols = vols + [0.1]
    cfg = VolumeProfileConfig(price_bin_size=1.0)
    replay = compute_volume_profile_features(prices, vols, config=cfg)
    live = compute_volume_profile_features(live_prices, live_vols, config=cfg)
    assert replay == live
