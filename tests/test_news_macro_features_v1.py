import math

import pytest

from news_macro_features import (
    classify_macro_regime,
    compute_news_macro_features,
    get_upcoming_events,
    load_event_calendar,
)


def test_event_lookup_and_time_delta():
    calendar = load_event_calendar(
        [
            {"event_type": "CPI", "timestamp": 1_060.0, "expected_impact": "HIGH"},
            {"event_type": "NFP", "timestamp": 2_400.0, "expected_impact": "MEDIUM"},
        ]
    )
    snapshot = get_upcoming_events(calendar, current_time=1_000.0, horizon_minutes=5)
    assert snapshot["next_event_type"] == "CPI"
    assert math.isclose(snapshot["next_event_time_delta"], 1.0, rel_tol=1e-6)
    assert snapshot["event_risk_window"] == "PRE_EVENT"


def test_pre_event_and_post_event_windows():
    calendar = load_event_calendar(
        [{"event_type": "FOMC", "timestamp": 12_000.0, "expected_impact": "HIGH"}]
    )
    pre_snapshot = get_upcoming_events(
        calendar, current_time=12_000.0 - 3_600.0, horizon_minutes=180
    )
    assert pre_snapshot["event_risk_window"] == "PRE_EVENT"
    post_snapshot = get_upcoming_events(
        calendar, current_time=12_000.0 + 600.0, horizon_minutes=180
    )
    assert post_snapshot["event_risk_window"] == "POST_EVENT"


def test_expected_volatility_state_for_each_event_type():
    calendar = load_event_calendar(
        [
            {"event_type": "FOMC", "timestamp": 10_000.0, "expected_impact": "HIGH"},
            {"event_type": "PMI", "timestamp": 11_000.0, "expected_impact": "MEDIUM"},
            {
                "event_type": "FedSpeaker",
                "timestamp": 12_000.0,
                "expected_impact": "LOW",
            },
        ]
    )
    res_fomc = get_upcoming_events(calendar, current_time=9_500.0, horizon_minutes=120)
    assert res_fomc["expected_volatility_state"] == "EXTREME"
    res_pmi = get_upcoming_events(
        calendar[1:], current_time=10_900.0, horizon_minutes=30
    )
    assert res_pmi["expected_volatility_state"] == "MEDIUM"
    res_speaker = get_upcoming_events(
        calendar[2:], current_time=11_950.0, horizon_minutes=20
    )
    assert res_speaker["expected_volatility_state"] == "LOW"


def test_liquidity_withdrawal_flag_behavior():
    calendar = load_event_calendar(
        [{"event_type": "CPI", "timestamp": 20_000.0, "expected_impact": "HIGH"}]
    )
    snapshot = get_upcoming_events(
        calendar, current_time=20_000.0 - 1_800.0, horizon_minutes=120
    )
    assert snapshot["event_risk_window"] == "PRE_EVENT"
    assert snapshot["liquidity_withdrawal_flag"] is True
    outside = get_upcoming_events(calendar, current_time=10_000.0, horizon_minutes=60)
    assert outside["liquidity_withdrawal_flag"] is False


def test_macro_regime_classification_on_synthetic_inputs():
    risk_on_regime, risk_on_score = classify_macro_regime(
        {"vix": 12.0, "dxy": 102.5, "us10y": 3.2, "spx_trend": 0.5}
    )
    assert risk_on_regime == "RISK_ON"
    assert risk_on_score > 0.6
    risk_off_regime, risk_off_score = classify_macro_regime(
        {"vix": 30.0, "dxy": 106.0, "us10y": 4.1, "spx_trend": -0.2}
    )
    assert risk_off_regime == "RISK_OFF"
    assert risk_off_score > 0.6


def test_replay_live_parity_for_news_macro_features():
    calendar = load_event_calendar(
        [
            {"event_type": "NFP", "timestamp": 50_000.0, "expected_impact": "HIGH"},
            {"event_type": "PMI", "timestamp": 60_000.0, "expected_impact": "MEDIUM"},
        ]
    )
    inputs = {"vix": 18.0, "dxy": 104.0, "us10y": 3.8, "spx_trend": 0.1}
    snap_a = compute_news_macro_features(
        current_time=49_000.0,
        calendar_data=calendar,
        macro_inputs=inputs,
        horizon_minutes=120,
    )
    snap_b = compute_news_macro_features(
        current_time=49_000.0,
        calendar_data=calendar,
        macro_inputs=inputs,
        horizon_minutes=120,
    )
    assert snap_a == snap_b
