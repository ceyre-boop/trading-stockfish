from datetime import datetime

import pytest

from engine.market_profile_features import MarketProfileFeatures
from engine.market_profile_model import (
    TrainedMarketProfileModel,
    predict_market_profile_proba,
    train_market_profile_model,
)
from engine.market_profile_rules import coarse_classify_market_profile
from engine.market_profile_state_machine import MarketProfileStateMachine
from engine.structure_brain import classify_market_profile


def _features(**overrides) -> MarketProfileFeatures:
    base = dict(
        timestamp_utc=datetime(2026, 2, 4),
        session_context="NY",
        time_of_day_bucket="OPEN",
        dist_pdh=5.0,
        dist_pdl=5.0,
        dist_prev_session_high=4.0,
        dist_prev_session_low=4.0,
        dist_weekly_high=10.0,
        dist_weekly_low=10.0,
        nearest_draw_side="NONE",
        atr=1.0,
        atr_vs_session_baseline=1.0,
        realized_vol=1.0,
        intraday_range_vs_typical=1.0,
        trend_slope_htf=0.0,
        trend_dir_htf="FLAT",
        trend_slope_ltf=0.0,
        trend_dir_ltf="FLAT",
        displacement_score=0.0,
        num_impulsive_bars=0,
        swept_pdh=False,
        swept_pdl=False,
        swept_session_high=False,
        swept_session_low=False,
        swept_equal_highs=False,
        swept_equal_lows=False,
        fvg_created=False,
        fvg_filled=False,
        fvg_respected=False,
        ob_created=False,
        ob_respected=False,
        ob_violated=False,
        volume_spike=False,
        volume_vs_mean=1.0,
    )
    base.update(overrides)
    return MarketProfileFeatures(**base)


def _train_dummy_model(label: str = "ACCUMULATION") -> TrainedMarketProfileModel:
    feats = _features()
    df = []
    for cls in ["ACCUMULATION", "MANIPULATION", "DISTRIBUTION"]:
        row = feats.to_vector()
        row["coarse_label"] = cls
        df.append(row)
    return train_market_profile_model(__import__("pandas").DataFrame(df))


def test_rules_accumulation_range_low_vol():
    f = _features(
        trend_dir_ltf="FLAT",
        atr_vs_session_baseline=0.6,
        displacement_score=0.1,
        num_impulsive_bars=0,
    )
    label = coarse_classify_market_profile(f)
    assert label == "ACCUMULATION"


def test_rules_manipulation_on_sweep_and_displacement():
    f = _features(
        swept_pdh=True,
        displacement_score=0.8,
        num_impulsive_bars=3,
        atr_vs_session_baseline=1.2,
        volume_spike=True,
    )
    label = coarse_classify_market_profile(f)
    assert label == "MANIPULATION"


def test_rules_distribution_on_trend_away_from_range():
    f = _features(
        trend_dir_ltf="UP",
        atr_vs_session_baseline=1.1,
        intraday_range_vs_typical=1.2,
        nearest_draw_side="UP",
        fvg_filled=True,
        ob_respected=True,
    )
    label = coarse_classify_market_profile(f)
    assert label == "DISTRIBUTION"


def test_state_machine_resists_flicker():
    sm = MarketProfileStateMachine({}, min_dwell_bars=2, hysteresis=2)
    f = _features()
    # Alternate low probabilities that don't meet dwell + thresholds
    seq = [
        {"ACCUMULATION": 0.6, "MANIPULATION": 0.2, "DISTRIBUTION": 0.1, "UNKNOWN": 0.1},
        {"ACCUMULATION": 0.4, "MANIPULATION": 0.5, "DISTRIBUTION": 0.1, "UNKNOWN": 0.0},
        {"ACCUMULATION": 0.6, "MANIPULATION": 0.2, "DISTRIBUTION": 0.1, "UNKNOWN": 0.1},
    ]
    states = [sm.step(p, f)["state"] for p in seq]
    assert states[-1] == states[0]


def test_state_machine_respects_transition_constraints():
    sm = MarketProfileStateMachine(
        {"prob_manipulation": 0.4, "prob_distribution": 0.4},
        min_dwell_bars=1,
        hysteresis=1,
    )
    f_acc = _features(displacement_score=0.2)
    f_man = _features(swept_pdh=True, displacement_score=0.8, trend_dir_ltf="UP")
    f_dist = _features(
        trend_dir_ltf="UP", atr_vs_session_baseline=1.2, displacement_score=0.3
    )

    state1 = sm.step(
        {"ACCUMULATION": 0.7, "MANIPULATION": 0.2, "DISTRIBUTION": 0.1, "UNKNOWN": 0.0},
        f_acc,
    )
    state2 = sm.step(
        {"ACCUMULATION": 0.3, "MANIPULATION": 0.6, "DISTRIBUTION": 0.1, "UNKNOWN": 0.0},
        f_man,
    )
    state3 = sm.step(
        {"ACCUMULATION": 0.1, "MANIPULATION": 0.3, "DISTRIBUTION": 0.6, "UNKNOWN": 0.0},
        f_dist,
    )

    assert state1["state"] == "ACCUMULATION"
    assert state2["state"] == "MANIPULATION"
    assert state3["state"] == "DISTRIBUTION"


def test_classify_market_profile_returns_evidence():
    ml_model = _train_dummy_model()
    sm = MarketProfileStateMachine({}, min_dwell_bars=1, hysteresis=1)
    f = _features(
        swept_pdh=True,
        displacement_score=0.8,
        num_impulsive_bars=3,
        atr_vs_session_baseline=1.2,
        volume_spike=True,
    )
    result = classify_market_profile(f, ml_model, sm)
    assert result["state"] in {
        "ACCUMULATION",
        "MANIPULATION",
        "DISTRIBUTION",
        "TRANSITION",
    }
    assert "coarse_label" in result
    assert "probs" in result
    assert "evidence" in result
