import math

from engine.structure_brain import (
    classify_market_profile_simple,
    classify_session_profile,
    compute_liquidity_frame,
)


def test_market_profile_accumulation_vs_manipulation():
    acc = classify_market_profile_simple(
        {
            "volatility_regime": "LOW",
            "trend_strength": 0.1,
            "displacement_events": False,
            "sweep_recent": False,
            "compression": True,
            "expansion": False,
            "volume_anomaly": False,
        }
    )
    assert acc.state == "ACCUMULATION"

    man = classify_market_profile_simple(
        {
            "volatility_regime": "HIGH",
            "trend_strength": 0.2,
            "displacement_events": True,
            "sweep_recent": True,
            "compression": False,
            "expansion": False,
            "volume_anomaly": True,
        }
    )
    assert man.state == "MANIPULATION"
    assert man.confidence >= acc.confidence


def test_session_profile_variants():
    prof_a = classify_session_profile(
        {
            "previous_session_profile": "PROFILE_1A",
            "early_volatility": 0.2,
            "sweep_first_leg": True,
            "displacement_first_leg": True,
            "continuation_after_reversal": False,
            "manipulation_detected": True,
        }
    )
    assert prof_a.profile == "PROFILE_1A"

    prof_c = classify_session_profile(
        {
            "previous_session_profile": "PROFILE_1B",
            "early_volatility": 0.4,
            "sweep_first_leg": True,
            "displacement_first_leg": True,
            "continuation_after_reversal": True,
            "manipulation_detected": True,
        }
    )
    assert prof_c.profile == "PROFILE_1C"
    assert prof_c.confidence >= 0.3


def test_compute_liquidity_frame_selects_nearest_unswept():
    lf = compute_liquidity_frame(
        {
            "PDH": {"distance": 10.0, "swept": False},
            "PDL": {"distance": 5.0, "swept": True},
            "SESSION_LOW": {"distance": 6.0, "swept": False},
        }
    )
    assert lf.primary_target == "SESSION_LOW"
    assert lf.target_side == "DOWN"
    assert lf.distances["PDH"] == 10.0
    assert lf.swept["PDL"] is True

    lf_all_swept = compute_liquidity_frame(
        {
            "PDH": {"distance": 7.0, "swept": True},
            "PDL": {"distance": 3.0, "swept": True},
        }
    )
    assert lf_all_swept.primary_target == "PDL"
    assert lf_all_swept.target_side == "DOWN"
    assert math.isfinite(lf_all_swept.distances["PDL"])
