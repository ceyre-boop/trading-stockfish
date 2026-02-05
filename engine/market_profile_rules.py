from typing import List

from .market_profile_features import MarketProfileFeatures


def _any_sweep(f: MarketProfileFeatures) -> bool:
    return any(
        [
            f.swept_pdh,
            f.swept_pdl,
            f.swept_session_high,
            f.swept_session_low,
            f.swept_equal_highs,
            f.swept_equal_lows,
        ]
    )


def coarse_classify_market_profile(features: MarketProfileFeatures) -> str:
    """Rule-based coarse A/M/D classifier.

    Returns one of ACCUMULATION / MANIPULATION / DISTRIBUTION / UNKNOWN.
    Priority: MANIPULATION > DISTRIBUTION > ACCUMULATION.
    """

    f = features

    # ACCUMULATION rules
    acc_rules: List[bool] = [
        f.trend_dir_ltf.upper() == "FLAT",
        f.atr_vs_session_baseline < 0.8,
        f.displacement_score < 0.3,
        f.num_impulsive_bars <= 1,
        not _any_sweep(f),
        f.intraday_range_vs_typical < 0.8,
    ]
    is_acc = acc_rules.count(True) >= 4

    # MANIPULATION rules
    man_rules: List[bool] = [
        _any_sweep(f),
        f.displacement_score >= 0.6,
        f.num_impulsive_bars >= 2,
        f.atr_vs_session_baseline > 1.1,
        f.volume_spike,
        f.time_of_day_bucket.upper() in {"OPEN", "KILLZONE"},
    ]
    is_man = man_rules.count(True) >= 3

    # DISTRIBUTION rules
    dist_rules: List[bool] = [
        f.trend_dir_ltf.upper() in {"UP", "DOWN"},
        f.atr_vs_session_baseline >= 1.0,
        f.intraday_range_vs_typical >= 0.9,
        (f.nearest_draw_side.upper() == "UP" and f.trend_dir_ltf.upper() == "UP")
        or (
            f.nearest_draw_side.upper() == "DOWN" and f.trend_dir_ltf.upper() == "DOWN"
        ),
        f.fvg_filled or f.fvg_respected,
        f.ob_respected and not f.ob_violated,
    ]
    is_dist = dist_rules.count(True) >= 3

    if is_man:
        return "MANIPULATION"
    if is_dist:
        return "DISTRIBUTION"
    if is_acc:
        return "ACCUMULATION"
    return "UNKNOWN"
