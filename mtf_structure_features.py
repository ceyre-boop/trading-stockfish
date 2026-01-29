"""
Deterministic multi-timeframe structure engine v1.

Provides higher-timeframe (HTF) structure snapshots (swings, BOS/CHOCH,
impulse/correction, trend direction/strength) and HTF/LTF alignment plus a
fractal compression/expansion signal. No lookahead: only completed HTF bars
are used.
"""

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from structure_features import (
    LEG_CORRECTION,
    LEG_IMPULSE,
    SWING_HIGH,
    SWING_LOW,
    compute_structure_features,
    detect_swings,
)

TIMEFRAME_SECONDS: Dict[str, int] = {
    "1H": 60 * 60,
    "4H": 4 * 60 * 60,
    "D": 24 * 60 * 60,
    "W": 7 * 24 * 60 * 60,
}


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def get_htf_view(candles: Iterable[Tuple[float, float]], timeframe: str) -> List[float]:
    tf = timeframe.upper()
    if tf not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    bucket_seconds = TIMEFRAME_SECONDS[tf]
    buckets: List[Dict[str, Any]] = []
    current_bucket_id = None
    for ts, price in sorted(candles, key=lambda x: x[0]):
        bucket_id = int(ts // bucket_seconds)
        if current_bucket_id is None:
            current_bucket_id = bucket_id
            buckets.append(
                {
                    "high": price,
                    "low": price,
                    "close": price,
                    "id": bucket_id,
                    "count": 1,
                }
            )
            continue
        if bucket_id != current_bucket_id:
            buckets.append(
                {
                    "high": price,
                    "low": price,
                    "close": price,
                    "id": bucket_id,
                    "count": 1,
                }
            )
            current_bucket_id = bucket_id
        else:
            bucket = buckets[-1]
            bucket["high"] = max(bucket["high"], price)
            bucket["low"] = min(bucket["low"], price)
            bucket["close"] = price
            bucket["count"] = bucket.get("count", 0) + 1
    # Exclude the last bucket only if it has a single print (likely incomplete)
    completed = buckets[:-1]
    if buckets and buckets[-1].get("count", 0) > 1:
        completed.append(buckets[-1])
    return [b["close"] for b in completed]


def _htf_snapshot(series: List[float]) -> Dict[str, Any]:
    if not series:
        return {
            "trend_direction": "RANGE",
            "trend_strength": 0.0,
            "last_bos_direction": "NONE",
            "last_choch_direction": "NONE",
            "current_leg_type": LEG_CORRECTION,
            "last_swing_high": 0.0,
            "last_swing_low": 0.0,
        }
    structure = compute_structure_features(series)
    swings = detect_swings(series)
    last_high = max(
        (series[i] for i, t in enumerate(swings) if t == SWING_HIGH), default=0.0
    )
    last_low = max(
        (series[i] for i, t in enumerate(swings) if t == SWING_LOW), default=0.0
    )
    first_price = series[0]
    last_price = series[-1]
    change = last_price - first_price
    pct_change = change / (abs(first_price) + 1e-6)
    if pct_change > 0.001:
        direction = "UP"
    elif pct_change < -0.001:
        direction = "DOWN"
    else:
        direction = "RANGE"
    strength = min(100.0, abs(pct_change) * 10000.0)
    leg_type = structure.get("current_leg_type", LEG_CORRECTION)
    if direction != "RANGE" and strength > 5.0:
        leg_type = LEG_IMPULSE
    return {
        "trend_direction": direction,
        "trend_strength": strength,
        "last_bos_direction": structure.get("last_bos_direction", "NONE"),
        "last_choch_direction": structure.get("last_choch_direction", "NONE"),
        "current_leg_type": leg_type,
        "last_swing_high": float(last_high),
        "last_swing_low": float(last_low),
    }


def _fractal_state(htf_strength: float, ltf_volatility: float) -> Tuple[str, float]:
    htf_norm = max(0.0, min(1.0, htf_strength / 100.0))
    vol_norm = max(0.0, min(1.0, ltf_volatility))
    score = round(htf_norm - vol_norm, 4)
    if htf_norm > 0.6 and vol_norm < 0.3:
        state = "COMPRESSED"
    elif vol_norm > 0.6 and htf_norm > 0.4:
        state = "EXPANDING"
    else:
        state = "NEUTRAL"
    return state, score


def _direction_to_sign(direction: str) -> float:
    if direction.upper() == "UP" or direction.upper() == "BULLISH":
        return 1.0
    if direction.upper() == "DOWN" or direction.upper() == "BEARISH":
        return -1.0
    return 0.0


def _alignment_score(
    htf_dir: str, htf_strength: float, ltf_dir: str, ltf_strength: float
) -> float:
    htf_sign = _direction_to_sign(htf_dir)
    ltf_sign = _direction_to_sign(ltf_dir)
    score = 0.4 * htf_sign * (htf_strength / 100.0) + 0.6 * ltf_sign * max(
        0.0, min(1.0, ltf_strength / 100.0)
    )
    return max(-1.0, min(1.0, score))


def _bias(htf_dir: str, htf_strength: float) -> str:
    if htf_dir == "UP" and htf_strength > 30.0:
        return "BULLISH"
    if htf_dir == "DOWN" and htf_strength > 30.0:
        return "BEARISH"
    return "NEUTRAL"


def compute_mtf_structure_features(
    prices: Sequence[float],
    timestamps: Sequence[float],
    timeframes: Sequence[str] = ("1H", "4H", "D"),
    ltf_trend_direction: str = "RANGE",
    ltf_trend_strength: float = 0.0,
    ltf_volatility: float = 0.0,
) -> Dict[str, Any]:
    candles = list(zip(timestamps, prices))
    result: Dict[str, Any] = {}
    htf_primary_direction = "RANGE"
    htf_primary_strength = 0.0

    for tf in timeframes:
        tf_series = get_htf_view(candles, tf)
        snapshot = _htf_snapshot(tf_series)
        key = tf.lower()
        result[f"htf_{key}_trend_direction"] = snapshot["trend_direction"]
        result[f"htf_{key}_trend_strength"] = snapshot["trend_strength"]
        result[f"htf_{key}_last_bos_direction"] = snapshot["last_bos_direction"]
        result[f"htf_{key}_last_choch_direction"] = snapshot["last_choch_direction"]
        result[f"htf_{key}_current_leg_type"] = snapshot["current_leg_type"]
        result[f"htf_{key}_last_swing_high"] = snapshot["last_swing_high"]
        result[f"htf_{key}_last_swing_low"] = snapshot["last_swing_low"]
        if (
            tf in ("D", "4H", "1H")
            and snapshot["trend_strength"] >= htf_primary_strength
        ):
            htf_primary_direction = snapshot["trend_direction"]
            htf_primary_strength = snapshot["trend_strength"]

    fractal_state, fractal_score = _fractal_state(htf_primary_strength, ltf_volatility)
    alignment = _alignment_score(
        htf_primary_direction,
        htf_primary_strength,
        ltf_trend_direction,
        ltf_trend_strength,
    )
    bias = _bias(htf_primary_direction, htf_primary_strength)

    result.update(
        {
            "fractal_state": fractal_state,
            "fractal_score": fractal_score,
            "htf_ltf_alignment_score": alignment,
            "htf_bias": bias,
        }
    )
    return result
