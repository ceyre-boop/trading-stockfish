from typing import Any, Dict

from pattern_stats import PatternProbabilities


def _get(state: Any, key: str, default: float | bool | str = 0.0):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, value)))


def extract_pattern_probabilities(state: Any) -> PatternProbabilities:
    return PatternProbabilities(
        p_sweep_reversal=float(_get(state, "p_sweep_reversal", 0.5) or 0.5),
        p_sweep_continuation=float(_get(state, "p_sweep_continuation", 0.5) or 0.5),
        p_ob_hold=float(_get(state, "p_ob_hold", 0.5) or 0.5),
        p_ob_fail=float(_get(state, "p_ob_fail", 0.5) or 0.5),
        p_fvg_fill=float(_get(state, "p_fvg_fill", 0.5) or 0.5),
        expected_move_after_sweep=float(
            _get(state, "expected_move_after_sweep", 0.0) or 0.0
        ),
        expected_move_after_ob_touch=float(
            _get(state, "expected_move_after_ob_touch", 0.0) or 0.0
        ),
        expected_move_after_fvg_fill=float(
            _get(state, "expected_move_after_fvg_fill", 0.0) or 0.0
        ),
    )


def compute_probability_tilts(
    state: Any, probs: PatternProbabilities | None = None
) -> Dict[str, float]:
    pattern = probs or extract_pattern_probabilities(state)

    def _sweep_sign(direction: str) -> float:
        if direction in {"UP", "BUY", "BSL"}:
            return 1.0
        if direction in {"DOWN", "SELL", "SSL"}:
            return -1.0
        return 0.0

    sweep_dir = str(_get(state, "last_sweep_direction", "NONE"))
    sweep_sign = _sweep_sign(sweep_dir)
    sweep_diff = pattern.p_sweep_reversal - pattern.p_sweep_continuation
    sweep_bias = -sweep_sign * sweep_diff * 0.4  # reversal is opposite sweep dir

    ob_type = str(_get(state, "last_touched_ob_type", "NONE"))
    ob_sign = 1.0 if ob_type == "BULLISH" else -1.0 if ob_type == "BEARISH" else 0.0
    ob_diff = pattern.p_ob_hold - pattern.p_ob_fail
    ob_bias = ob_sign * ob_diff * 0.3

    fvg_bias = (pattern.p_fvg_fill - 0.5) * 0.2

    total = sweep_bias + ob_bias + fvg_bias
    total = _clamp(total, -0.5, 0.5)

    return {
        "sweep_reversal_bias": sweep_bias,
        "ob_respect_bias": ob_bias,
        "fvg_fill_bias": fvg_bias,
        "total_probability_tilt": total,
    }
