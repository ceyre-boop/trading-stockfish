from dataclasses import dataclass
from typing import Any, Iterable, List


@dataclass(frozen=True)
class PatternProbabilities:
    p_sweep_reversal: float = 0.5
    p_sweep_continuation: float = 0.5
    p_ob_hold: float = 0.5
    p_ob_fail: float = 0.5
    p_fvg_fill: float = 0.5
    expected_move_after_sweep: float = 0.0
    expected_move_after_ob_touch: float = 0.0
    expected_move_after_fvg_fill: float = 0.0


def _get(state: Any, key: str, default: float | bool | str = 0.0):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _laplace(k: float, n: float) -> float:
    return float((k + 1.0) / (n + 2.0))


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, value)))


def compute_pattern_probabilities(
    history_window: Iterable[Any],
    lookahead: int = 1,
    max_window: int = 500,
) -> PatternProbabilities:
    window: List[Any] = list(history_window or [])
    if len(window) <= lookahead:
        return PatternProbabilities()

    window = window[-(max_window + lookahead) :]

    sweep_condition = sweep_reversal = sweep_continuation = 0.0
    ob_condition = ob_hold = ob_fail = 0.0
    fvg_condition = fvg_fill = 0.0

    sweep_moves: List[float] = []
    ob_moves: List[float] = []
    fvg_moves: List[float] = []

    for i in range(len(window) - lookahead):
        cur = window[i]
        nxt = window[i + lookahead]
        price_now = float(_get(cur, "current_price", _get(cur, "mid", 0.0)) or 0.0)
        price_next = float(
            _get(nxt, "current_price", _get(nxt, "mid", price_now)) or price_now
        )

        # Sweep patterns
        sweep_dir = str(_get(cur, "last_sweep_direction", "NONE"))
        swept_bsl = bool(_get(cur, "swept_bsl", False))
        swept_ssl = bool(_get(cur, "swept_ssl", False))
        sweep_triggered = sweep_dir != "NONE" or swept_bsl or swept_ssl
        if sweep_triggered:
            sweep_condition += 1.0
            delta = price_next - price_now
            if sweep_dir in {"UP", "BUY", "BSL"}:
                if delta < 0:
                    sweep_reversal += 1.0
                elif delta > 0:
                    sweep_continuation += 1.0
            elif sweep_dir in {"DOWN", "SELL", "SSL"}:
                if delta > 0:
                    sweep_reversal += 1.0
                elif delta < 0:
                    sweep_continuation += 1.0
            sweep_moves.append(_clamp(delta, -5.0, 5.0))

        # Orderblock respect/fail
        bull_low = float(_get(cur, "current_bullish_ob_low", 0.0) or 0.0)
        bull_high = float(_get(cur, "current_bullish_ob_high", 0.0) or 0.0)
        bear_low = float(_get(cur, "current_bearish_ob_low", 0.0) or 0.0)
        bear_high = float(_get(cur, "current_bearish_ob_high", 0.0) or 0.0)
        last_ob = str(_get(cur, "last_touched_ob_type", "NONE"))
        in_bull = bull_low and bull_high and bull_low <= price_now <= bull_high
        in_bear = bear_low and bear_high and bear_low <= price_now <= bear_high
        if in_bull or in_bear:
            ob_condition += 1.0
            ob_moves.append(_clamp(price_next - price_now, -5.0, 5.0))
            if in_bull:
                if bull_low <= price_next <= bull_high:
                    ob_hold += 1.0
                elif price_next < bull_low:
                    ob_fail += 1.0
            elif in_bear:
                if bear_low <= price_next <= bear_high:
                    ob_hold += 1.0
                elif price_next > bear_high:
                    ob_fail += 1.0

        # FVG fill
        has_fvg = bool(_get(cur, "has_fvg", False))
        fvg_low = float(_get(cur, "fvg_lower", 0.0) or 0.0)
        fvg_high = float(_get(cur, "fvg_upper", 0.0) or 0.0)
        if has_fvg and fvg_low and fvg_high:
            fvg_condition += 1.0
            if fvg_low <= price_next <= fvg_high:
                fvg_fill += 1.0
                fvg_moves.append(_clamp(price_next - price_now, -5.0, 5.0))

    p_sweep_reversal = _laplace(sweep_reversal, sweep_condition)
    p_sweep_continuation = _laplace(sweep_continuation, sweep_condition)
    p_ob_hold = _laplace(ob_hold, ob_condition)
    p_ob_fail = _laplace(ob_fail, ob_condition)
    p_fvg_fill = _laplace(fvg_fill, fvg_condition)

    def _avg(moves: List[float]) -> float:
        if not moves:
            return 0.0
        return float(sum(moves) / len(moves))

    return PatternProbabilities(
        p_sweep_reversal=_clamp(p_sweep_reversal, 0.0, 1.0),
        p_sweep_continuation=_clamp(p_sweep_continuation, 0.0, 1.0),
        p_ob_hold=_clamp(p_ob_hold, 0.0, 1.0),
        p_ob_fail=_clamp(p_ob_fail, 0.0, 1.0),
        p_fvg_fill=_clamp(p_fvg_fill, 0.0, 1.0),
        expected_move_after_sweep=_avg(sweep_moves),
        expected_move_after_ob_touch=_avg(ob_moves),
        expected_move_after_fvg_fill=_avg(fvg_moves),
    )
