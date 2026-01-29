"""
Deterministic momentum indicators v2.
Implements RSI (Wilder), MACD, Stochastic, momentum regime, and simple divergence.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

_DEF_EPS = 1e-9


@dataclass(frozen=True)
class MACDSeries:
    macd_line: List[float]
    signal_line: List[float]
    histogram: List[float]


@dataclass(frozen=True)
class StochSeries:
    k: List[float]
    d: List[float]


def _ema(values: Sequence[float], period: int) -> List[float]:
    if period <= 0:
        return [0.0 for _ in values]
    if not values:
        return []
    alpha = 2.0 / (period + 1)
    ema_vals: List[float] = [values[0]]
    for v in values[1:]:
        ema_vals.append(alpha * v + (1 - alpha) * ema_vals[-1])
    return ema_vals


def compute_rsi(prices: Sequence[float], period: int = 14) -> List[float]:
    if period <= 0 or len(prices) < 2:
        return [0.0 for _ in prices]
    gains: List[float] = [0.0]
    losses: List[float] = [0.0]
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))
    # Wilder smoothing
    rsis: List[float] = []
    avg_gain = sum(gains[1 : period + 1]) / period if len(gains) > period else 0.0
    avg_loss = sum(losses[1 : period + 1]) / period if len(losses) > period else 0.0
    if avg_loss == 0:
        rs = float("inf") if avg_gain > 0 else 0.0
    else:
        rs = avg_gain / (avg_loss + _DEF_EPS)
    rsi = 100 - (100 / (1 + rs)) if rs != float("inf") else 100.0
    rsis = [0.0 for _ in range(len(prices))]
    start = min(period, len(prices))
    if start < len(prices):
        rsis[start] = rsi
    for i in range(start + 1, len(prices)):
        gain = gains[i]
        loss = losses[i]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = avg_gain / (avg_loss + _DEF_EPS) if avg_loss > 0 else float("inf")
        rsi = 100 - (100 / (1 + rs)) if rs != float("inf") else 100.0
        rsis[i] = rsi
    # Fill leading zeros with first computed RSI for stability
    first_non_zero = next((v for v in rsis if v != 0.0), 0.0)
    rsis = [first_non_zero if v == 0.0 else v for v in rsis]
    return rsis


def compute_macd(
    prices: Sequence[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> MACDSeries:
    if not prices:
        return MACDSeries([], [], [])
    fast_ema = _ema(prices, fast_period)
    slow_ema = _ema(prices, slow_period)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_line = _ema(macd_line, signal_period)
    # Pad signal_line to match length
    if signal_line:
        pad = len(macd_line) - len(signal_line)
        if pad > 0:
            signal_line = [signal_line[0]] * pad + signal_line
    histogram = [m - s for m, s in zip(macd_line, signal_line)]
    return MACDSeries(macd_line=macd_line, signal_line=signal_line, histogram=histogram)


def compute_stochastic(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> StochSeries:
    if (
        not highs
        or not lows
        or not closes
        or len(closes) != len(highs)
        or len(lows) != len(highs)
    ):
        return StochSeries([], [])
    k_vals: List[float] = []
    for i in range(len(closes)):
        start = max(0, i - k_period + 1)
        window_high = max(highs[start : i + 1])
        window_low = min(lows[start : i + 1])
        denom = window_high - window_low
        if denom == 0:
            k = 50.0
        else:
            k = ((closes[i] - window_low) / (denom + _DEF_EPS)) * 100
        k_vals.append(k)
    # Smooth %K
    smooth_vals: List[float] = []
    for i in range(len(k_vals)):
        start = max(0, i - smooth_k + 1)
        smooth_vals.append(sum(k_vals[start : i + 1]) / (i - start + 1))
    d_vals: List[float] = []
    for i in range(len(smooth_vals)):
        start = max(0, i - d_period + 1)
        d_vals.append(sum(smooth_vals[start : i + 1]) / (i - start + 1))
    return StochSeries(k=smooth_vals, d=d_vals)


def _state_from_rsi(value: float) -> str:
    if value >= 70.0:
        return "OVERBOUGHT"
    if value <= 30.0:
        return "OVERSOLD"
    return "NEUTRAL"


def _state_from_stoch(k: float, d: float) -> str:
    if k >= 80.0 and d >= 80.0:
        return "OVERBOUGHT"
    if k <= 20.0 and d <= 20.0:
        return "OVERSOLD"
    return "NEUTRAL"


def _detect_divergence(
    prices: Sequence[float], indicator: Sequence[float], lookback: int = 10
) -> Tuple[bool, bool]:
    if len(prices) < 3 or len(indicator) < 3:
        return False, False
    prices_slice = list(prices[-lookback:])
    indicator_slice = list(indicator[-lookback:])
    if len(prices_slice) != len(indicator_slice):
        m = min(len(prices_slice), len(indicator_slice))
        prices_slice = prices_slice[-m:]
        indicator_slice = indicator_slice[-m:]

    def _two_extrema(vals: List[float], find_high: bool) -> List[int]:
        if not vals:
            return []
        enumerated = list(enumerate(vals))
        sorted_by_val = sorted(
            enumerated, key=lambda x: (-x[1] if find_high else x[1], x[0])
        )
        idxs = (
            sorted([sorted_by_val[0][0], sorted_by_val[1][0]])
            if len(sorted_by_val) > 1
            else [sorted_by_val[0][0]]
        )
        return idxs[-2:] if len(idxs) >= 2 else idxs

    low_idxs = _two_extrema(prices_slice, find_high=False)
    high_idxs = _two_extrema(prices_slice, find_high=True)
    bullish = False
    bearish = False
    if len(low_idxs) >= 2:
        p1, p2 = low_idxs[-2], low_idxs[-1]
        if (
            prices_slice[p2] < prices_slice[p1]
            and indicator_slice[p2] > indicator_slice[p1]
        ):
            bullish = True
    if len(high_idxs) >= 2:
        h1, h2 = high_idxs[-2], high_idxs[-1]
        if (
            prices_slice[h2] > prices_slice[h1]
            and indicator_slice[h2] < indicator_slice[h1]
        ):
            bearish = True
    return bullish, bearish


def compute_momentum_indicators(
    prices: Sequence[float],
    highs: Sequence[float] | None = None,
    lows: Sequence[float] | None = None,
    volatility: float | None = None,
    trend_strength: float | None = None,
) -> Dict[str, object]:
    if highs is None:
        highs = prices
    if lows is None:
        lows = prices

    # Replay/live parity guard: drop tiny trailing micro move
    if len(prices) >= 2:
        delta = abs(prices[-1] - prices[-2])
        threshold = max(abs(prices[-2]) * 0.001, 1e-4)
        if delta <= threshold:
            prices = prices[:-1]
            highs = highs[:-1]
            lows = lows[:-1]

    rsi_series = compute_rsi(prices)
    rsi_value = rsi_series[-1] if rsi_series else 50.0
    rsi_state = _state_from_rsi(rsi_value)

    macd = compute_macd(prices)
    macd_value = macd.macd_line[-1] if macd.macd_line else 0.0
    macd_signal = macd.signal_line[-1] if macd.signal_line else 0.0
    macd_hist = macd.histogram[-1] if macd.histogram else 0.0
    if macd_value > macd_signal and macd_hist > 0:
        macd_state = "BULLISH"
    elif macd_value < macd_signal and macd_hist < 0:
        macd_state = "BEARISH"
    else:
        macd_state = "NEUTRAL"

    stoch = compute_stochastic(highs, lows, prices)
    stoch_k = stoch.k[-1] if stoch.k else 50.0
    stoch_d = stoch.d[-1] if stoch.d else 50.0
    stoch_state = _state_from_stoch(stoch_k, stoch_d)

    # Divergence detection using RSI and MACD line
    rsi_bull, rsi_bear = _detect_divergence(prices, rsi_series)
    macd_bull, macd_bear = _detect_divergence(prices, macd.macd_line)

    # Momentum regime and confidence
    vol_norm = max(0.0, min(1.0, (volatility or 0.0)))
    trend_norm = max(0.0, min(1.0, (abs(trend_strength or 0.0) / 100.0)))
    hist_strength = min(
        1.0,
        (
            abs(macd_hist) / (abs(macd_value) + abs(macd_signal) + 1e-3)
            if (macd_value or macd_signal)
            else abs(macd_hist)
        ),
    )
    rsi_bias = abs(rsi_value - 50.0) / 50.0
    momentum_confidence = max(
        0.0, min(1.0, 0.4 * hist_strength + 0.3 * rsi_bias + 0.3 * trend_norm)
    )
    if (
        (
            macd_state != "NEUTRAL"
            and abs(macd_hist) >= 0.001
            and (
                (macd_hist > 0 and rsi_value >= 55)
                or (macd_hist < 0 and rsi_value <= 45)
            )
        )
        or rsi_value >= 60
        or rsi_value <= 40
    ):
        momentum_regime = "STRONG_TREND"
    elif 45 <= rsi_value <= 55 and abs(macd_hist) < 0.2:
        momentum_regime = "CHOP"
    else:
        momentum_regime = "MEAN_REVERT"

    return {
        "rsi_value": rsi_value,
        "rsi_state": rsi_state,
        "macd_value": macd_value,
        "macd_signal": macd_signal,
        "macd_histogram": macd_hist,
        "macd_state": macd_state,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "stoch_state": stoch_state,
        "momentum_regime": momentum_regime,
        "momentum_confidence": momentum_confidence,
        "rsi_bullish_divergence": rsi_bull,
        "rsi_bearish_divergence": rsi_bear,
        "macd_bullish_divergence": macd_bull,
        "macd_bearish_divergence": macd_bear,
    }
