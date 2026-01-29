"""
Deterministic momentum and rate-of-change (ROC) utilities.
"""

from typing import Dict, Iterable, List


def _safe_prices(prices: Iterable[float]) -> List[float]:
    return [float(p) for p in prices if p is not None]


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def close_to_close_roc(prices: Iterable[float], window: int) -> float:
    series = _safe_prices(prices)
    if len(series) <= window:
        return 0.0
    past = series[-window - 1]
    if past == 0:
        return 0.0
    roc = (series[-1] - past) / abs(past)
    return _clamp(roc)


def rolling_momentum(prices: Iterable[float], window: int) -> float:
    series = _safe_prices(prices)
    if len(series) <= window:
        return 0.0
    window_slice = series[-(window + 1) :]
    gains = [window_slice[i + 1] - window_slice[i] for i in range(window)]
    if not gains:
        return 0.0
    avg_gain = sum(gains) / max(len(gains), 1)
    base = abs(window_slice[0]) or 1.0
    momentum = avg_gain / base
    return _clamp(momentum)


def compute_momentum_features(prices: Iterable[float]) -> Dict[str, float]:
    windows = [5, 10, 20]
    feats: Dict[str, float] = {}
    for w in windows:
        feats[f"roc_{w}"] = close_to_close_roc(prices, w)
        feats[f"momentum_{w}"] = rolling_momentum(prices, w)
    return feats
