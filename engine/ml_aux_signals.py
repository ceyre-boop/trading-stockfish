"""
Deterministic ML-inspired auxiliary signals (advisory only).

These helpers provide bounded, normalized hints for regime detection,
volatility clustering, and anomaly detection without impacting the core
Stockfish-style deterministic evaluator. All computations are pure functions
with no randomness or external services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_std(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = _safe_mean(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance**0.5


@dataclass(frozen=True)
class MLAuxConfig:
    vol_cluster_window: int = 20
    anomaly_zscore_limit: float = 3.0
    hint_cap: float = 1.0


def compute_volatility_cluster_hint(returns: List[float], window: int) -> float:
    """Return a bounded hint for volatility clustering based on rolling std ratio."""
    if not returns or len(returns) < window:
        return 0.0
    recent = returns[-window:]
    older = returns[:-window] or returns[-window:]
    recent_std = _safe_std(recent)
    base_std = _safe_std(older)
    if base_std == 0:
        return 0.0
    ratio = recent_std / base_std
    # Map ratio to [-1, 1] where >1 implies higher clustering
    normalized = _clamp((ratio - 1.0), -1.0, 1.0)
    return normalized


def compute_anomaly_hint(series: List[float], z_limit: float) -> float:
    """Detect simple z-score anomalies; returns bounded advisory score."""
    if not series:
        return 0.0
    mean = _safe_mean(series)
    std = _safe_std(series)
    if std == 0:
        return 0.0
    latest = series[-1]
    z_score = abs(latest - mean) / std
    bounded = _clamp(z_score / z_limit, 0.0, 1.0)
    # Higher anomaly increases warning; encode as positive advisory magnitude
    return bounded


def compute_ml_hints(
    state: Dict, config: MLAuxConfig | None = None
) -> Dict[str, float]:
    """Compute deterministic, bounded ML-style hints (advisory only)."""
    cfg = config or MLAuxConfig()
    returns = state.get("recent_returns", []) or []
    volatility_state = state.get("volatility_state", {}) or {}

    vol_cluster_hint = compute_volatility_cluster_hint(
        returns, window=cfg.vol_cluster_window
    )
    anomaly_hint = compute_anomaly_hint(returns, z_limit=cfg.anomaly_zscore_limit)

    # Macro hint using realized_vol relative to a neutral band
    realized_vol = float(volatility_state.get("realized_vol", 0.0) or 0.0)
    neutral_band = float(volatility_state.get("neutral_band", 1.0) or 1.0)
    if neutral_band == 0:
        neutral_band = 1.0
    macro_vol_hint = _clamp(
        (realized_vol / neutral_band) - 1.0, -cfg.hint_cap, cfg.hint_cap
    )

    return {
        "vol_cluster_hint": _clamp(vol_cluster_hint, -cfg.hint_cap, cfg.hint_cap),
        "anomaly_hint": _clamp(anomaly_hint, 0.0, cfg.hint_cap),
        "macro_vol_hint": _clamp(macro_vol_hint, -cfg.hint_cap, cfg.hint_cap),
    }
