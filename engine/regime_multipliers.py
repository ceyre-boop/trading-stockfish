"""Regime-conditioned multiplier computation.

Consumes stats outputs and emits deterministic multipliers that can be
embedded directly into policy_config.json under ``regime_multipliers``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class StatsResult:
    """Minimal stats carrier for multiplier computation."""

    feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_stability: Dict[str, float] = field(default_factory=dict)
    feature_performance_by_regime: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiplierConfig:
    """Bounds and scaling parameters for multipliers."""

    min_multiplier: float = 0.5
    max_multiplier: float = 2.0
    default_multiplier: float = 1.0
    positive_scale: float = 0.5  # scale of uplift for positive performance
    negative_scale: float = 0.5  # scale of downweight for negative performance
    smoothing: float = 0.2  # 0=no smoothing; 1=fully default


def _clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def compute_regime_multipliers(
    stats: StatsResult, config: MultiplierConfig | dict | None = None
) -> Dict[str, Dict[str, float]]:
    """Compute per-regime multipliers from stats.

    Args:
        stats: StatsResult with feature_performance_by_regime populated.
        config: MultiplierConfig bounds and scaling.

    Returns:
        Dict mapping regime -> {feature -> multiplier} suitable for policy_config.
    """

    if isinstance(config, dict):
        cfg = MultiplierConfig(**config)
    elif isinstance(config, MultiplierConfig):
        cfg = config
    else:
        cfg = MultiplierConfig()
    regime_multipliers: Dict[str, Dict[str, float]] = {}

    perf_by_regime = stats.feature_performance_by_regime or {}
    for feature, regime_perf in perf_by_regime.items():
        if not isinstance(regime_perf, dict):
            continue
        for regime, perf in regime_perf.items():
            try:
                pval = float(perf)
            except Exception:
                continue
            default = cfg.default_multiplier
            # Positive performance uplifts; negative performance dampens.
            if pval > 0:
                raw = default * (1.0 + cfg.positive_scale * min(abs(pval), 1.0))
            elif pval < 0:
                raw = default * (1.0 - cfg.negative_scale * min(abs(pval), 1.0))
            else:
                raw = default
            # Apply optional smoothing toward default to avoid overreaction.
            # If the raw value already hits a bound, preserve the cap/floor.
            raw_clamped = _clamp(raw, cfg.min_multiplier, cfg.max_multiplier)
            if (
                raw >= cfg.max_multiplier
                or raw <= cfg.min_multiplier
                or cfg.smoothing <= 0
            ):
                mult = raw_clamped
            else:
                mult = default + (raw_clamped - default) * max(
                    0.0, min(1.0, 1.0 - cfg.smoothing)
                )
            regime_multipliers.setdefault(str(regime), {})[str(feature)] = mult

    # Ensure deterministic ordering by sorting keys when serialized by caller
    return regime_multipliers
