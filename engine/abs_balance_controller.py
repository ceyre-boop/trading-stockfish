"""
ABS break-level balance controller for deterministic feature normalization.

This module enforces absolute break levels and weight caps so that no single
factor dominates the evaluator. It preserves Stockfish-style determinism by
only using pure functions, fixed defaults, and clamping.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class BalanceConfig:
    """Configuration for ABS break-level balancing.

    Attributes:
        max_feature_weight: Global hard cap for any individual weight.
        regime_weight_caps: Optional per-regime caps applied after global cap.
        volatility_sensitivity_limits: Caps applied when volatility regimes are high.
        break_levels: Absolute break levels used to normalize raw feature magnitudes.
    """

    max_feature_weight: float = 1.5
    regime_weight_caps: Dict[str, float] = field(
        default_factory=lambda: {
            "LOW": 1.2,
            "NORMAL": 1.5,
            "HIGH": 1.0,
            "EXTREME": 0.8,
        }
    )
    volatility_sensitivity_limits: Dict[str, float] = field(
        default_factory=lambda: {
            "LOW": 1.0,
            "NORMAL": 1.0,
            "HIGH": 0.8,
            "EXTREME": 0.6,
        }
    )
    break_levels: Dict[str, float] = field(
        default_factory=lambda: {
            "trend": 1.0,
            "order_flow": 1.0,
            "liquidity": 1.0,
            "volatility": 1.0,
            "macro": 1.0,
            "long_bias": 1.0,
        }
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ABSBalanceController:
    """Deterministic ABS break-level balancing for evaluator features."""

    def __init__(self, config: BalanceConfig | None = None):
        self.config = config or BalanceConfig()

    def normalize_feature(self, name: str, value: float) -> float:
        """Normalize a raw feature value against its absolute break level."""
        level = abs(self.config.break_levels.get(name, 1.0)) or 1.0
        normalized = value / level
        return _clamp(
            normalized, -self.config.max_feature_weight, self.config.max_feature_weight
        )

    def balance_weights(
        self,
        features: Dict[str, float],
        regime_state: Dict[str, str] | None = None,
        volatility_state: Dict[str, str] | None = None,
    ) -> Dict[str, float]:
        """Apply ABS break-level balancing and deterministic caps.

        Steps (all deterministic):
        1. Normalize by absolute break levels.
        2. Apply global max_feature_weight clamp.
        3. Apply regime-specific cap.
        4. Apply volatility sensitivity limits.
        """
        regime_state = regime_state or {}
        volatility_state = volatility_state or {}
        vol_regime = volatility_state.get("vol_regime", "NORMAL")
        balanced: Dict[str, float] = {}

        for name, raw_value in features.items():
            normalized = self.normalize_feature(name, float(raw_value))
            capped = _clamp(
                normalized,
                -self.config.max_feature_weight,
                self.config.max_feature_weight,
            )

            # Regime cap (advisory clamp)
            regime_cap = self.config.regime_weight_caps.get(
                vol_regime, self.config.max_feature_weight
            )
            capped = _clamp(capped, -regime_cap, regime_cap)

            # Volatility sensitivity guard
            vol_cap = self.config.volatility_sensitivity_limits.get(vol_regime, 1.0)
            capped = _clamp(capped, -vol_cap, vol_cap)

            balanced[name] = capped

        return balanced

    def apply_confidence_guard(
        self, confidence: float, balanced_weights: Dict[str, float]
    ) -> Tuple[float, float]:
        """Deterministically downscale confidence when any feature is near its cap."""
        if not balanced_weights:
            return confidence, 1.0
        max_abs = max(abs(v) for v in balanced_weights.values())
        if max_abs <= 1.0:
            return confidence, 1.0
        # Scale confidence inversely to the amount of capping to avoid dominance
        scale = _clamp(1.0 / max_abs, 0.5, 1.0)
        return confidence * scale, scale


# Singleton-style controller for module-wide reuse
BALANCE_CONTROLLER = ABSBalanceController()
