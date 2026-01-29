"""
Deterministic regime engine for Trading Stockfish v4.0‑D
"""

from collections import deque
from typing import Dict, Optional

from session_regime import SessionRegime

from .ml_aux_signals import compute_ml_hints
from .regime_helpers import (
    classify_liquidity,
    classify_macro,
    classify_trend,
    classify_volatility,
    liquidity_penalty,
    macro_bias,
    trend_strength,
    volatility_intensity,
)
from .trend_structure import compute_trend_structure
from .types import MarketState


class RegimeEngine:
    def __init__(self, window: int = 10):
        self.window = window
        self.prev_regimes = deque(maxlen=window)

    def compute(
        self,
        volatility_state: Dict,
        liquidity_state: Dict,
        macro_news_state: Dict,
        ml_state: Optional[Dict] = None,
        amd_state: Optional[Dict] = None,
        price_series: Optional[list] = None,
        high_series: Optional[list] = None,
        low_series: Optional[list] = None,
        session_regime: str = "UNKNOWN",
    ) -> Dict:
        # Vol regime
        vol_regime = volatility_state.get("vol_regime", "NORMAL")
        vol_shock = bool(volatility_state.get("volatility_shock", False))
        vol_shock_strength = float(
            volatility_state.get("volatility_shock_strength", 0.0)
        )
        trend_struct = compute_trend_structure(
            price_series or [],
            highs=high_series,
            lows=low_series,
            window=20,
            volatility_state=volatility_state,
        )
        # Liquidity regime
        liq_res = liquidity_state.get("liquidity_resilience", 0.0)
        depth_imb = liquidity_state.get("depth_imbalance", 0.0)
        if liq_res > 0.2 and abs(depth_imb) < 0.2:
            liq_regime = "DEEP"
        elif liq_res > 0.05:
            liq_regime = "NORMAL"
        elif liq_res > -0.1:
            liq_regime = "THIN"
        else:
            liq_regime = "FRAGILE"
        # Macro regime
        hawk = macro_news_state.get("hawkishness", 0.0)
        risk = macro_news_state.get("risk_sentiment", 0.0)
        if risk > 0.5:
            macro_regime = "RISK_ON"
        elif risk < -0.5:
            macro_regime = "RISK_OFF"
        else:
            macro_regime = "EVENT"
        # Confidence
        regime_confidence = min(
            1.0,
            max(
                0.0,
                0.5 * abs(liq_res)
                + 0.5 * abs(volatility_state.get("realized_vol", 0.0)),
            ),
        )

        amd_tag = "NEUTRAL"
        amd_conf = 0.0
        if amd_state:
            amd_tag = amd_state.get("amd_tag", "NEUTRAL")
            amd_conf = float(amd_state.get("amd_confidence", 0.0))

        # Optional ML hints (advisory, bounded, deterministic). Recorded only.
        ml_hints = compute_ml_hints(ml_state or {}) if ml_state is not None else {}
        # Transition
        current_tag = (vol_regime, liq_regime, macro_regime)
        regime_transition = False
        if self.prev_regimes and self.prev_regimes[-1] != current_tag:
            regime_transition = True
        self.prev_regimes.append(current_tag)
        return {
            "vol_regime": vol_regime,
            "volatility_shock": vol_shock,
            "volatility_shock_strength": vol_shock_strength,
            "liq_regime": liq_regime,
            "macro_regime": macro_regime,
            "regime_confidence": regime_confidence,
            "regime_transition": regime_transition,
            "ml_hints": ml_hints,
            "amd_regime": amd_tag,
            "amd_confidence": amd_conf,
            "vol_shock_tag": "VOL_SHOCK" if vol_shock else "NORMAL",
            "trend_regime": _trend_regime_from_structure(
                trend_struct["swing_structure"]
            ),
            "swing_structure": trend_struct["swing_structure"],
            "trend_direction": trend_struct["trend_direction"],
            "trend_strength": trend_struct["trend_strength"],
            "swing_high": trend_struct["swing_high"],
            "swing_low": trend_struct["swing_low"],
            "session_regime": session_regime,
        }


# ---------------------------------------------------------------------------
# Deterministic regime bundle (Phase 3 expanded)
# ---------------------------------------------------------------------------


def _stddev(values):
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance**0.5


def compute_regime_bundle(state: MarketState) -> Dict[str, float | str]:
    """Compute regime labels and modifiers from MarketState."""

    trend_reg = classify_trend(state.ma_short, state.ma_long)
    vol_reg = classify_volatility(state.volatility, state.recent_returns)
    liq_reg = classify_liquidity(state.liquidity)
    mac_reg = classify_macro(state.momentum, state.rsi)
    amd_reg = getattr(state, "amd_regime", "NEUTRAL")
    vol_shock = getattr(state, "volatility_shock", False)
    swing_struct = getattr(state, "swing_structure", "NEUTRAL")
    trend_dir = getattr(state, "trend_direction", "RANGE")
    trend_str = getattr(state, "trend_strength", 0.0)
    session_reg = getattr(state, "session", "UNKNOWN")

    stddev = _stddev(state.recent_returns)

    swing_trend_regime = _trend_regime_from_structure(swing_struct)
    trend_regime = swing_trend_regime if swing_struct != "NEUTRAL" else trend_reg
    trend_strength_ma = trend_strength(state.ma_short, state.ma_long)
    combined_trend_strength = max(trend_str, trend_strength_ma)

    return {
        "trend_regime": trend_regime,
        "volatility_regime": vol_reg,
        "liquidity_regime": liq_reg,
        "macro_regime": mac_reg,
        "amd_regime": amd_reg,
        "volatility_shock": vol_shock,
        "swing_structure": swing_struct,
        "trend_direction": trend_dir,
        "trend_strength": combined_trend_strength,
        "volatility_intensity": volatility_intensity(state.volatility, stddev),
        "liquidity_penalty": liquidity_penalty(state.liquidity),
        "macro_bias": macro_bias(state.momentum, state.rsi),
        "session_regime": session_reg,
    }


def _trend_regime_from_structure(structure: str) -> str:
    if structure in {"HH", "HL"}:
        return "UPTREND"
    if structure in {"LH", "LL"}:
        return "DOWNTREND"
    return "RANGE"
