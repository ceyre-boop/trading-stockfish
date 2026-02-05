from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class StrategyDefinition:
    id: str
    name: str
    family: str
    required_features: List[str]
    entry_model_id: str
    exit_model_id: str
    allowed_regimes: List[str]
    allowed_sessions: List[str]
    allowed_volatility: List[str]


_STRATEGY_REGISTRY: List[StrategyDefinition] = [
    StrategyDefinition(
        id="trend_ma_crossover",
        name="Trend MA Crossover",
        family="TREND",
        required_features=["sma_fast", "sma_slow", "trend_direction"],
        entry_model_id="breakout",
        exit_model_id="regime_flip",
        allowed_regimes=["UP", "DOWN"],
        allowed_sessions=["RTH_OPEN", "MIDDAY", "POWER_HOUR"],
        allowed_volatility=["LOW", "NORMAL", "HIGH"],
    ),
    StrategyDefinition(
        id="momentum_rsi_breakout",
        name="Momentum RSI Breakout",
        family="MOMENTUM",
        required_features=["rsi", "roc_20", "momentum"],
        entry_model_id="breakout",
        exit_model_id="time_based",
        allowed_regimes=["MOMENTUM_UP", "MOMENTUM_DOWN"],
        allowed_sessions=["RTH_OPEN", "NEW_YORK", "CLOSE"],
        allowed_volatility=["NORMAL", "HIGH"],
    ),
    StrategyDefinition(
        id="mean_rev_bollinger_fade",
        name="Mean Reversion Bollinger Fade",
        family="MEAN_REVERSION",
        required_features=["z_score", "bollinger_upper", "bollinger_lower"],
        entry_model_id="fade",
        exit_model_id="time_based",
        allowed_regimes=["RANGE_BOUND"],
        allowed_sessions=["GLOBEX", "MIDDAY"],
        allowed_volatility=["LOW", "NORMAL"],
    ),
    StrategyDefinition(
        id="vol_atr_expansion_breakout",
        name="Volatility ATR Expansion Breakout",
        family="VOLATILITY",
        required_features=["atr", "range", "volatility_shock"],
        entry_model_id="breakout",
        exit_model_id="trailing",
        allowed_regimes=["VOLATILITY_EXPANDING"],
        allowed_sessions=["RTH_OPEN", "POWER_HOUR"],
        allowed_volatility=["HIGH", "EXTREME"],
    ),
    StrategyDefinition(
        id="structure_vwap_reversion",
        name="Structure VWAP Reversion",
        family="STRUCTURE",
        required_features=["vwap_price", "distance_from_vwap", "session_regime"],
        entry_model_id="fade",
        exit_model_id="stop_target",
        allowed_regimes=["SESSION_RANGE", "OPEN_DRIVE"],
        allowed_sessions=["RTH_OPEN", "CLOSE"],
        allowed_volatility=["LOW", "NORMAL"],
    ),
]


def get_strategy(strategy_id: str) -> Optional[StrategyDefinition]:
    """Return a strategy definition by id, or None if not found."""

    return next((s for s in _STRATEGY_REGISTRY if s.id == strategy_id), None)


def list_strategies() -> List[StrategyDefinition]:
    """Return all registered strategies (static, deterministic)."""

    return list(_STRATEGY_REGISTRY)
