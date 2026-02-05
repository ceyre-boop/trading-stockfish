from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .decision_actions import DecisionAction


@dataclass
class PricePath:
    bar_indices: Optional[List[int]] = None
    prices: Optional[List[float]] = None
    volatility: Optional[List[float]] = None
    liquidity: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCRActionContext:
    decision_frame_ref: Any
    initial_position_state: Dict[str, Any]
    decision_action: DecisionAction
    risk_envelope: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCRRolloutResult:
    realized_R: float
    max_adverse_excursion: float
    max_favorable_excursion: float
    time_in_trade_bars: int
    hit_stop: bool
    hit_tp: bool
    closed_by_rule: bool
    path_metadata: Dict[str, Any] = field(default_factory=dict)
