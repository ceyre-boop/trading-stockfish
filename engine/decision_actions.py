from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ActionType(str, Enum):
    NO_TRADE = "NO_TRADE"
    OPEN_LONG = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    MANAGE_POSITION = "MANAGE_POSITION"


@dataclass
class DecisionAction:
    action_type: ActionType
    entry_model_id: Optional[str] = None
    direction: Optional[str] = None
    size_bucket: Optional[str] = None
    stop_structure: Optional[Dict[str, Any]] = None
    tp_structure: Optional[Dict[str, Any]] = None
    manage_payload: Optional[Dict[str, Any]] = None


@dataclass
class DecisionOutcome:
    realized_R: float
    max_adverse_excursion: float
    max_favorable_excursion: float
    time_in_trade_bars: int
    drawdown_impact: float


@dataclass
class DecisionRecord:
    decision_id: str
    timestamp_utc: Optional[str] = None
    bar_index: Optional[int] = None
    state_ref: Optional[str] = None
    action: Optional[DecisionAction] = None
    outcome: Optional[DecisionOutcome] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
