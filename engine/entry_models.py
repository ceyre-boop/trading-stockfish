from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

from .decision_frame import DecisionFrame
from .entry_eligibility_overrides import (
    eligible_mean_reversion_range_extreme,
    eligible_sweep_displacement_reversal,
)


class RiskProfileSpec(TypedDict):
    expected_R: float
    mae_bucket: Literal["LOW", "MEDIUM", "HIGH"]
    mfe_bucket: Literal["LOW", "MEDIUM", "HIGH"]
    time_horizon: Literal["SCALP", "INTRADAY", "SWING"]
    aggressiveness: Literal["CONSERVATIVE", "NEUTRAL", "AGGRESSIVE"]


class EntryModelSpec(TypedDict, total=False):
    id: str
    family: str
    direction: str
    required_market_profile_states: List[str]
    required_session_profiles: List[str]
    required_liquidity_context: Dict[str, List[str]]
    required_vol_regimes: List[str]
    required_trend_regimes: List[str]
    required_signals: Dict[str, bool]
    time_horizon: str
    description: str
    risk_profile: RiskProfileSpec
    eligibility_fn: Optional[Callable[[DecisionFrame], bool]]


@dataclass(frozen=True)
class EntryModelDefinition:
    id: str
    family: str  # "SWEEP", "FVG", "OB", "BREAKER", "MITIGATION", "PA"
    direction: str  # "LONG", "SHORT", "BOTH"

    required_market_profile_states: List[str] = field(default_factory=list)
    required_session_profiles: List[str] = field(default_factory=list)

    required_liquidity_context: Dict[str, List[str]] = field(default_factory=dict)

    required_vol_regimes: List[str] = field(default_factory=list)
    required_trend_regimes: List[str] = field(default_factory=list)

    required_signals: Dict[str, bool] = field(default_factory=dict)

    time_horizon: str = "INTRADAY"
    description: str = ""
    risk_profile: RiskProfileSpec = field(default_factory=dict)
    eligibility_fn: Optional[Callable[[Any], bool]] = None


ENTRY_MODELS: Dict[str, EntryModelDefinition] = {}


def _register(model: EntryModelDefinition) -> EntryModelDefinition:
    ENTRY_MODELS[model.id] = model
    return model


def _elig_sweep_displacement_reversal(frame: Any) -> bool:
    if frame is None:
        return False
    signals = getattr(frame, "entry_signals_present", {}) or {}
    sweep = bool(signals.get("sweep")) if isinstance(signals, dict) else False
    evidence = getattr(frame, "market_profile_evidence", {}) or {}
    disp = None
    if isinstance(evidence, dict):
        disp = evidence.get("displacement_score")
    displacement_ok = True
    try:
        displacement_ok = disp is None or float(disp) >= 0.4
    except Exception:
        displacement_ok = False
    return sweep and displacement_ok


def _elig_mean_reversion_range_extreme(frame: Any) -> bool:
    if frame is None:
        return False
    signals = getattr(frame, "entry_signals_present", {}) or {}
    sweep = bool(signals.get("sweep")) if isinstance(signals, dict) else False
    evidence = getattr(frame, "market_profile_evidence", {}) or {}
    disp = None
    if isinstance(evidence, dict):
        disp = evidence.get("displacement_score")
    mild_displacement = True
    try:
        mild_displacement = disp is None or abs(float(disp)) <= 0.3
    except Exception:
        mild_displacement = True
    return (not sweep) and mild_displacement


# -------------------------------
# SWEEP family
# -------------------------------

_register(
    EntryModelDefinition(
        id="ENTRY_SWEEP_DISPLACEMENT_REVERSAL",
        family="SWEEP",
        direction="BOTH",
        required_market_profile_states=["MANIPULATION", "DISTRIBUTION"],
        required_session_profiles=["PROFILE_1A", "PROFILE_1B"],
        required_liquidity_context={
            "bias_side": ["UP", "DOWN"],
            "sweep_state": ["POST_SWEEP"],
            "distance_bucket": ["NEAR"],
        },
        required_vol_regimes=["HIGH", "NORMAL"],
        required_trend_regimes=["UP", "DOWN"],
        required_signals={
            "needs_sweep": True,
            "needs_displacement": True,
            "needs_fvg": False,
            "needs_ob": False,
            "needs_ifvg": False,
        },
        time_horizon="INTRADAY",
        description="Sweep key level then displacement reversal.",
        risk_profile={
            "expected_R": 2.0,
            "mae_bucket": "HIGH",
            "mfe_bucket": "HIGH",
            "time_horizon": "INTRADAY",
            "aggressiveness": "AGGRESSIVE",
        },
        eligibility_fn=eligible_sweep_displacement_reversal,
    )
)

_register(
    EntryModelDefinition(
        id="ENTRY_SWEEP_CONTINUATION",
        family="SWEEP",
        direction="BOTH",
        required_market_profile_states=["MANIPULATION", "DISTRIBUTION"],
        required_session_profiles=["PROFILE_1C", "UNKNOWN"],
        required_liquidity_context={
            "bias_side": ["UP", "DOWN"],
            "sweep_state": ["POST_SWEEP"],
            "distance_bucket": ["NEAR", "INSIDE"],
        },
        required_vol_regimes=["NORMAL", "HIGH"],
        required_trend_regimes=["UP", "DOWN"],
        required_signals={
            "needs_sweep": True,
            "needs_displacement": False,
            "needs_fvg": False,
            "needs_ob": False,
            "needs_ifvg": False,
        },
        time_horizon="INTRADAY",
        description="Sweep then continuation with trend.",
        risk_profile={
            "expected_R": 1.0,
            "mae_bucket": "LOW",
            "mfe_bucket": "LOW",
            "time_horizon": "SCALP",
            "aggressiveness": "CONSERVATIVE",
        },
    )
)

# -------------------------------
# FVG family
# -------------------------------

_register(
    EntryModelDefinition(
        id="ENTRY_FVG_RESPECT_CONTINUATION",
        family="FVG",
        direction="BOTH",
        required_market_profile_states=["DISTRIBUTION"],
        required_session_profiles=["PROFILE_1C", "UNKNOWN", "PROFILE_1B", "PROFILE_1A"],
        required_liquidity_context={
            "bias_side": ["UP", "DOWN"],
            "sweep_state": ["ANY"],
            "distance_bucket": ["NEAR", "INSIDE", "FAR"],
        },
        required_vol_regimes=["NORMAL", "HIGH"],
        required_trend_regimes=["UP", "DOWN"],
        required_signals={
            "needs_sweep": False,
            "needs_displacement": False,
            "needs_fvg": True,
            "needs_ob": False,
            "needs_ifvg": False,
        },
        time_horizon="INTRADAY",
        description="Trend-aligned FVG respect continuation.",
        risk_profile={
            "expected_R": 1.5,
            "mae_bucket": "MEDIUM",
            "mfe_bucket": "MEDIUM",
            "time_horizon": "INTRADAY",
            "aggressiveness": "NEUTRAL",
        },
    )
)

_register(
    EntryModelDefinition(
        id="ENTRY_FVG_DISRESPECT_REVERSAL",
        family="FVG",
        direction="BOTH",
        required_market_profile_states=["MANIPULATION", "TRANSITION"],
        required_session_profiles=["UNKNOWN", "PROFILE_1A", "PROFILE_1B"],
        required_liquidity_context={
            "bias_side": ["UP", "DOWN"],
            "sweep_state": ["ANY"],
            "distance_bucket": ["INSIDE", "FAR"],
        },
        required_vol_regimes=["NORMAL", "HIGH"],
        required_trend_regimes=["UP", "DOWN", "FLAT"],
        required_signals={
            "needs_sweep": False,
            "needs_displacement": False,
            "needs_fvg": True,
            "needs_ob": False,
            "needs_ifvg": False,
        },
        time_horizon="INTRADAY",
        description="FVG violation implying reversal/invalidation.",
        risk_profile={
            "expected_R": 1.0,
            "mae_bucket": "LOW",
            "mfe_bucket": "LOW",
            "time_horizon": "SCALP",
            "aggressiveness": "CONSERVATIVE",
        },
    )
)

_register(
    EntryModelDefinition(
        id="ENTRY_IFVG_REVERSAL",
        family="FVG",
        direction="BOTH",
        required_market_profile_states=["MANIPULATION", "DISTRIBUTION", "ACCUMULATION"],
        required_session_profiles=["UNKNOWN", "PROFILE_1A", "PROFILE_1B", "PROFILE_1C"],
        required_liquidity_context={
            "bias_side": ["UP", "DOWN"],
            "sweep_state": ["ANY"],
            "distance_bucket": ["NEAR", "INSIDE"],
        },
        required_vol_regimes=["NORMAL", "HIGH"],
        required_trend_regimes=["UP", "DOWN", "FLAT"],
        required_signals={
            "needs_sweep": False,
            "needs_displacement": False,
            "needs_fvg": False,
            "needs_ob": False,
            "needs_ifvg": True,
        },
        time_horizon="INTRADAY",
        description="Internal FVG with confluence for reversal/continuation.",
        risk_profile={
            "expected_R": 1.0,
            "mae_bucket": "LOW",
            "mfe_bucket": "LOW",
            "time_horizon": "SCALP",
            "aggressiveness": "CONSERVATIVE",
        },
    )
)

# -------------------------------
# OB family
# -------------------------------

_register(
    EntryModelDefinition(
        id="ENTRY_OB_CONTINUATION",
        family="OB",
        direction="BOTH",
        required_market_profile_states=["DISTRIBUTION"],
        required_session_profiles=["PROFILE_1C", "UNKNOWN", "PROFILE_1B"],
        required_liquidity_context={
            "bias_side": ["UP", "DOWN"],
            "sweep_state": ["ANY"],
            "distance_bucket": ["NEAR", "INSIDE"],
        },
        required_vol_regimes=["NORMAL", "HIGH"],
        required_trend_regimes=["UP", "DOWN"],
        required_signals={
            "needs_sweep": False,
            "needs_displacement": True,
            "needs_fvg": False,
            "needs_ob": True,
            "needs_ifvg": False,
        },
        time_horizon="INTRADAY",
        description="Trend OB respect continuation.",
        risk_profile={
            "expected_R": 1.2,
            "mae_bucket": "MEDIUM",
            "mfe_bucket": "MEDIUM",
            "time_horizon": "INTRADAY",
            "aggressiveness": "NEUTRAL",
        },
    )
)

_register(
    EntryModelDefinition(
        id="ENTRY_OB_REVERSAL",
        family="OB",
        direction="BOTH",
        required_market_profile_states=["MANIPULATION", "TRANSITION"],
        required_session_profiles=["PROFILE_1A", "PROFILE_1B", "UNKNOWN"],
        required_liquidity_context={
            "bias_side": ["UP", "DOWN"],
            "sweep_state": ["POST_SWEEP", "ANY"],
            "distance_bucket": ["NEAR", "INSIDE"],
        },
        required_vol_regimes=["NORMAL", "HIGH"],
        required_trend_regimes=["UP", "DOWN", "FLAT"],
        required_signals={
            "needs_sweep": False,
            "needs_displacement": False,
            "needs_fvg": False,
            "needs_ob": True,
            "needs_ifvg": False,
        },
        time_horizon="INTRADAY",
        description="OB at extreme used for reversal.",
        risk_profile={
            "expected_R": 1.0,
            "mae_bucket": "LOW",
            "mfe_bucket": "LOW",
            "time_horizon": "SCALP",
            "aggressiveness": "CONSERVATIVE",
        },
    )
)

# -------------------------------
# BREAKER / MITIGATION (stubs)
# -------------------------------

_register(
    EntryModelDefinition(
        id="ENTRY_BREAKER_FLIP",
        family="BREAKER",
        direction="BOTH",
        required_market_profile_states=["MANIPULATION", "DISTRIBUTION"],
        required_session_profiles=["UNKNOWN", "PROFILE_1B", "PROFILE_1C"],
        required_liquidity_context={
            "bias_side": ["UP", "DOWN"],
            "sweep_state": ["ANY"],
            "distance_bucket": ["INSIDE", "NEAR"],
        },
        required_vol_regimes=["NORMAL", "HIGH"],
        required_trend_regimes=["UP", "DOWN", "FLAT"],
        required_signals={
            "needs_sweep": False,
            "needs_displacement": False,
            "needs_fvg": False,
            "needs_ob": True,
            "needs_ifvg": False,
        },
        time_horizon="INTRADAY",
        description="Breaker flip after failed OB.",
        risk_profile={
            "expected_R": 1.0,
            "mae_bucket": "LOW",
            "mfe_bucket": "LOW",
            "time_horizon": "SCALP",
            "aggressiveness": "CONSERVATIVE",
        },
    )
)

_register(
    EntryModelDefinition(
        id="ENTRY_MITIGATION_BLOCK",
        family="MITIGATION",
        direction="BOTH",
        required_market_profile_states=["DISTRIBUTION"],
        required_session_profiles=["PROFILE_1C", "UNKNOWN", "PROFILE_1B"],
        required_liquidity_context={
            "bias_side": ["UP", "DOWN"],
            "sweep_state": ["ANY"],
            "distance_bucket": ["NEAR", "INSIDE"],
        },
        required_vol_regimes=["NORMAL", "HIGH"],
        required_trend_regimes=["UP", "DOWN"],
        required_signals={
            "needs_sweep": False,
            "needs_displacement": False,
            "needs_fvg": False,
            "needs_ob": False,
            "needs_ifvg": False,
        },
        time_horizon="INTRADAY",
        description="Mitigation block return/continuation (stub).",
        risk_profile={
            "expected_R": 1.0,
            "mae_bucket": "LOW",
            "mfe_bucket": "LOW",
            "time_horizon": "SCALP",
            "aggressiveness": "CONSERVATIVE",
        },
    )
)

# -------------------------------
# PA / INDICATOR
# -------------------------------

_register(
    EntryModelDefinition(
        id="ENTRY_TREND_PULLBACK",
        family="PA",
        direction="BOTH",
        required_market_profile_states=["DISTRIBUTION"],
        required_session_profiles=["PROFILE_1C", "PROFILE_1B", "UNKNOWN"],
        required_liquidity_context={
            "bias_side": ["UP", "DOWN"],
            "sweep_state": ["ANY"],
            "distance_bucket": ["INSIDE", "NEAR"],
        },
        required_vol_regimes=["NORMAL"],
        required_trend_regimes=["UP", "DOWN"],
        required_signals={
            "needs_sweep": False,
            "needs_displacement": False,
            "needs_fvg": False,
            "needs_ob": False,
            "needs_ifvg": False,
        },
        time_horizon="INTRADAY",
        description="Trend pullback with structure confluence.",
        risk_profile={
            "expected_R": 1.0,
            "mae_bucket": "LOW",
            "mfe_bucket": "LOW",
            "time_horizon": "SCALP",
            "aggressiveness": "CONSERVATIVE",
        },
    )
)

_register(
    EntryModelDefinition(
        id="ENTRY_MEAN_REVERSION_RANGE_EXTREME",
        family="PA",
        direction="BOTH",
        required_market_profile_states=["ACCUMULATION"],
        required_session_profiles=["UNKNOWN", "PROFILE_1A"],
        required_liquidity_context={
            "bias_side": ["UP", "DOWN", "NEUTRAL"],
            "sweep_state": ["ANY"],
            "distance_bucket": ["INSIDE", "NEAR", "FAR"],
        },
        required_vol_regimes=["LOW", "NORMAL"],
        required_trend_regimes=["FLAT"],
        required_signals={
            "needs_sweep": False,
            "needs_displacement": False,
            "needs_fvg": False,
            "needs_ob": False,
            "needs_ifvg": False,
        },
        time_horizon="INTRADAY",
        description="Range extreme mean reversion inside accumulation.",
        risk_profile={
            "expected_R": 1.0,
            "mae_bucket": "LOW",
            "mfe_bucket": "LOW",
            "time_horizon": "SCALP",
            "aggressiveness": "CONSERVATIVE",
        },
        eligibility_fn=eligible_mean_reversion_range_extreme,
    )
)
