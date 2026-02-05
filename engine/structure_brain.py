import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from .market_profile_features import MarketProfileFeatures
from .market_profile_model import (
    TrainedMarketProfileModel,
    predict_market_profile_proba,
)
from .market_profile_rules import coarse_classify_market_profile
from .market_profile_state_machine import MarketProfileStateMachine

# Phase 12B – Core structures and rule-based classifiers (Modules 1–3)


# ----------------------------
# Data structures
# ----------------------------


@dataclass
class MarketProfileFrame:
    state: str
    confidence: float
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionProfileFrame:
    profile: str
    confidence: float
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LiquidityFrame:
    primary_target: Optional[str]
    target_side: Optional[str]
    distances: Dict[str, float]
    swept: Dict[str, bool]
    bias: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------
# Rule helpers
# ----------------------------


def _bool(val: Any) -> bool:
    return bool(val) if val is not None else False


def _score(values: List[Tuple[bool, float]]) -> float:
    # sum of weights for true conditions, clipped
    total = sum(weight for cond, weight in values if cond)
    return float(max(0.0, min(1.0, total)))


# ----------------------------
# Module 1 — Market Structure Ontology (A/M/D)
# ----------------------------


def classify_market_profile_simple(features: Dict[str, Any]) -> MarketProfileFrame:
    """Classify market profile as ACCUMULATION / MANIPULATION / DISTRIBUTION / TRANSITION.

    Deterministic heuristics with bounded confidence. Inputs expected in `features`:
    - volatility_regime: str in {LOW, NORMAL, HIGH, EXTREME}
    - trend_strength: float (0-1)
    - displacement_events: bool
    - sweep_recent: bool
    - compression: bool
    - expansion: bool
    - volume_anomaly: bool
    - trend_direction: str
    """

    vol = str(features.get("volatility_regime", "UNKNOWN")).upper()
    trend_strength = float(features.get("trend_strength", 0.0) or 0.0)
    displacement = _bool(features.get("displacement_events"))
    sweep = _bool(features.get("sweep_recent"))
    compression = _bool(features.get("compression"))
    expansion = _bool(features.get("expansion"))
    volume_anomaly = _bool(features.get("volume_anomaly"))

    evidence = {
        "volatility_regime": vol,
        "trend_strength": trend_strength,
        "displacement": displacement,
        "sweep_recent": sweep,
        "compression": compression,
        "expansion": expansion,
        "volume_anomaly": volume_anomaly,
    }

    # ACCUMULATION: low vol, compression, no displacement
    acc_score = _score(
        [
            (vol in {"LOW", "NORMAL"}, 0.3),
            (compression, 0.4),
            (not displacement, 0.2),
            (not sweep, 0.1),
        ]
    )

    # MANIPULATION: sweep + displacement + volume anomaly or vol spike
    man_score = _score(
        [
            (sweep, 0.35),
            (displacement, 0.35),
            (vol in {"HIGH", "EXTREME"}, 0.2),
            (volume_anomaly, 0.2),
        ]
    )

    # DISTRIBUTION: expansion + sustained trend
    dist_score = _score(
        [
            (expansion, 0.3),
            (trend_strength >= 0.4, 0.4),
            (vol in {"NORMAL", "HIGH", "EXTREME"}, 0.2),
        ]
    )

    scores = {
        "ACCUMULATION": acc_score,
        "MANIPULATION": man_score,
        "DISTRIBUTION": dist_score,
        "TRANSITION": 0.2,
    }

    state = max(scores.items(), key=lambda kv: kv[1])[0]
    confidence = float(max(0.1, min(1.0, scores[state])))

    return MarketProfileFrame(state=state, confidence=confidence, evidence=evidence)


# High-level Phase 12B surface
def classify_market_profile(
    features: MarketProfileFeatures,
    ml_model: TrainedMarketProfileModel,
    state_machine: MarketProfileStateMachine,
) -> Dict[str, Any]:
    coarse_label = coarse_classify_market_profile(features)
    probs = predict_market_profile_proba(ml_model, features)
    state_info = state_machine.step(probs, features)

    evidence = {
        "coarse_label": coarse_label,
        "sweeps": {
            "pdh": features.swept_pdh,
            "pdl": features.swept_pdl,
            "session_high": features.swept_session_high,
            "session_low": features.swept_session_low,
            "eqh": features.swept_equal_highs,
            "eql": features.swept_equal_lows,
        },
        "displacement_score": features.displacement_score,
        "atr_vs_session_baseline": features.atr_vs_session_baseline,
        "intraday_range_vs_typical": features.intraday_range_vs_typical,
        "nearest_draw_side": features.nearest_draw_side,
        "trend_dir_ltf": features.trend_dir_ltf,
        "volume_spike": features.volume_spike,
    }

    return {
        "state": state_info.get("state"),
        "confidence": state_info.get("confidence"),
        "coarse_label": coarse_label,
        "probs": probs,
        "evidence": evidence,
        "transition": {
            "from_state": state_info.get("from_state"),
            "to_state": state_info.get("to_state"),
            "reason": state_info.get("transition_reason"),
        },
    }


# ----------------------------
# Module 2 — Session Profile Engine (1A / 1B / 1C)
# ----------------------------


def classify_session_profile(features: Dict[str, Any]) -> SessionProfileFrame:
    """Classify session profile: PROFILE_1A / PROFILE_1B / PROFILE_1C / UNKNOWN.

    Inputs expected in `features`:
    - previous_session_profile: str
    - early_volatility: float
    - sweep_first_leg: bool
    - displacement_first_leg: bool
    - continuation_after_reversal: bool
    - manipulation_detected: bool
    """

    prev = str(features.get("previous_session_profile", "UNKNOWN")).upper()
    early_vol = float(features.get("early_volatility", 0.0) or 0.0)
    sweep_first_leg = _bool(features.get("sweep_first_leg"))
    displacement_first_leg = _bool(features.get("displacement_first_leg"))
    continuation = _bool(features.get("continuation_after_reversal"))
    manipulation = _bool(features.get("manipulation_detected"))

    evidence = {
        "previous_session_profile": prev,
        "early_volatility": early_vol,
        "sweep_first_leg": sweep_first_leg,
        "displacement_first_leg": displacement_first_leg,
        "continuation_after_reversal": continuation,
        "manipulation_detected": manipulation,
    }

    # PROFILE_1A: consolidation -> manipulation -> reversal (needs sweep + displacement, then reversal)
    one_a_score = _score(
        [
            (manipulation, 0.3),
            (sweep_first_leg, 0.25),
            (displacement_first_leg, 0.2),
            (early_vol <= 0.5, 0.1),
        ]
    )

    # PROFILE_1B: manipulation -> reversal (no explicit consolidation, moderate vol)
    one_b_score = _score(
        [
            (manipulation, 0.35),
            (sweep_first_leg or displacement_first_leg, 0.25),
            (0.3 <= early_vol <= 0.8, 0.2),
        ]
    )

    # PROFILE_1C: manipulation + reversal -> continuation (needs continuation flag)
    one_c_score = _score(
        [
            (manipulation, 0.3),
            (sweep_first_leg or displacement_first_leg, 0.25),
            (continuation, 0.35),
        ]
    )

    scores = {
        "PROFILE_1A": one_a_score,
        "PROFILE_1B": one_b_score,
        "PROFILE_1C": one_c_score,
        "UNKNOWN": 0.1,
    }

    profile = max(scores.items(), key=lambda kv: kv[1])[0]
    confidence = float(max(0.1, min(1.0, scores[profile])))

    return SessionProfileFrame(
        profile=profile, confidence=confidence, evidence=evidence
    )


# ----------------------------
# Module 3 — Liquidity & Draw Model
# ----------------------------


def compute_liquidity_frame(draws: Dict[str, Dict[str, Any]]) -> LiquidityFrame:
    """Compute liquidity target selection and bias from draws.

    Expects `draws` mapping name -> {distance: float, swept: bool}.
    Example keys: PDH, PDL, SESSION_HIGH, SESSION_LOW, WEEKLY_HIGH, WEEKLY_LOW, EQH, EQL.
    """

    distances: Dict[str, float] = {}
    swept: Dict[str, bool] = {}
    for name, info in (draws or {}).items():
        distances[name] = float(info.get("distance", math.inf))
        swept[name] = _bool(info.get("swept"))

    # Choose nearest unswept target; if all swept, nearest overall
    unswept = {k: v for k, v in distances.items() if not swept.get(k, False)}
    candidates = unswept if unswept else distances
    primary_target = None
    min_dist = math.inf
    for k, v in candidates.items():
        if v < min_dist:
            min_dist = v
            primary_target = k

    # Determine side bias
    upper_keys = {"PDH", "SESSION_HIGH", "WEEKLY_HIGH", "EQH"}
    lower_keys = {"PDL", "SESSION_LOW", "WEEKLY_LOW", "EQL"}
    target_side = None
    if primary_target in upper_keys:
        target_side = "UP"
    elif primary_target in lower_keys:
        target_side = "DOWN"

    bias = target_side or "NEUTRAL"

    return LiquidityFrame(
        primary_target=primary_target,
        target_side=target_side,
        distances=distances,
        swept=swept,
        bias=bias,
    )
