from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from .decision_frame import DecisionFrame
from .entry_models import ENTRY_MODELS

MAE_BUCKET_ENCODING = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
MFE_BUCKET_ENCODING = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
TIME_HORIZON_ENCODING = {"SCALP": 0, "INTRADAY": 1, "SWING": 2}
AGGRESSIVENESS_ENCODING = {"CONSERVATIVE": 0, "NEUTRAL": 1, "AGGRESSIVE": 2}

ENTRY_FEATURE_NAMES: List[str] = [
    "market_profile_state",
    "session_profile",
    "liquidity_bias_side",
    "liquidity_nearest_target_distance",
    "liquidity_sweep_state",
    "vol_regime",
    "trend_regime",
    "sweep_flag",
    "displacement_score",
    "fvg_flag",
    "ob_flag",
    "ifvg_flag",
    "risk_per_trade",
    "position_size",
    "risk_expected_R",
    "risk_mae_bucket",
    "risk_mfe_bucket",
    "risk_horizon",
    "risk_aggressiveness",
    "risk_mae_bucket_encoded",
    "risk_mfe_bucket_encoded",
    "risk_time_horizon_encoded",
    "risk_aggressiveness_encoded",
]


@dataclass(frozen=True)
class EntryModelFeatureTemplate:
    id: str
    feature_names: List[str]


ENTRY_FEATURE_TEMPLATES: Dict[str, EntryModelFeatureTemplate] = {
    entry_id: EntryModelFeatureTemplate(
        id=entry_id, feature_names=list(ENTRY_FEATURE_NAMES)
    )
    for entry_id in ENTRY_MODELS
}


def _coerce_frame(frame: Optional[Any]) -> DecisionFrame:
    if isinstance(frame, DecisionFrame):
        return frame
    coerced = DecisionFrame()
    if isinstance(frame, Mapping):
        for key, value in frame.items():
            if hasattr(coerced, key):
                setattr(coerced, key, value)
    return coerced


def _nearest_target_distance(liquidity: Optional[Dict[str, Any]]) -> Any:
    if not isinstance(liquidity, dict):
        return None
    distances = liquidity.get("distances")
    primary_target = liquidity.get("primary_target") or liquidity.get("target")
    if isinstance(distances, dict) and primary_target is not None:
        return distances.get(primary_target)
    return None


def extract_entry_features(entry_id: str, frame: Optional[Any]) -> Dict[str, Any]:
    template = ENTRY_FEATURE_TEMPLATES.get(entry_id)
    if template is None:
        raise KeyError(f"Unknown entry model id: {entry_id}")

    model_def = ENTRY_MODELS.get(entry_id)
    risk = model_def.risk_profile if model_def else {}

    def _encode(mapping: Dict[str, int], value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        key = str(value).upper()
        return mapping.get(key)

    coerced = _coerce_frame(frame)
    liquidity = (
        coerced.liquidity_frame if isinstance(coerced.liquidity_frame, dict) else {}
    )
    entry_signals = coerced.entry_signals_present or {}
    displacement_score = None
    if isinstance(coerced.market_profile_evidence, dict):
        displacement_score = coerced.market_profile_evidence.get("displacement_score")

    values: Dict[str, Any] = {
        "market_profile_state": coerced.market_profile_state,
        "session_profile": coerced.session_profile,
        "liquidity_bias_side": liquidity.get("bias"),
        "liquidity_nearest_target_distance": _nearest_target_distance(liquidity),
        "liquidity_sweep_state": liquidity.get("sweep_state"),
        "vol_regime": coerced.vol_regime,
        "trend_regime": coerced.trend_regime,
        "sweep_flag": (
            entry_signals.get("sweep") if isinstance(entry_signals, dict) else None
        ),
        "displacement_score": displacement_score,
        "fvg_flag": (
            entry_signals.get("fvg") if isinstance(entry_signals, dict) else None
        ),
        "ob_flag": entry_signals.get("ob") if isinstance(entry_signals, dict) else None,
        "ifvg_flag": (
            entry_signals.get("ifvg") if isinstance(entry_signals, dict) else None
        ),
        "risk_per_trade": coerced.risk_per_trade,
        "position_size": coerced.position_size,
        "risk_expected_R": risk.get("expected_R"),
        "risk_mae_bucket": risk.get("mae_bucket"),
        "risk_mfe_bucket": risk.get("mfe_bucket"),
        "risk_horizon": risk.get("time_horizon"),
        "risk_aggressiveness": risk.get("aggressiveness"),
        "risk_mae_bucket_encoded": _encode(MAE_BUCKET_ENCODING, risk.get("mae_bucket")),
        "risk_mfe_bucket_encoded": _encode(MFE_BUCKET_ENCODING, risk.get("mfe_bucket")),
        "risk_time_horizon_encoded": _encode(
            TIME_HORIZON_ENCODING, risk.get("time_horizon")
        ),
        "risk_aggressiveness_encoded": _encode(
            AGGRESSIVENESS_ENCODING, risk.get("aggressiveness")
        ),
    }

    features: Dict[str, Any] = {}
    for name in template.feature_names:
        features[name] = values.get(name)
    return features
