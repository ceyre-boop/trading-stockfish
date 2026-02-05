from typing import Any, Dict, List, Optional

import pandas as pd

from .decision_frame import DecisionFrame
from .entry_features import extract_entry_features

_COLUMNS = [
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
    "risk_expected_R",
    "risk_mae_bucket",
    "risk_mfe_bucket",
    "risk_horizon",
    "risk_aggressiveness",
    "entry_model_id",
    "eligible_flag",
    "chosen_flag",
    "entry_outcome_R",
    "entry_success",
    "timestamp_utc",
    "symbol",
]


def _frame_from_dict(raw: Optional[Dict[str, Any]]) -> DecisionFrame:
    frame = DecisionFrame()
    if isinstance(raw, dict):
        for k, v in raw.items():
            if hasattr(frame, k):
                setattr(frame, k, v)
    return frame


def _resolve_eligible(row: Dict[str, Any], frame: DecisionFrame) -> List[str]:
    eligible = row.get("eligible_entry_models")
    if not eligible and isinstance(frame, DecisionFrame):
        eligible = frame.eligible_entry_models
    if not eligible:
        eligible = []
    return list(eligible)


def build_entry_selector_dataset(decision_logs: pd.DataFrame) -> pd.DataFrame:
    if decision_logs is None or decision_logs.empty:
        return pd.DataFrame(columns=_COLUMNS)

    records: List[Dict[str, Any]] = []

    for _, row in decision_logs.iterrows():
        frame_raw = row.get("decision_frame")
        frame = _frame_from_dict(frame_raw)
        chosen_id = row.get("entry_model_id")
        if isinstance(frame_raw, dict):
            # Only override when a chosen entry is explicitly populated; avoid replacing
            # the decision log's chosen id with a null from the frame dict.
            chosen_in_frame = frame_raw.get("chosen_entry_model_id")
            chosen_id = chosen_in_frame or chosen_id
        outcome = (
            row.get("entry_outcome") if "entry_outcome" in row else row.get("outcome")
        )
        timestamp = row.get("timestamp_utc") or getattr(frame, "timestamp_utc", None)
        symbol = row.get("symbol") or getattr(frame, "symbol", None)

        eligible_models = _resolve_eligible(row, frame)
        if not eligible_models and chosen_id:
            eligible_models = [chosen_id]

        for entry_id in eligible_models:
            try:
                feats = extract_entry_features(entry_id, frame)
            except Exception:
                feats = {k: None for k in _COLUMNS}

            chosen_flag = 1 if entry_id == chosen_id else 0
            entry_outcome = outcome if chosen_flag else None
            entry_success = None
            if entry_outcome is not None:
                try:
                    entry_success = 1 if float(entry_outcome) > 0 else 0
                except Exception:
                    entry_success = None

            record = {
                "market_profile_state": feats.get("market_profile_state"),
                "session_profile": feats.get("session_profile"),
                "liquidity_bias_side": feats.get("liquidity_bias_side"),
                "liquidity_nearest_target_distance": feats.get(
                    "liquidity_nearest_target_distance"
                ),
                "liquidity_sweep_state": feats.get("liquidity_sweep_state"),
                "vol_regime": feats.get("vol_regime"),
                "trend_regime": feats.get("trend_regime"),
                "sweep_flag": feats.get("sweep_flag"),
                "displacement_score": feats.get("displacement_score"),
                "fvg_flag": feats.get("fvg_flag"),
                "ob_flag": feats.get("ob_flag"),
                "ifvg_flag": feats.get("ifvg_flag"),
                "risk_expected_R": feats.get("risk_expected_R"),
                "risk_mae_bucket": feats.get("risk_mae_bucket"),
                "risk_mfe_bucket": feats.get("risk_mfe_bucket"),
                "risk_horizon": feats.get("risk_horizon"),
                "risk_aggressiveness": feats.get("risk_aggressiveness"),
                "entry_model_id": entry_id,
                "eligible_flag": 1,
                "chosen_flag": chosen_flag,
                "entry_outcome_R": entry_outcome,
                "entry_success": entry_success,
                "timestamp_utc": timestamp,
                "symbol": symbol,
            }
            records.append(record)

    result = pd.DataFrame(records, columns=_COLUMNS)
    if result.empty:
        return result

    if "timestamp_utc" in result.columns:
        try:
            result["timestamp_utc"] = pd.to_datetime(
                result["timestamp_utc"], errors="coerce"
            )
        except Exception:
            pass

    result = result.sort_values(
        ["timestamp_utc", "entry_model_id"], na_position="last"
    ).reset_index(drop=True)
    return result
