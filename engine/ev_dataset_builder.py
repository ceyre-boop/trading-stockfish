import json
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .decision_actions import ActionType, DecisionRecord
from .decision_frame import DecisionFrame
from .entry_models import ENTRY_MODELS

# Deterministic encoders for categorical fields
ACTION_TYPE_ENCODING = {
    ActionType.NO_TRADE: 0,
    ActionType.OPEN_LONG: 1,
    ActionType.OPEN_SHORT: 2,
    ActionType.MANAGE_POSITION: 3,
}

DIR_ENCODING = {None: 0, "LONG": 1, "SHORT": -1}
SIZE_BUCKET_ENCODING = {None: 0, "SMALL": 1, "MEDIUM": 2, "LARGE": 3}

ENTRY_ID_TO_IDX = {
    entry_id: idx + 1 for idx, entry_id in enumerate(ENTRY_MODELS.keys())
}


def _flatten_structure(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except Exception:
        try:
            return json.dumps(str(value))
        except Exception:
            return None


def _coerce_frame(raw: Any) -> DecisionFrame:
    if isinstance(raw, DecisionFrame):
        return raw
    frame = DecisionFrame()
    if not isinstance(raw, dict):
        return frame
    for key, value in raw.items():
        if hasattr(frame, key):
            setattr(frame, key, value)
    return frame


def _extract_state_features(
    frame: DecisionFrame, replay_row: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    structural = replay_row.get("structural_context") if replay_row else {}
    liq_bias = None
    if isinstance(frame.liquidity_frame, dict):
        liq_bias = frame.liquidity_frame.get("bias")
    elif isinstance(structural, dict):
        liq_bias = structural.get("liquidity_bias_side")

    return {
        "state_timestamp_utc": (
            replay_row.get("timestamp_utc")
            if replay_row
            else getattr(frame, "timestamp_utc", None)
        ),
        "state_market_profile_state": frame.market_profile_state
        or (
            structural.get("market_profile_state")
            if isinstance(structural, dict)
            else None
        ),
        "state_market_profile_confidence": frame.market_profile_confidence,
        "state_session_profile": frame.session_profile,
        "state_session_profile_confidence": frame.session_profile_confidence,
        "state_vol_regime": frame.vol_regime
        or (structural.get("vol_regime") if isinstance(structural, dict) else None),
        "state_trend_regime": frame.trend_regime
        or (structural.get("trend_regime") if isinstance(structural, dict) else None),
        "state_liquidity_bias": liq_bias,
        "state_condition_vector": frame.condition_vector,
    }


def _encode_action(record: DecisionRecord) -> Dict[str, Any]:
    action = record.action
    action_type = action.action_type if action else ActionType.NO_TRADE
    action_type_id = ACTION_TYPE_ENCODING.get(action_type, 0)

    direction_raw = (action.direction or "").upper() if action else None
    direction = direction_raw if direction_raw in ("LONG", "SHORT") else None
    direction_id = DIR_ENCODING.get(direction, 0)

    size_raw = (action.size_bucket or "").upper() if action else None
    size_bucket = size_raw if size_raw in ("SMALL", "MEDIUM", "LARGE") else None
    size_bucket_id = SIZE_BUCKET_ENCODING.get(size_bucket, 0)

    entry_id = action.entry_model_id if action else None
    entry_idx = ENTRY_ID_TO_IDX.get(entry_id, 0)

    return {
        "action_type": (
            action_type.value
            if isinstance(action_type, ActionType)
            else str(action_type)
        ),
        "action_type_id": action_type_id,
        "entry_model_id": entry_id,
        "entry_model_id_idx": entry_idx,
        "direction": direction,
        "direction_id": direction_id,
        "size_bucket": size_bucket,
        "size_bucket_id": size_bucket_id,
        "stop_structure_json": _flatten_structure(
            action.stop_structure if action else None
        ),
        "tp_structure_json": _flatten_structure(
            action.tp_structure if action else None
        ),
        "manage_payload_json": _flatten_structure(
            action.manage_payload if action else None
        ),
    }


def _encode_outcome(record: DecisionRecord) -> Dict[str, Any]:
    outcome = record.outcome
    if outcome is None:
        return {}
    return {
        "label_realized_R": outcome.realized_R,
        "label_max_adverse_excursion": outcome.max_adverse_excursion,
        "label_max_favorable_excursion": outcome.max_favorable_excursion,
        "label_time_in_trade_bars": outcome.time_in_trade_bars,
        "label_drawdown_impact": outcome.drawdown_impact,
    }


def build_ev_dataset(
    replay_logs: pd.DataFrame,
    decision_records: Iterable[DecisionRecord],
    *,
    version: str = "v1",
) -> pd.DataFrame:
    """Build EV brain training tuples (state, action, outcome) with deterministic schema.

    Only rows with known outcomes are emitted. No replay behavior is altered; this
    consumes already-logged decisions and replay outputs.
    """

    records: List[Dict[str, Any]] = []

    replay_index: Dict[str, Dict[str, Any]] = {}
    if isinstance(replay_logs, pd.DataFrame) and not replay_logs.empty:
        for _, row in replay_logs.iterrows():
            did = row.get("decision_id") or row.get("id")
            if did:
                replay_index[str(did)] = row.to_dict()

    for rec in decision_records or []:
        if rec is None or rec.outcome is None:
            continue

        replay_row = replay_index.get(rec.decision_id, {})

        frame_raw = replay_row.get("decision_frame")
        frame = _coerce_frame(frame_raw)

        state_features = _extract_state_features(frame, replay_row)
        action_features = _encode_action(rec)
        outcome_labels = _encode_outcome(rec)

        row = {
            "decision_id": rec.decision_id,
            "dataset_version": version,
            "timestamp_utc": rec.timestamp_utc
            or state_features.get("state_timestamp_utc"),
            "bar_index": rec.bar_index,
            "state_ref": rec.state_ref,
        }
        row.update(state_features)
        row.update(action_features)
        row.update(outcome_labels)

        records.append(row)

    if not records:
        return pd.DataFrame()

    ordered_columns = [
        # metadata
        "decision_id",
        "timestamp_utc",
        "bar_index",
        "state_ref",
        "dataset_version",
        # state features
        "state_timestamp_utc",
        "state_market_profile_state",
        "state_market_profile_confidence",
        "state_session_profile",
        "state_session_profile_confidence",
        "state_vol_regime",
        "state_trend_regime",
        "state_liquidity_bias",
        "state_condition_vector",
        # action features
        "action_type",
        "action_type_id",
        "entry_model_id",
        "entry_model_id_idx",
        "direction",
        "direction_id",
        "size_bucket",
        "size_bucket_id",
        "stop_structure_json",
        "tp_structure_json",
        "manage_payload_json",
        # outcome labels
        "label_realized_R",
        "label_max_adverse_excursion",
        "label_max_favorable_excursion",
        "label_time_in_trade_bars",
        "label_drawdown_impact",
    ]

    df = pd.DataFrame(records)
    # Ensure deterministic column ordering and presence
    for col in ordered_columns:
        if col not in df.columns:
            df[col] = None
    df = df[ordered_columns]

    return df.reset_index(drop=True)
