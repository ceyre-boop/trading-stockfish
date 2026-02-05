from typing import Any, Dict, List, Tuple

from .decision_actions import ActionType, DecisionAction
from .decision_frame import DecisionFrame
from .entry_brain import BrainPolicy
from .entry_eligibility import get_eligible_entry_models, is_entry_eligible
from .entry_models import ENTRY_MODELS, EntryModelDefinition, EntryModelSpec
from .ev_brain_features import FEATURE_COLUMNS
from .ev_brain_inference import evaluate_actions
from .ev_dataset_builder import (
    ACTION_TYPE_ENCODING,
    DIR_ENCODING,
    ENTRY_ID_TO_IDX,
    SIZE_BUCKET_ENCODING,
    _flatten_structure,
)


def _resolve_entry_models(
    entry_models: List[EntryModelSpec],
) -> List[EntryModelDefinition]:
    resolved: List[EntryModelDefinition] = []
    for model in entry_models or []:
        entry_id = (
            model.get("id") if isinstance(model, dict) else getattr(model, "id", None)
        )
        if entry_id and entry_id in ENTRY_MODELS:
            resolved.append(ENTRY_MODELS[entry_id])
    return resolved


def _direction_candidates(model: EntryModelDefinition) -> List[str]:
    direction = (model.direction or "BOTH").upper()
    if direction == "LONG":
        return ["LONG"]
    if direction == "SHORT":
        return ["SHORT"]
    return ["LONG", "SHORT"]


def _within_risk_envelope(action: DecisionAction, risk_envelope: Any) -> bool:
    # Placeholder: accept all; structure exists for future checks
    return True


def generate_candidate_actions(
    frame: DecisionFrame,
    position_state: Any,
    entry_models: List[EntryModelSpec],
    brain_policy: BrainPolicy,
    risk_envelope: Any,
) -> List[DecisionAction]:
    candidates: List[DecisionAction] = [DecisionAction(action_type=ActionType.NO_TRADE)]

    resolved_models = _resolve_entry_models(entry_models) or list(ENTRY_MODELS.values())
    eligible_ids = set(get_eligible_entry_models(frame))

    for model in resolved_models:
        if model.id not in eligible_ids:
            continue
        policy_label = (
            brain_policy.lookup(model.id, frame)
            if hasattr(brain_policy, "lookup")
            else "DISABLED"
        )
        if policy_label == "DISABLED":
            continue
        for direction in _direction_candidates(model):
            action_type = (
                ActionType.OPEN_LONG if direction == "LONG" else ActionType.OPEN_SHORT
            )
            action = DecisionAction(
                action_type=action_type,
                entry_model_id=model.id,
                direction=direction,
                size_bucket="SMALL",
                stop_structure={"preset": "default"},
                tp_structure={"preset": "default"},
            )
            if _within_risk_envelope(action, risk_envelope):
                candidates.append(action)

    is_open = False
    if isinstance(position_state, dict):
        is_open = bool(position_state.get("is_open"))
    if is_open:
        manage_payloads = [
            {"action": "CLOSE_POSITION"},
            {"action": "SCALE_OUT", "fraction": 0.5},
            {"action": "MOVE_STOP", "mode": "BREAKEVEN"},
        ]
        for payload in manage_payloads:
            action = DecisionAction(
                action_type=ActionType.MANAGE_POSITION,
                manage_payload=payload,
            )
            if _within_risk_envelope(action, risk_envelope):
                candidates.append(action)

    return candidates


def _state_features_from_frame(frame: DecisionFrame) -> Dict[str, Any]:
    return {
        "state_market_profile_state": getattr(frame, "market_profile_state", None),
        "state_market_profile_confidence": getattr(
            frame, "market_profile_confidence", None
        ),
        "state_session_profile": getattr(frame, "session_profile", None),
        "state_session_profile_confidence": getattr(
            frame, "session_profile_confidence", None
        ),
        "state_vol_regime": getattr(frame, "vol_regime", None),
        "state_trend_regime": getattr(frame, "trend_regime", None),
        "state_liquidity_bias": (
            None if frame.liquidity_frame is None else frame.liquidity_frame.get("bias")
        ),
        "state_condition_vector": getattr(frame, "condition_vector", None),
    }


def _encode_action_features(action: DecisionAction) -> Dict[str, Any]:
    action_type_id = ACTION_TYPE_ENCODING.get(action.action_type, 0)
    direction = (action.direction or "").upper() if action.direction else None
    direction_id = DIR_ENCODING.get(direction, 0)
    size_bucket = (action.size_bucket or "").upper() if action.size_bucket else None
    size_bucket_id = SIZE_BUCKET_ENCODING.get(size_bucket, 0)
    entry_id = action.entry_model_id
    entry_idx = ENTRY_ID_TO_IDX.get(entry_id, 0)

    return {
        "action_type": action.action_type.value,
        "action_type_id": action_type_id,
        "entry_model_id": entry_id,
        "entry_model_id_idx": entry_idx,
        "direction": direction if direction in ("LONG", "SHORT") else None,
        "direction_id": direction_id,
        "size_bucket": (
            size_bucket if size_bucket in ("SMALL", "MEDIUM", "LARGE") else None
        ),
        "size_bucket_id": size_bucket_id,
        "stop_structure_json": _flatten_structure(action.stop_structure),
        "tp_structure_json": _flatten_structure(action.tp_structure),
        "manage_payload_json": _flatten_structure(action.manage_payload),
    }


def build_action_feature_rows(
    frame: DecisionFrame, candidate_actions: List[DecisionAction]
) -> List[Dict[str, Any]]:
    state_features = _state_features_from_frame(frame)
    rows: List[Dict[str, Any]] = []
    for action in candidate_actions:
        row: Dict[str, Any] = {col: None for col in FEATURE_COLUMNS}
        row.update(state_features)
        row.update(_encode_action_features(action))
        rows.append(row)
    return rows


def score_candidate_actions(
    ev_brain,
    frame: DecisionFrame,
    position_state: Any,
    entry_models: List[EntryModelSpec],
    brain_policy: BrainPolicy,
    risk_envelope: Any,
) -> List[Tuple[DecisionAction, float]]:
    candidate_actions = generate_candidate_actions(
        frame, position_state, entry_models, brain_policy, risk_envelope
    )
    feature_rows = build_action_feature_rows(frame, candidate_actions)
    scores = evaluate_actions(ev_brain, feature_rows)
    return list(zip(candidate_actions, scores))
