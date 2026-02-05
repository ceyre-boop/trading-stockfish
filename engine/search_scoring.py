from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .decision_actions import DecisionAction
from .decision_frame import DecisionFrame
from .endgame_tablebases import EndgameTablebasesV1
from .ev_brain_features import FEATURE_COLUMNS, build_feature_matrix
from .mcr_engine import evaluate_action_via_mcr
from .mcr_scenarios import MCRActionContext
from .opening_book import OpeningBookV1, build_action_id
from .pattern_templates import PATTERN_TEMPLATES, PatternFamily
from .template_performance import TemplatePolicyLabel
from .transposition_table import TranspositionTable


def _action_to_feature_row(
    frame: DecisionFrame, action: DecisionAction
) -> Dict[str, Any]:
    row: Dict[str, Any] = {col: None for col in FEATURE_COLUMNS}
    row.update(
        {
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
                None
                if frame.liquidity_frame is None
                else frame.liquidity_frame.get("bias")
            ),
            "state_condition_vector": getattr(frame, "condition_vector", None),
        }
    )
    row.update(
        {
            "action_type_id": None,
            "entry_model_id_idx": None,
            "direction_id": None,
            "size_bucket_id": None,
            "stop_structure_json": None,
            "tp_structure_json": None,
            "manage_payload_json": None,
            "template_id": None,
            "template_eco_code": None,
            "template_family_one_hot": {},
            "template_risk_profile": {},
        }
    )
    # Leverage ev_dataset_builder encodings by mimicking fields
    from .ev_dataset_builder import (
        ACTION_TYPE_ENCODING,
        DIR_ENCODING,
        ENTRY_ID_TO_IDX,
        SIZE_BUCKET_ENCODING,
        _flatten_structure,
    )

    action_type_id = ACTION_TYPE_ENCODING.get(action.action_type, 0)
    direction = (action.direction or "").upper() if action.direction else None
    direction_id = DIR_ENCODING.get(direction, 0)
    size_bucket = (action.size_bucket or "").upper() if action.size_bucket else None
    size_bucket_id = SIZE_BUCKET_ENCODING.get(size_bucket, 0)
    entry_idx = ENTRY_ID_TO_IDX.get(action.entry_model_id, 0)

    row.update(
        {
            "action_type": action.action_type.value,
            "action_type_id": action_type_id,
            "entry_model_id": action.entry_model_id,
            "entry_model_id_idx": entry_idx,
            "direction": direction,
            "direction_id": direction_id,
            "size_bucket": size_bucket,
            "size_bucket_id": size_bucket_id,
            "stop_structure_json": _flatten_structure(action.stop_structure),
            "tp_structure_json": _flatten_structure(action.tp_structure),
            "manage_payload_json": _flatten_structure(action.manage_payload),
        }
    )

    tmpl_meta = _resolve_template_context(action)
    row.update(
        {
            "template_id": tmpl_meta["template_id"],
            "template_eco_code": tmpl_meta["eco_code"],
            "template_family_one_hot": tmpl_meta["family_one_hot"],
            "template_risk_profile": tmpl_meta["risk_profile"],
        }
    )
    return row


def _ev_brain_score(ev_brain, frame: DecisionFrame, action: DecisionAction) -> float:
    row = _action_to_feature_row(frame, action)
    X = build_feature_matrix(pd.DataFrame([row]))
    preds = ev_brain.predict(X)
    return float(preds[0])


def _resolve_template_context(action: DecisionAction) -> Dict[str, Any]:
    template_id = getattr(action, "template_id", None) or getattr(
        action, "entry_model_id", None
    )
    template = PATTERN_TEMPLATES.get(template_id) if template_id else None
    eco_code = template.eco_code if template else getattr(action, "eco_code", None)
    family = template.family if template else None

    family_one_hot: Dict[str, int] = {}
    if family is not None and isinstance(family, PatternFamily):
        for fam in PatternFamily:
            family_one_hot[fam.value] = 1 if fam == family else 0
    elif template_id is not None:
        # Known template id but not registered; keep deterministic zero vector
        family_one_hot = {fam.value: 0 for fam in PatternFamily}

    risk_profile = template.risk_profile if template else {}

    return {
        "template_id": template_id,
        "eco_code": eco_code,
        "family_one_hot": family_one_hot,
        "risk_profile": risk_profile,
    }


def _apply_template_policy(
    ev_hat: float,
    action: DecisionAction,
    template_policy: Dict[str, TemplatePolicyLabel] | None,
) -> Tuple[float, str | None, float]:
    """Apply deterministic template policy adjustments to EV.

    Boosts/penalties are small and documented to keep them secondary:
    - PREFERRED: +0.05
    - DISCOURAGED: -0.05
    - DISABLED: hard floor to -10.0
    - ALLOWED/unknown: 0.0
    """

    if template_policy is None:
        return ev_hat, None, 0.0

    template_id = getattr(action, "template_id", None) or getattr(
        action, "entry_model_id", None
    )
    if not template_id:
        return ev_hat, None, 0.0

    label_obj = template_policy.get(template_id)
    label = None
    if isinstance(label_obj, TemplatePolicyLabel):
        label = label_obj.label
    elif isinstance(label_obj, dict):
        label = label_obj.get("label")

    if label is None:
        return ev_hat, None, 0.0

    boost = 0.0
    if label == "PREFERRED":
        boost = 0.05
    elif label == "DISCOURAGED":
        boost = -0.05
    elif label == "DISABLED":
        return -10.0, label, -10.0 - ev_hat

    return ev_hat + boost, label, boost


def score_actions_via_search(
    frame: DecisionFrame,
    position_state: Any,
    candidate_actions: List[DecisionAction],
    *,
    ev_brain,
    brain_policy,
    risk_envelope,
    n_paths: int,
    horizon_bars: int,
    seed: int,
    template_policy: Dict[str, TemplatePolicyLabel] | None = None,
) -> List[Tuple[DecisionAction, Dict[str, Any]]]:
    results: List[Tuple[DecisionAction, Dict[str, Any]]] = []

    for idx, action in enumerate(candidate_actions):
        mcr_ctx = MCRActionContext(
            decision_frame_ref=getattr(frame, "timestamp_utc", None)
            or getattr(frame, "symbol", None)
            or idx,
            initial_position_state=(
                position_state if isinstance(position_state, dict) else {}
            ),
            decision_action=action,
            risk_envelope=risk_envelope if isinstance(risk_envelope, dict) else {},
        )
        mcr_metrics = evaluate_action_via_mcr(
            frame,
            mcr_ctx,
            n_paths=n_paths,
            horizon_bars=horizon_bars,
            seed=seed + idx,
        )

        ev_hat = _ev_brain_score(ev_brain, frame, action)
        ev_hat_adj, tpl_label, tpl_adj = _apply_template_policy(
            ev_hat, action, template_policy
        )

        entry_id = action.entry_model_id or ""
        label = (
            brain_policy.lookup(entry_id, frame)
            if hasattr(brain_policy, "lookup")
            else "DISABLED"
        )
        multiplier = (
            brain_policy.multiplier_for(label)
            if hasattr(brain_policy, "multiplier_for")
            else 0.0
        )

        unified = (
            ev_hat_adj
            + 0.5 * mcr_metrics.get("mean_EV", 0.0)
            - 0.2 * mcr_metrics.get("variance_EV", 0.0)
            - 0.3 * mcr_metrics.get("tail_risk", 0.0)
            + 0.1 * mcr_metrics.get("tp_hit_rate", 0.0)
            - 0.1 * mcr_metrics.get("stop_hit_rate", 0.0)
            + multiplier
        )

        score = {
            "EV_brain_raw": ev_hat,
            "EV_brain": ev_hat_adj,
            "MCR": mcr_metrics,
            "policy_multiplier": multiplier,
            "template_policy_label": tpl_label,
            "template_policy_adjustment": tpl_adj,
            "unified_score": float(unified),
        }
        results.append((action, score))

    return results


def score_actions_with_cache(
    frame: DecisionFrame,
    position_state: Any,
    candidate_actions: List[DecisionAction],
    *,
    ev_brain,
    brain_policy,
    risk_envelope,
    n_paths: int,
    horizon_bars: int,
    seed: int,
    table: TranspositionTable,
    template_policy: Dict[str, TemplatePolicyLabel] | None = None,
) -> List[Tuple[DecisionAction, Dict[str, Any]]]:
    results: List[Tuple[DecisionAction, Dict[str, Any]]] = []
    for action in candidate_actions:
        tpl_id = getattr(action, "template_id", None) or getattr(
            action, "entry_model_id", None
        )
        tpl_label = None
        if template_policy and tpl_id in template_policy:
            label_obj = template_policy.get(tpl_id)
            tpl_label = (
                label_obj.label
                if isinstance(label_obj, TemplatePolicyLabel)
                else label_obj.get("label") if isinstance(label_obj, dict) else None
            )
        base_key = table.compute_state_hash(frame, position_state, action)
        key = f"{base_key}|tpl:{tpl_label}" if tpl_label is not None else base_key
        cached = table.lookup(key)
        if cached is not None:
            results.append((action, cached))
            continue
        scores = score_actions_via_search(
            frame,
            position_state,
            [action],
            ev_brain=ev_brain,
            brain_policy=brain_policy,
            risk_envelope=risk_envelope,
            n_paths=n_paths,
            horizon_bars=horizon_bars,
            seed=seed,
            template_policy=template_policy,
        )[0][1]
        table.store(key, scores)
        results.append((action, scores))
    return results


def apply_opening_book_scores(
    opening_book: OpeningBookV1,
    frame: DecisionFrame,
    position_state: Any,
    candidate_actions: List[DecisionAction],
    score_dicts: List[Dict[str, Any]],
) -> None:
    """
    Apply opening book adjustments to unified scores in-place.
    """

    opening_scores = opening_book.lookup(frame, position_state, candidate_actions)
    for action, score_dict in zip(candidate_actions, score_dicts):
        aid = build_action_id(action)
        boost = float(opening_scores.get(aid, 0.0))
        score_dict["opening_book_score"] = boost
        score_dict["unified_score"] = float(
            score_dict.get("unified_score", 0.0) + boost
        )


def apply_endgame_tablebases(
    tablebases: EndgameTablebasesV1,
    frame: DecisionFrame,
    position_state: Any,
    candidate_actions: List[DecisionAction],
    score_dicts: List[Dict[str, Any]],
) -> None:
    """
    Apply endgame tablebase overrides to unified scores in-place.
    """

    tb_scores = tablebases.lookup(frame, position_state, candidate_actions)
    for action, score_dict in zip(candidate_actions, score_dicts):
        aid = build_action_id(action)
        boost = float(tb_scores.get(aid, 0.0))
        score_dict["endgame_score"] = boost
        score_dict["unified_score"] = float(
            score_dict.get("unified_score", 0.0) + boost
        )
