#!/usr/bin/env python3
"""
Engine Evaluator Module - Trading Stockfish

Evaluates market state and generates trading decisions.
Implements multi-layered decision logic with risk filters and safety checks.

Decision Output: "buy", "sell", "hold", "close"

Logic Flow:
1. Safety checks (stale data, missing indicators, extreme conditions)
2. Trend regime detection (uptrend, downtrend, sideways)
3. Multi-timeframe confirmation (M1, M5, M15, H1)
4. Volatility and spread filters
5. Sentiment weighting
6. Final decision with confidence score

CausalEvaluator Integration:
- When use_causal_evaluator=True, uses Stockfish-style evaluation combining 8 market factors
- Deterministic, rule-based, fully explainable
- Requires all 8 market state components
- Produces eval_score [-1, +1] + confidence [0, 1]
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import test_flags
from .abs_balance_controller import BALANCE_CONTROLLER
from .canonical_validator import (
    assert_causal_required,
    assert_ml_advisory_only,
    canonical_enforced,
)
from .condition_encoder import encode_conditions
from .decision_frame import DecisionFrame
from .decision_logger import DecisionLogger
from .entry_eligibility import get_eligible_entry_models
from .evaluator_probabilities import compute_probability_tilts
from .market_profile_features import MarketProfileFeatures
from .market_profile_model import TrainedMarketProfileModel
from .market_profile_state_machine import MarketProfileStateMachine
from .ml_aux_signals import compute_ml_hints

# Regime engine helpers
from .regime_engine import compute_regime_bundle
from .structure_brain import (
    MarketProfileFrame,
    SessionProfileFrame,
    classify_market_profile,
    classify_session_profile,
    compute_liquidity_frame,
)

# Minimal core types (placeholder scaffolding)
from .types import EvaluationOutput, MarketState

# Microstructure imports
try:
    from engine.liquidity_metrics import compute_liquidity_metrics
except ImportError:
    compute_liquidity_metrics = None

from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)

_BRAIN_POLICY_DEFAULT_PATH = Path("storage/policies/brain/brain_policy.json")
_ENTRY_BRAIN_POLICY_PATH = Path(
    "storage/policies/brain/brain_policy_entries.active.json"
)
_BRAIN_POLICY_CACHE: Dict[str, pd.DataFrame] = {}
_ENTRY_BRAIN_POLICY_CACHE: Optional[pd.DataFrame] = None
_ENTRY_BRAIN_POLICY_CACHE_KEY: Optional[str] = None
_BRAIN_POLICY_REQUIRED_COLUMNS = [
    "strategy_id",
    "entry_model_id",
    "session",
    "macro",
    "vol",
    "trend",
    "liquidity",
    "tod",
    "prob_good",
    "expected_reward",
    "sample_size",
    "label",
]

_BRAIN_DEFAULT_CFG: Dict[str, Any] = {
    "enabled": True,
    "preferred_boost": 1.2,
    "discouraged_penalty": 0.5,
    "min_sample_size": 20,
    "min_prob_good": 0.55,
    "min_expected_reward": 0.0,
}

_DECISION_LOGGER = DecisionLogger(
    log_path=Path("logs/decision_log.jsonl"),
    schema_path=Path("schemas/decision_log.schema.json"),
)


def load_brain_policy(path: str | Path) -> pd.DataFrame:
    """Load a brain policy artifact (JSON or Parquet) with required columns.

    Deterministic, cached on disk path. Returns empty DataFrame on errors.
    """

    resolved = Path(path)
    cache_key = str(resolved.resolve()) if resolved.exists() else str(resolved)
    if cache_key in _BRAIN_POLICY_CACHE:
        return _BRAIN_POLICY_CACHE[cache_key]

    if not resolved.exists():
        logger.debug("Brain policy not found at %s", resolved)
        df = pd.DataFrame(columns=_BRAIN_POLICY_REQUIRED_COLUMNS)
        _BRAIN_POLICY_CACHE[cache_key] = df
        return df

    try:
        if resolved.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(resolved)
        else:
            with resolved.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict) and "policy" in payload:
                df = pd.DataFrame(payload.get("policy") or [])
            else:
                df = pd.DataFrame(payload)
    except Exception as exc:
        logger.debug("Failed to load brain policy %s: %s", resolved, exc)
        df = pd.DataFrame(columns=_BRAIN_POLICY_REQUIRED_COLUMNS)

    missing = [c for c in _BRAIN_POLICY_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.debug("Brain policy missing columns %s; ignoring", missing)
        df = pd.DataFrame(columns=_BRAIN_POLICY_REQUIRED_COLUMNS)
    else:
        df = df[_BRAIN_POLICY_REQUIRED_COLUMNS].copy()
        df = df.sort_values(_BRAIN_POLICY_REQUIRED_COLUMNS[:8]).reset_index(drop=True)

    _BRAIN_POLICY_CACHE[cache_key] = df
    return df


def _get_cached_brain_policy() -> pd.DataFrame:
    path = os.getenv("BRAIN_POLICY_PATH", str(_BRAIN_POLICY_DEFAULT_PATH))
    return load_brain_policy(path)


def _load_entry_brain_policy() -> pd.DataFrame:
    global _ENTRY_BRAIN_POLICY_CACHE, _ENTRY_BRAIN_POLICY_CACHE_KEY
    path = _ENTRY_BRAIN_POLICY_PATH

    target_path = path
    policy_payload = None
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                pointer_payload = json.load(handle)
            if isinstance(pointer_payload, dict) and pointer_payload.get("path"):
                target_path = Path(pointer_payload.get("path"))
            else:
                policy_payload = pointer_payload
        except Exception:
            policy_payload = None

    cache_key = f"{path.resolve()}::{target_path.resolve() if target_path else 'none'}"
    if (
        _ENTRY_BRAIN_POLICY_CACHE is not None
        and cache_key == _ENTRY_BRAIN_POLICY_CACHE_KEY
    ):
        return _ENTRY_BRAIN_POLICY_CACHE

    df = pd.DataFrame()
    if (
        policy_payload is not None
        and isinstance(policy_payload, dict)
        and "policy" in policy_payload
    ):
        df = pd.DataFrame(policy_payload.get("policy") or [])
    else:
        if target_path and target_path.exists():
            try:
                with target_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                policy = payload.get("policy") if isinstance(payload, dict) else payload
                df = pd.DataFrame(policy or [])
            except Exception:
                df = pd.DataFrame()

    if not df.empty:
        required_cols = [
            "entry_model_id",
            "market_profile_state",
            "session_profile",
            "liquidity_bias_side",
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        df = df.sort_values(required_cols).reset_index(drop=True)
    _ENTRY_BRAIN_POLICY_CACHE = df
    _ENTRY_BRAIN_POLICY_CACHE_KEY = cache_key
    return _ENTRY_BRAIN_POLICY_CACHE


def _load_brain_config() -> Dict[str, Any]:
    cfg = dict(_BRAIN_DEFAULT_CFG)
    cfg_path = Path("config/policy_config.json")
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg.update((data.get("brain") or {}))
        except Exception:
            logger.debug("Brain config load failed, using defaults", exc_info=True)
    # Clamp multipliers for safety
    cfg["preferred_boost"] = float(max(0.0, min(5.0, cfg.get("preferred_boost", 1.2))))
    cfg["discouraged_penalty"] = float(
        max(0.0, min(1.0, cfg.get("discouraged_penalty", 0.5)))
    )
    return cfg


def _safe_mode_active() -> bool:
    env_flag = str(os.getenv("SAFE_MODE", os.getenv("SAFE_MODE_ACTIVE", ""))).lower()
    if env_flag in {"1", "true", "on", "yes"}:
        return True
    try:
        marker = Path("logs/safe_mode_state.txt")
        if marker.exists():
            content = marker.read_text(encoding="utf-8").strip().upper()
            if "SAFE_MODE" in content:
                return True
    except Exception:
        pass
    return False


def _condition_vector_to_dict(condition_vector: Any) -> Dict[str, Any]:
    if condition_vector is None:
        return {}
    if hasattr(condition_vector, "__dict__"):
        try:
            from dataclasses import asdict

            return asdict(condition_vector)
        except Exception:
            return dict(condition_vector.__dict__)
    if isinstance(condition_vector, dict):
        return condition_vector
    return {}


def _extract_strategy_ids(
    state: Optional[Dict], market_state: Optional[Any]
) -> Tuple[Any, Any]:
    strategy_id = None
    entry_model_id = None
    if market_state is not None:
        strategy_id = getattr(market_state, "strategy_id", None)
        entry_model_id = getattr(market_state, "entry_model_id", None)
    if strategy_id is None and isinstance(state, dict):
        strategy_id = state.get("strategy_id")
    if entry_model_id is None and isinstance(state, dict):
        entry_model_id = state.get("entry_model_id")
    return strategy_id, entry_model_id


def _lookup_brain_recommendation(
    strategy_id: Any, entry_model_id: Any, condition_vector: Any
) -> Optional[Dict[str, Any]]:
    policy_df = _get_cached_brain_policy()
    if policy_df.empty:
        return None

    cv = _condition_vector_to_dict(condition_vector)
    if not cv:
        return None

    def _v(key: str) -> Any:
        return cv.get(key)

    mask = (
        (policy_df["strategy_id"] == strategy_id)
        & (policy_df["entry_model_id"] == entry_model_id)
        & (policy_df["session"] == _v("session"))
        & (policy_df["macro"] == _v("macro"))
        & (policy_df["vol"] == _v("vol"))
        & (policy_df["trend"] == _v("trend"))
        & (policy_df["liquidity"] == _v("liquidity"))
        & (policy_df["tod"] == _v("tod"))
    )

    matches = policy_df[mask]
    if matches.empty:
        return None

    row = matches.iloc[0]
    return {
        "brain_label": row.get("label"),
        "brain_prob_good": float(row.get("prob_good", 0.0) or 0.0),
        "brain_expected_reward": float(row.get("expected_reward", 0.0) or 0.0),
        "brain_sample_size": int(row.get("sample_size", 0) or 0),
    }


def _match_entry_policy(
    entry_id: str, frame: Optional[DecisionFrame]
) -> Optional[Dict[str, Any]]:
    df = _load_entry_brain_policy()
    if df.empty or not entry_id:
        return None

    mp_state = (frame.market_profile_state if frame else None) or "UNKNOWN"
    session_profile = (frame.session_profile if frame else None) or "UNKNOWN"
    liq_bias = None
    if frame and isinstance(frame.liquidity_frame, dict):
        liq_bias = frame.liquidity_frame.get("bias")
    liq_bias = liq_bias or "UNKNOWN"

    candidates = df[df["entry_model_id"] == entry_id]
    if candidates.empty:
        return None

    exact = candidates[
        (candidates["market_profile_state"] == mp_state)
        & (candidates["session_profile"] == session_profile)
        & (candidates["liquidity_bias_side"] == liq_bias)
    ]
    if not exact.empty:
        row = exact.iloc[0]
    else:
        row = candidates.iloc[0]

    return {
        "label": row.get("label"),
        "prob_good": float(row.get("prob_good", 0.0) or 0.0),
        "expected_reward": float(row.get("expected_reward", 0.0) or 0.0),
        "sample_size": int(row.get("sample_size", 0) or 0),
    }


def _attach_entry_brain_shadow(decision_frame: Optional[DecisionFrame]) -> None:
    if decision_frame is None:
        return
    eligible = decision_frame.eligible_entry_models or []
    if not eligible:
        return

    labels: Dict[str, Any] = {}
    scores: Dict[str, Any] = {}
    for entry_id in eligible:
        policy_row = _match_entry_policy(entry_id, decision_frame)
        if policy_row is None:
            continue
        labels[entry_id] = policy_row.get("label")
        scores[entry_id] = {
            "prob_good": policy_row.get("prob_good"),
            "expected_reward": policy_row.get("expected_reward"),
            "sample_size": policy_row.get("sample_size"),
        }
        logger.debug(
            "Shadow tactical brain: entry_id=%s label=%s prob_good=%.3f expected_R=%.3f",
            entry_id,
            policy_row.get("label"),
            float(policy_row.get("prob_good", 0.0) or 0.0),
            float(policy_row.get("expected_reward", 0.0) or 0.0),
        )

    if labels:
        decision_frame.entry_brain_labels = labels
        decision_frame.entry_brain_scores = scores


def _apply_shadow_brain_annotations(
    strategy_id: Any,
    entry_model_id: Any,
    condition_vector: Any,
    result_payload: Dict[str, Any],
    entry: Optional[Dict[str, Any]] = None,
) -> None:
    brain_info = _lookup_brain_recommendation(
        strategy_id, entry_model_id, condition_vector
    )
    if not brain_info:
        return

    result_payload.update(brain_info)
    if entry is not None:
        entry.update(brain_info)

    logger.debug(
        "Shadow brain: strategy_id=%s entry_model_id=%s label=%s prob_good=%.4f expected_reward=%.4f sample_size=%s",
        strategy_id,
        entry_model_id,
        brain_info.get("brain_label"),
        brain_info.get("brain_prob_good", 0.0),
        brain_info.get("brain_expected_reward", 0.0),
        brain_info.get("brain_sample_size"),
    )


def _apply_brain_weighting(
    score: float,
    brain_info: Optional[Dict[str, Any]],
    brain_cfg: Dict[str, Any],
    safe_mode: bool,
    risk_limit_hit: bool,
    policy_allowed: bool,
) -> Tuple[float, bool]:
    if not brain_cfg.get("enabled", True):
        return score, False
    if safe_mode:
        logger.debug("SAFE_MODE active — brain influence suppressed.")
        return score, False
    if risk_limit_hit:
        logger.debug("Risk limit hit — brain influence suppressed.")
        return score, False
    if not policy_allowed:
        logger.debug("Policy gating disallows strategy — brain influence suppressed.")
        return score, False
    if not brain_info:
        return score, False

    label = (brain_info.get("brain_label") or "").upper()
    sample_size = int(brain_info.get("brain_sample_size", 0) or 0)
    prob_good = float(brain_info.get("brain_prob_good", 0.0) or 0.0)
    expected_reward = float(brain_info.get("brain_expected_reward", 0.0) or 0.0)

    if (
        sample_size < brain_cfg.get("min_sample_size", 0)
        or prob_good < brain_cfg.get("min_prob_good", 0.0)
        or expected_reward < brain_cfg.get("min_expected_reward", 0.0)
    ):
        logger.debug(
            "Brain thresholds not met (sample=%s prob=%.4f reward=%.4f); treated as DISABLED.",
            sample_size,
            prob_good,
            expected_reward,
        )
        return 0.0, True

    score_before = score
    applied = False
    if label == "DISABLED":
        score = 0.0
        applied = True
        logger.debug("Brain disabled strategy_id: score set to 0.0")
    elif label == "DISCOURAGED":
        penalty = brain_cfg.get("discouraged_penalty", 0.5)
        score *= penalty
        applied = True
        logger.debug(
            "Brain influence: label=DISCOURAGED score_before=%.4f score_after=%.4f penalty=%.3f",
            score_before,
            score,
            penalty,
        )
    elif label == "PREFERRED":
        boost = brain_cfg.get("preferred_boost", 1.0)
        score *= boost
        applied = True
        logger.debug(
            "Brain influence: label=PREFERRED score_before=%.4f score_after=%.4f boost=%.3f",
            score_before,
            score,
            boost,
        )
    else:
        applied = False

    # Bound score
    score = float(np.clip(score, -1.0, 1.0))
    return score, applied


def _safe_get(source: Any, key: str, default: Any = None) -> Any:
    try:
        if source is None:
            return default
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _enrich_liquidity_dict(liq: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    liq = dict(liq or {})
    swept = liq.get("swept") or {}
    if "sweep_state" not in liq:
        liq["sweep_state"] = (
            "POST_SWEEP" if any(bool(v) for v in swept.values()) else "NO_SWEEP"
        )
    if "distance_bucket" not in liq:
        distances = liq.get("distances") or {}
        bucket = "UNKNOWN"
        if distances:
            try:
                finite = [float(v) for v in distances.values() if v is not None]
                if finite:
                    min_dist = min(finite)
                    if min_dist <= 0:
                        bucket = "INSIDE"
                    elif min_dist <= 5:
                        bucket = "NEAR"
                    else:
                        bucket = "FAR"
            except Exception:
                bucket = "UNKNOWN"
        liq["distance_bucket"] = bucket
    if "bias" in liq and liq.get("bias"):
        liq["bias"] = str(liq.get("bias")).upper()
    return liq


def _derive_entry_signals(
    decision_frame: DecisionFrame,
    state: Optional[Dict[str, Any]],
    market_state: Optional[Any],
) -> Dict[str, bool]:
    signals = {
        "sweep": False,
        "displacement": False,
        "fvg": False,
        "ob": False,
        "ifvg": False,
    }

    liq = decision_frame.liquidity_frame or {}
    swept = liq.get("swept") or {}
    if any(bool(v) for v in swept.values()):
        signals["sweep"] = True

    evidence = decision_frame.market_profile_evidence or {}
    sweeps_ev = evidence.get("sweeps") if isinstance(evidence, dict) else {}
    if isinstance(sweeps_ev, dict) and any(bool(v) for v in sweeps_ev.values()):
        signals["sweep"] = True

    displacement_score = None
    if isinstance(evidence, dict):
        displacement_score = evidence.get("displacement_score")

    structure_features: Dict[str, Any] = {}
    if isinstance(state, dict):
        structure_features = (
            state.get("structure_features") or state.get("features", {}) or {}
        )
    elif hasattr(market_state, "structure_features"):
        try:
            structure_features = getattr(market_state, "structure_features") or {}
        except Exception:
            structure_features = {}

    displacement_score = (
        displacement_score
        if displacement_score is not None
        else structure_features.get("displacement_score")
    )
    signals["displacement"] = bool(
        displacement_score is not None and displacement_score > 0.4
    )
    signals["fvg"] = bool(
        structure_features.get("fvg_created")
        or structure_features.get("fvg_present")
        or structure_features.get("fvg_respected")
    )
    signals["ob"] = bool(
        structure_features.get("ob_created")
        or structure_features.get("ob_present")
        or structure_features.get("ob_respected")
    )
    signals["ifvg"] = bool(
        structure_features.get("ifvg_created")
        or structure_features.get("ifvg_present")
        or structure_features.get("ifvg_respected")
    )

    return signals


def _build_market_profile_features(
    market_state: Optional[Any],
    legacy_state: Optional[Dict[str, Any]],
    session_context: Optional[Dict[str, Any]],
) -> Optional[MarketProfileFeatures]:
    existing = _safe_get(market_state, "market_profile_features") or _safe_get(
        legacy_state, "market_profile_features"
    )
    if isinstance(existing, MarketProfileFeatures):
        return existing

    features_src = (
        _safe_get(legacy_state, "structure_features", {})
        or _safe_get(market_state, "structure_features", {})
        or _safe_get(legacy_state, "features", {})
        or {}
    )

    timestamp_dt = datetime.now(timezone.utc)
    session_label = str(
        (session_context or {}).get("session_regime")
        or (session_context or {}).get("session")
        or _safe_get(legacy_state, "session_regime")
        or _safe_get(legacy_state, "session")
        or "UNKNOWN"
    )
    tod_bucket = str(
        features_src.get("time_of_day_bucket")
        or _safe_get(legacy_state, "time_of_day_bucket")
        or "UNKNOWN"
    )

    def _str_field(key: str, default: str = "NONE") -> str:
        return str(features_src.get(key, default) or default).upper()

    return MarketProfileFeatures(
        timestamp_utc=timestamp_dt,
        session_context=session_label,
        time_of_day_bucket=tod_bucket,
        dist_pdh=_safe_float(features_src.get("dist_pdh", 0.0)),
        dist_pdl=_safe_float(features_src.get("dist_pdl", 0.0)),
        dist_prev_session_high=_safe_float(
            features_src.get("dist_prev_session_high", 0.0)
        ),
        dist_prev_session_low=_safe_float(
            features_src.get("dist_prev_session_low", 0.0)
        ),
        dist_weekly_high=_safe_float(features_src.get("dist_weekly_high", 0.0)),
        dist_weekly_low=_safe_float(features_src.get("dist_weekly_low", 0.0)),
        nearest_draw_side=_str_field("nearest_draw_side", "NONE"),
        atr=_safe_float(features_src.get("atr", 0.0)),
        atr_vs_session_baseline=_safe_float(
            features_src.get("atr_vs_session_baseline", 0.0)
        ),
        realized_vol=_safe_float(features_src.get("realized_vol", 0.0)),
        intraday_range_vs_typical=_safe_float(
            features_src.get("intraday_range_vs_typical", 0.0)
        ),
        trend_slope_htf=_safe_float(features_src.get("trend_slope_htf", 0.0)),
        trend_dir_htf=_str_field("trend_dir_htf", "FLAT"),
        trend_slope_ltf=_safe_float(features_src.get("trend_slope_ltf", 0.0)),
        trend_dir_ltf=_str_field("trend_dir_ltf", "FLAT"),
        displacement_score=_safe_float(features_src.get("displacement_score", 0.0)),
        num_impulsive_bars=int(features_src.get("num_impulsive_bars", 0)),
        swept_pdh=_safe_bool(features_src.get("swept_pdh")),
        swept_pdl=_safe_bool(features_src.get("swept_pdl")),
        swept_session_high=_safe_bool(features_src.get("swept_session_high")),
        swept_session_low=_safe_bool(features_src.get("swept_session_low")),
        swept_equal_highs=_safe_bool(features_src.get("swept_equal_highs")),
        swept_equal_lows=_safe_bool(features_src.get("swept_equal_lows")),
        fvg_created=_safe_bool(features_src.get("fvg_created")),
        fvg_filled=_safe_bool(features_src.get("fvg_filled")),
        fvg_respected=_safe_bool(features_src.get("fvg_respected")),
        ob_created=_safe_bool(features_src.get("ob_created")),
        ob_respected=_safe_bool(features_src.get("ob_respected")),
        ob_violated=_safe_bool(features_src.get("ob_violated")),
        volume_spike=_safe_bool(features_src.get("volume_spike")),
        volume_vs_mean=_safe_float(features_src.get("volume_vs_mean", 0.0)),
    )


def _extract_liquidity_inputs(
    legacy_state: Optional[Dict[str, Any]], market_state: Optional[Any]
) -> Dict[str, Dict[str, Any]]:
    candidates = [
        _safe_get(legacy_state, "liquidity_draws"),
        _safe_get(legacy_state, "draws"),
        _safe_get(market_state, "liquidity_draws"),
        _safe_get(market_state, "draws"),
    ]
    for cand in candidates:
        if isinstance(cand, dict):
            return cand
    return {}


def _build_decision_frame(
    *,
    state: Optional[Dict[str, Any]],
    market_state: Optional[Any],
    session_context: Dict[str, Any],
    condition_vector: Any,
    decision_timestamp: str,
) -> Optional[DecisionFrame]:
    try:
        mp_features = _build_market_profile_features(
            market_state, state, session_context
        )
        ml_model: Optional[TrainedMarketProfileModel] = _safe_get(
            market_state, "market_profile_model"
        ) or _safe_get(state, "market_profile_model")
        state_machine: Optional[MarketProfileStateMachine] = _safe_get(
            market_state, "market_profile_state_machine"
        ) or _safe_get(state, "market_profile_state_machine")
        if state_machine is None:
            state_machine = MarketProfileStateMachine(thresholds={})

        mp_frame: Optional[MarketProfileFrame] = None
        if mp_features is not None and ml_model is not None:
            mp_result = classify_market_profile(mp_features, ml_model, state_machine)
            mp_frame = MarketProfileFrame(
                state=mp_result.get("state"),
                confidence=mp_result.get("confidence"),
                evidence=mp_result.get("evidence") or {},
            )

        session_features = (
            _safe_get(state, "session_profile_features", {})
            or _safe_get(market_state, "session_profile_features", {})
            or {}
        )
        session_profile_frame: Optional[SessionProfileFrame] = classify_session_profile(
            session_features
        )

        liquidity_inputs = _extract_liquidity_inputs(state, market_state)
        liquidity_frame = compute_liquidity_frame(liquidity_inputs)

        symbol = str(
            _safe_get(state, "symbol")
            or _safe_get(market_state, "symbol")
            or _safe_get(state, "market")
            or _safe_get(market_state, "market")
            or "UNKNOWN"
        )

        frame = DecisionFrame.from_frames(
            timestamp_utc=decision_timestamp,
            symbol=symbol,
            session_context=session_context,
            condition_vector=_condition_vector_to_dict(condition_vector),
            market_profile=mp_frame,
            session_profile=session_profile_frame,
            liquidity=liquidity_frame,
        )
        if frame and isinstance(frame.liquidity_frame, dict):
            frame.liquidity_frame = _enrich_liquidity_dict(frame.liquidity_frame)
        return frame
    except Exception:
        return None


# Integration imports (for causal + policy pipeline)
try:
    from engine.integration import (
        create_integrated_evaluator_factory,
        evaluate_and_decide,
    )

    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    logger.warning("Integration module not available - causal+policy pipeline disabled")


# ---------------------------------------------------------------------------
# Evaluator v1.3 (deterministic scaffolding with regime engine)
# ---------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def evaluate_state(state: MarketState) -> EvaluationOutput:
    """Deterministic evaluator v1.2 with regime classification.

    Steps:
    1) Base trend/momentum scoring (v1.1 logic)
    2) Regime classification via helpers (overrides state fields)
    3) Regime-aware adjustments to score/confidence
    4) Clamp score and derive confidence
    """

    # TODO(phase12): Surface strategy_id/entry_model_id/exit_model_id from registry once
    # strategy selection is enabled; currently placeholders live on EvaluationOutput only.
    # TODO(phase12): Attach condition_vector via condition_encoder.encode_conditions(state)
    # when the brain consumes it; no runtime logic changes in this layer.

    if not isinstance(state, MarketState):
        raise TypeError("state must be engine.types.MarketState")

    risk_flags: List[str] = []

    # Base scoring
    trend_score = 0.0
    if state.current_price > state.ma_short:
        trend_score += 0.3
    elif state.current_price < state.ma_short:
        trend_score -= 0.3

    if state.current_price > state.ma_long:
        trend_score += 0.2
    elif state.current_price < state.ma_long:
        trend_score -= 0.2

    momentum_score = 0.0
    if state.momentum > 0:
        momentum_score += 0.2
    elif state.momentum < 0:
        momentum_score -= 0.2

    total_score = trend_score + momentum_score
    score = clamp(total_score, -1.0, 1.0)

    # Regime bundle (overrides state-provided fields)
    regime = compute_regime_bundle(state)

    # Volatility sanity check
    avg_abs_returns = 0.0
    if state.recent_returns:
        avg_abs_returns = sum(abs(r) for r in state.recent_returns) / len(
            state.recent_returns
        )
    if avg_abs_returns > 0 and state.volatility > 2.0 * avg_abs_returns:
        risk_flags.append("high_volatility")

    # Regime-aware adjustments
    score += regime["trend_strength"] * 0.4
    score -= regime["liquidity_penalty"]
    score += regime["macro_bias"]

    amd_regime = regime.get("amd_regime", getattr(state, "amd_regime", "NEUTRAL"))
    amd_conf = getattr(state, "amd_confidence", 0.0)
    amd_adj = 0.0
    if amd_regime == "ACCUMULATION":
        amd_adj = min(0.12, 0.08 + 0.05 * amd_conf)
    elif amd_regime == "DISTRIBUTION":
        amd_adj = -min(0.12, 0.08 + 0.05 * amd_conf)
    elif amd_regime == "MANIPULATION":
        amd_adj = -min(0.18, 0.12 + 0.06 * (1.0 + amd_conf))
        risk_flags.append("amd_manipulation")
    score += amd_adj

    vol_shock = regime.get(
        "volatility_shock", getattr(state, "volatility_shock", False)
    )
    vol_shock_strength = float(getattr(state, "volatility_shock_strength", 0.0))
    if vol_shock:
        risk_flags.append("volatility_shock")
        score *= max(0.4, 1.0 - 0.5 * max(0.0, min(1.0, vol_shock_strength)))
        score -= 0.05 * vol_shock_strength

    prob_tilts = compute_probability_tilts(state)
    score += prob_tilts["total_probability_tilt"]

    score = clamp(score, -1.0, 1.0)

    confidence = min(1.0, abs(score))
    confidence *= 1.0 - regime["volatility_intensity"] * 0.5
    session_regime = regime.get("session_regime", getattr(state, "session", "UNKNOWN"))
    if session_regime == "ASIA":
        confidence *= 0.97
    elif session_regime == "NEW_YORK":
        confidence *= 1.02

    # Momentum / ROC tilts (bounded and deterministic)
    m20 = float(getattr(state, "momentum_20", 0.0))
    roc20 = float(getattr(state, "roc_20", 0.0))
    trend_dir = regime.get(
        "trend_direction", getattr(state, "trend_direction", "RANGE")
    )
    if trend_dir == "UP" and m20 > 0.01 and roc20 > 0.01:
        confidence *= 1.02
    elif trend_dir == "DOWN" and m20 < -0.01 and roc20 < -0.01:
        confidence *= 1.02

    depth_imbalance = float(getattr(state, "depth_imbalance", 0.0))
    if score > 0 and depth_imbalance < -0.3:
        confidence *= 0.97
    elif score < 0 and depth_imbalance > 0.3:
        confidence *= 0.97

    if vol_shock:
        confidence *= max(0.5, 1.0 - 0.4 * vol_shock_strength)
    if amd_regime == "MANIPULATION":
        confidence *= 0.8

    confidence = clamp(confidence, 0.0, 1.0)

    # Regime-based risk flags
    if regime["volatility_intensity"] > 0.8:
        risk_flags.append("extreme_volatility")
    if regime["liquidity_penalty"] > 0.2:
        risk_flags.append("thin_liquidity")
    if regime["macro_bias"] < -0.2:
        risk_flags.append("macro_headwind")

    confidence = clamp(confidence, 0.0, 1.0)

    return EvaluationOutput(
        score=score,
        confidence=confidence,
        trend_regime=regime["trend_regime"],
        volatility_regime=regime["volatility_regime"],
        liquidity_regime=regime["liquidity_regime"],
        macro_regime=regime["macro_regime"],
        risk_flags=risk_flags,
        veto_flags=[],
    )


# Decision types
class Decision(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


# Configuration thresholds
class EvaluatorConfig:
    # Microstructure config
    ENABLE_MICROSTRUCTURE = False  # Set True to enable microstructure logic
    MICROSTRUCTURE_SPREAD_KEY = "liquidity_metrics"  # Where to look for micro spread
    """Configuration for evaluator decision thresholds"""
    # Spread and liquidity
    MAX_SPREAD_PIPS = 3.0  # Don't trade if spread > 3 pips (EURUSD)
    MIN_SPREAD_PIPS = 0.5  # Don't trade if spread < 0.5 (likely stale)

    # Volatility
    MAX_VOLATILITY_PCT = 2.0  # Don't trade if volatility > 2% unless strong trend
    MIN_VOLATILITY_PCT = 0.01  # Don't trade if volatility < 0.01% (too quiet)
    HIGH_VOLATILITY_THRESHOLD = 1.5  # Requires stronger signal

    # Trend strength
    MIN_TREND_STRENGTH = 0.3  # Minimum confidence to trade a trend
    STRONG_TREND_STRENGTH = 0.7  # Strong enough to override other filters

    # RSI zones
    RSI_OVERSOLD = 30  # Buy signal zone
    RSI_OVERBOUGHT = 70  # Sell signal zone
    RSI_NEUTRAL_LOW = 40  # Lower neutral bound
    RSI_NEUTRAL_HIGH = 60  # Upper neutral bound

    # Sentiment
    SENTIMENT_WEIGHT = 0.15  # 15% of decision
    SENTIMENT_THRESHOLD = 0.3  # Minimum confidence to use sentiment

    # Data quality
    STALE_STATE_AGE_SEC = 5  # State > 5s old is stale
    MIN_CANDLES_REQUIRED = 20  # Need at least 20 candles for analysis

    # Multi-timeframe
    REQUIRE_HIGHER_TF_CONFIRMATION = True  # Require H1 confirmation for trades

    # Position management
    CLOSE_DECISION_RSI_THRESHOLD = 0.5  # Close if opposite RSI extreme reached
    CLOSE_DECISION_PROFIT_RATIO = 0.7  # Partial close at profit targets


class EvaluatorError(Exception):
    """Base exception for evaluator errors"""

    pass


class SafetyCheckError(EvaluatorError):
    """Raised when safety checks fail"""

    pass


def check_state_safety(state: Dict) -> Tuple[bool, List[str]]:
    """
    Perform safety checks on state before evaluation.

    Args:
        state: Market state dictionary from state_builder

    Returns:
        Tuple of (is_safe: bool, errors: list of warning messages)
    """
    errors = []

    # Check state exists
    if state is None:
        errors.append("State is None")
        return False, errors

    # Check state is not stale
    if state.get("health", {}).get("is_stale", False):
        errors.append("State data is stale")

    # Check for data health errors
    health_errors = state.get("health", {}).get("errors", [])
    if health_errors:
        errors.extend(health_errors)

    # Check timestamp exists and is recent
    timestamp = state.get("timestamp")
    if timestamp is None:
        errors.append("Missing state timestamp")

    # Check tick data
    tick = state.get("tick")
    if not tick or "bid" not in tick or "ask" not in tick:
        errors.append("Missing or invalid tick data")
        return False, errors

    # Check indicators
    indicators = state.get("indicators", {})
    required_indicators = ["rsi_14", "sma_50", "sma_200", "atr_14", "volatility"]
    missing_indicators = [
        ind for ind in required_indicators if indicators.get(ind) is None
    ]
    if missing_indicators:
        errors.append(f"Missing indicators: {missing_indicators}")

    # Check candles
    candles = state.get("candles", {})
    if not candles or "H1" not in candles or candles["H1"] is None:
        errors.append("Missing H1 candle data")

    # Check trend data
    trend = state.get("trend", {})
    if "regime" not in trend or "strength" not in trend:
        errors.append("Missing trend data")

    is_safe = len(errors) == 0
    return is_safe, errors


def check_spread_filter(
    state: Dict, use_microstructure: bool = False
) -> Tuple[bool, str]:
    """
    Check if spread is within acceptable range for trading.

    Args:
        state: Market state dictionary

    Returns:
        Tuple of (pass_filter: bool, reason: str)
    """
    if (
        use_microstructure
        and "liquidity_metrics" in state
        and state["liquidity_metrics"]
    ):
        spread = state["liquidity_metrics"].get("spread", float("inf"))
        liquidity_score = state["liquidity_metrics"].get("liquidity_score", 0)
        stress_flags = state["liquidity_metrics"].get("stress_flags", [])
        if spread > EvaluatorConfig.MAX_SPREAD_PIPS:
            return (
                False,
                f"Microstructure: Spread too wide: {spread:.2f} pips (max: {EvaluatorConfig.MAX_SPREAD_PIPS})",
            )
        if spread < EvaluatorConfig.MIN_SPREAD_PIPS:
            return (
                False,
                f"Microstructure: Spread too tight: {spread:.2f} pips (likely stale data)",
            )
        if "low_liquidity" in stress_flags:
            return (
                False,
                f"Microstructure: Low liquidity detected (score={liquidity_score})",
            )
        return (
            True,
            f"Microstructure: Spread OK: {spread:.2f} pips, liquidity_score={liquidity_score}",
        )
    else:
        spread = state.get("tick", {}).get("spread", float("inf"))
        if spread > EvaluatorConfig.MAX_SPREAD_PIPS:
            return (
                False,
                f"Spread too wide: {spread:.2f} pips (max: {EvaluatorConfig.MAX_SPREAD_PIPS})",
            )
        if spread < EvaluatorConfig.MIN_SPREAD_PIPS:
            return False, f"Spread too tight: {spread:.2f} pips (likely stale data)"
        return True, f"Spread OK: {spread:.2f} pips"


def check_volatility_filter(state: Dict, trend_strength: float) -> Tuple[bool, str]:
    """
    Check if volatility is within acceptable range.
    Allows higher volatility during strong trends.

    Args:
        state: Market state dictionary
        trend_strength: Trend confidence (0-1)

    Returns:
        Tuple of (pass_filter: bool, reason: str)
    """
    volatility = state.get("indicators", {}).get("volatility", 0)

    if volatility < EvaluatorConfig.MIN_VOLATILITY_PCT:
        return False, f"Volatility too low: {volatility:.3f}% (market too quiet)"

    if volatility > EvaluatorConfig.MAX_VOLATILITY_PCT:
        # Allow high volatility if trend is strong
        if trend_strength < EvaluatorConfig.STRONG_TREND_STRENGTH:
            return False, f"Volatility too high: {volatility:.3f}% and trend weak"
        else:
            logger.info(
                f"High volatility {volatility:.3f}% but trend strong ({trend_strength:.2f}), allowing"
            )
            return True, f"High volatility OK due to strong trend"

    return True, f"Volatility normal: {volatility:.3f}%"


def check_multitimeframe_alignment(state: Dict) -> Tuple[str, float]:
    """
    Check alignment across multiple timeframes (M1, M5, M15, H1).

    Returns signal direction ('buy', 'sell', 'hold') with confidence.

    Args:
        state: Market state dictionary

    Returns:
        Tuple of (signal: str, confidence: float)
    """
    candles = state.get("candles", {})
    signals = {}
    confidences = {}

    # Analyze each timeframe
    for tf in ["M1", "M5", "M15", "H1"]:
        if candles.get(tf) is None:
            logger.warning(f"Missing {tf} candles for multi-timeframe analysis")
            signals[tf] = "hold"
            confidences[tf] = 0.0
            continue

        tf_candle = candles[tf]
        tf_indicators = tf_candle.get("indicators", {})

        # Get RSI for this timeframe
        rsi = tf_indicators.get("rsi_14")
        if rsi is None:
            signals[tf] = "hold"
            confidences[tf] = 0.0
            continue

        # Generate signal from RSI
        if rsi < EvaluatorConfig.RSI_OVERSOLD:
            signals[tf] = "buy"
            confidences[tf] = (EvaluatorConfig.RSI_OVERSOLD - rsi) / 30.0  # 0-1
        elif rsi > EvaluatorConfig.RSI_OVERBOUGHT:
            signals[tf] = "sell"
            confidences[tf] = (rsi - EvaluatorConfig.RSI_OVERBOUGHT) / 30.0
        else:
            signals[tf] = "hold"
            confidences[tf] = 0.0

        logger.debug(
            f"{tf} RSI: {rsi:.1f} → {signals[tf]} (confidence: {confidences[tf]:.2f})"
        )

    # Aggregate signals (require H1 agreement if configured)
    buy_votes = sum(1 for s in signals.values() if s == "buy")
    sell_votes = sum(1 for s in signals.values() if s == "sell")

    logger.debug(
        f"Multi-TF votes: BUY={buy_votes}, SELL={sell_votes}, HOLD={4-buy_votes-sell_votes}"
    )

    # Determine consensus signal
    if EvaluatorConfig.REQUIRE_HIGHER_TF_CONFIRMATION:
        # Require H1 to align
        if signals.get("H1") == "buy" and buy_votes >= 2:
            return "buy", min(0.9, (buy_votes / 4.0) * 0.9)
        elif signals.get("H1") == "sell" and sell_votes >= 2:
            return "sell", min(0.9, (sell_votes / 4.0) * 0.9)
        else:
            return "hold", 0.0
    else:
        # Any alignment
        if buy_votes > sell_votes:
            return "buy", (buy_votes / 4.0)
        elif sell_votes > buy_votes:
            return "sell", (sell_votes / 4.0)
        else:
            return "hold", 0.0


def calculate_signal_confidence(
    trend_signal: str,
    trend_strength: float,
    multitf_signal: str,
    multitf_confidence: float,
    sentiment_score: float,
    sentiment_confidence: float,
) -> float:
    """
    Calculate overall confidence score for the trading signal.

    Combines trend, multi-timeframe, and sentiment signals.

    Args:
        trend_signal: 'buy', 'sell', or 'hold'
        trend_strength: Trend confidence (0-1)
        multitf_signal: Multi-timeframe signal
        multitf_confidence: Multi-timeframe confidence (0-1)
        sentiment_score: News sentiment (-1 to 1)
        sentiment_confidence: Sentiment confidence (0-1)

    Returns:
        Overall confidence (0-1)
    """
    confidence = 0.0

    # Trend contribution (40%)
    if trend_signal in ["buy", "sell"]:
        confidence += trend_strength * 0.4

    # Multi-timeframe contribution (45%)
    if multitf_signal == trend_signal:
        confidence += multitf_confidence * 0.45
    elif multitf_signal == "hold":
        confidence += multitf_confidence * 0.20  # Weak support

    # Sentiment contribution (15%)
    if sentiment_confidence > EvaluatorConfig.SENTIMENT_THRESHOLD:
        if trend_signal == "buy" and sentiment_score > 0:
            confidence += min(sentiment_score, 1.0) * EvaluatorConfig.SENTIMENT_WEIGHT
        elif trend_signal == "sell" and sentiment_score < 0:
            confidence += (
                abs(min(sentiment_score, -1.0)) * EvaluatorConfig.SENTIMENT_WEIGHT
            )

    return min(confidence, 1.0)


def evaluate_close_signal(
    state: Dict, open_position: Optional[Dict] = None
) -> Tuple[str, float]:
    """
    Determine if an open position should be closed.

    Args:
        state: Market state dictionary
        open_position: Optional dict with position info:
            {'direction': 'buy' or 'sell', 'entry_price': float, 'current_pnl_pct': float}

    Returns:
        Tuple of (decision: 'close' or 'hold', confidence: float)
    """
    if open_position is None:
        return "hold", 0.0

    rsi = state.get("indicators", {}).get("rsi_14")
    if rsi is None:
        return "hold", 0.0

    direction = open_position.get("direction")

    # Close if opposite extreme is reached
    if direction == "buy" and rsi > EvaluatorConfig.RSI_OVERBOUGHT:
        logger.info(f"Close signal: Buy position, RSI overbought at {rsi:.1f}")
        return "close", min(0.8, (rsi - EvaluatorConfig.RSI_OVERBOUGHT) / 30.0)

    if direction == "sell" and rsi < EvaluatorConfig.RSI_OVERSOLD:
        logger.info(f"Close signal: Sell position, RSI oversold at {rsi:.1f}")
        return "close", min(0.8, (EvaluatorConfig.RSI_OVERSOLD - rsi) / 30.0)

    return "hold", 0.0


def evaluate(
    state: Dict,
    open_position: Optional[Dict] = None,
    require_high_confidence: bool = False,
    enable_microstructure: Optional[bool] = None,
) -> Dict:
    """
    Main evaluation function - returns trading decision.

    Implements multi-layered decision logic:
    1. Safety checks
    2. Trend detection
    3. Multi-timeframe confirmation
    4. Volatility and spread filters
    5. Sentiment weighting
    6. Position management (close vs new entry)

    Args:
        state: Market state dictionary from state_builder
        open_position: Optional dict with position info for close evaluation
        require_high_confidence: If True, only return buy/sell for confidence > 0.6

    Returns:
        Dict with:
        {
            'decision': 'buy' | 'sell' | 'hold' | 'close',
            'confidence': float (0-1),
            'reason': str (explanation),
            'details': dict (internal analysis details),
        }
    """
    if canonical_enforced() and not getattr(test_flags, "CANONICAL_TEST_BYPASS", False):
        # Legacy evaluator must not be reachable in canonical/official modes
        raise ValueError(
            "Legacy evaluator is forbidden in canonical/official modes; use CausalEvaluator."
        )
    logger.info("=" * 60)
    logger.info("EVALUATION STARTED")
    logger.info("=" * 60)

    decision = Decision.HOLD
    confidence = 0.0
    reason = ""
    details = {}

    # LAYER 1: Safety Checks
    # ============================================================
    logger.info("\n[LAYER 1] Safety Checks")
    is_safe, safety_errors = check_state_safety(state)

    if not is_safe:
        reason = f"Safety check failed: {'; '.join(safety_errors)}"
        logger.warning(f"❌ {reason}")
        details["safety_errors"] = safety_errors
        return {
            "decision": Decision.HOLD.value,
            "confidence": 0.0,
            "reason": reason,
            "details": details,
        }

    logger.info("✓ Safety checks passed")

    # LAYER 2: Spread and Liquidity Filter
    # ============================================================
    logger.info("\n[LAYER 2] Spread Filter")
    # Determine if microstructure is enabled (config or override)
    use_micro = (
        enable_microstructure
        if enable_microstructure is not None
        else getattr(EvaluatorConfig, "ENABLE_MICROSTRUCTURE", False)
    )
    spread_pass, spread_reason = check_spread_filter(
        state, use_microstructure=use_micro
    )
    details["spread_check"] = spread_reason
    if use_micro:
        details["spread"] = state.get("spread")
        details["liquidity_score"] = state.get("liquidity_score")
        details["liquidity_stress_flags"] = state.get("liquidity_stress_flags")
        details["order_flow_features"] = state.get("order_flow_features")
    if not spread_pass:
        logger.warning(f"❌ {spread_reason}")
        reason = f"Liquidity insufficient: {spread_reason}"
        return {
            "decision": Decision.HOLD.value,
            "confidence": 0.0,
            "reason": reason,
            "details": details,
        }
    logger.info(f"✓ {spread_reason}")
    # LAYER 3.5: Microstructure EV/Risk/Cost Adjustments
    if use_micro:
        spread = state.get("spread", 0)
        liquidity_score = state.get("liquidity_score", 0)
        order_flow = state.get("order_flow_features", {})
        ev_penalty = 0.0
        risk_penalty = 0.0
        if spread > 2.0:
            ev_penalty += 0.1 * (spread - 2.0)
            risk_penalty += 0.1 * (spread - 2.0)
        if liquidity_score < 20:
            ev_penalty += 0.1
            risk_penalty += 0.1
        if order_flow.get("spoofing_score", 0) > 0:
            ev_penalty += 0.05 * order_flow["spoofing_score"]
            risk_penalty += 0.05 * order_flow["spoofing_score"]
        if order_flow.get("quote_pulling_score", 0) > 0:
            risk_penalty += 0.05 * order_flow["quote_pulling_score"]
        if order_flow.get("net_imbalance", 0) > 2:
            ev_penalty -= 0.05
        ev_penalty = min(max(ev_penalty, -0.2), 0.5)
        risk_penalty = min(max(risk_penalty, 0), 0.5)
        details["micro_ev_penalty"] = ev_penalty
        details["micro_risk_penalty"] = risk_penalty

    # LAYER 3: Extract core indicators
    # ============================================================
    logger.info("\n[LAYER 3] Extract Indicators")
    indicators = state.get("indicators", {})
    rsi = indicators.get("rsi_14")
    sma_50 = indicators.get("sma_50")
    sma_200 = indicators.get("sma_200")
    atr = indicators.get("atr_14")
    volatility = indicators.get("volatility", 0)

    logger.debug(
        f"RSI: {rsi:.1f}, SMA50: {sma_50:.4f}, SMA200: {sma_200:.4f}, ATR: {atr:.4f}, Vol: {volatility:.3f}%"
    )

    # LAYER 3.5: Adaptive Factor Weights (v4.0‑E)
    volatility_state = state.get("volatility_state", {})
    regime_state = state.get("regime_state", {})
    vol_regime = volatility_state.get("vol_regime", "NORMAL")
    liq_regime = regime_state.get("liq_regime", "NORMAL")
    macro_regime = regime_state.get("macro_regime", "RISK_ON")

    # Default weights
    trend_weight = 1.0
    order_flow_weight = 1.0
    liquidity_weight = 1.0
    volatility_weight = 1.0
    macro_weight = 1.0
    long_bias_weight = 1.0

    # Regime-conditioned adjustments
    if vol_regime in ["HIGH", "EXTREME"]:
        trend_weight *= 0.5
        liquidity_weight *= 1.5
    if liq_regime in ["THIN", "FRAGILE"]:
        order_flow_weight *= 0.7
        liquidity_weight *= 1.3
    if macro_regime == "RISK_OFF":
        long_bias_weight *= 0.5

    details["adaptive_weights"] = {
        "trend": trend_weight,
        "order_flow": order_flow_weight,
        "liquidity": liquidity_weight,
        "volatility": volatility_weight,
        "macro": macro_weight,
        "long_bias": long_bias_weight,
    }

    # Apply ABS break-level balancing so no factor can dominate.
    balanced_weights = BALANCE_CONTROLLER.balance_weights(
        details["adaptive_weights"],
        regime_state=regime_state,
        volatility_state=volatility_state,
    )
    details["balanced_weights"] = balanced_weights

    # Optional ML auxiliary hints (advisory only, passed through balancing).
    ml_hints_raw = compute_ml_hints(state)
    ml_hints_balanced = BALANCE_CONTROLLER.balance_weights(
        {
            "volatility": ml_hints_raw.get("vol_cluster_hint", 0.0),
            "macro": ml_hints_raw.get("macro_vol_hint", 0.0),
        },
        regime_state=regime_state,
        volatility_state=volatility_state,
    )
    details["ml_hints"] = {
        "raw": ml_hints_raw,
        "balanced": ml_hints_balanced,
    }

    # LAYER 4: Volatility Filter
    # ============================================================
    logger.info("\n[LAYER 4] Volatility Filter")
    trend = state.get("trend", {})
    trend_strength = trend.get("strength", 0)
    volatility_pass, volatility_reason = check_volatility_filter(state, trend_strength)
    details["volatility_check"] = volatility_reason

    if not volatility_pass:
        logger.warning(f"❌ {volatility_reason}")
        reason = f"Volatility condition failed: {volatility_reason}"
        return {
            "decision": Decision.HOLD.value,
            "confidence": 0.0,
            "reason": reason,
            "details": details,
        }

    logger.info(f"✓ {volatility_reason}")

    # LAYER 5: Trend Detection
    # ============================================================
    logger.info("\n[LAYER 5] Trend Detection")
    trend_regime = trend.get("regime", "sideways")
    logger.info(
        f"Trend Regime: {trend_regime.upper()} (strength: {trend_strength:.2f})"
    )

    # Generate base trend signal
    trend_signal = "hold"
    if (
        trend_regime == "uptrend"
        and trend_strength >= EvaluatorConfig.MIN_TREND_STRENGTH
    ):
        trend_signal = "buy"
        logger.info(f"→ Bullish trend signal (strength: {trend_strength:.2f})")
    elif (
        trend_regime == "downtrend"
        and trend_strength >= EvaluatorConfig.MIN_TREND_STRENGTH
    ):
        trend_signal = "sell"
        logger.info(f"→ Bearish trend signal (strength: {trend_strength:.2f})")
    else:
        logger.info(f"→ Insufficient trend strength ({trend_strength:.2f})")

    details["trend"] = {
        "regime": trend_regime,
        "strength": trend_strength,
        "signal": trend_signal,
    }

    # LAYER 6: Multi-Timeframe Confirmation
    # ============================================================
    logger.info("\n[LAYER 6] Multi-Timeframe Confirmation")
    multitf_signal, multitf_confidence = check_multitimeframe_alignment(state)
    logger.info(
        f"Multi-TF Signal: {multitf_signal.upper()} (confidence: {multitf_confidence:.2f})"
    )

    details["multitf"] = {
        "signal": multitf_signal,
        "confidence": multitf_confidence,
    }

    # LAYER 7: Sentiment Analysis
    # ============================================================
    logger.info("\n[LAYER 7] Sentiment Analysis")
    sentiment = state.get("sentiment", {})
    sentiment_score = sentiment.get("score", 0)
    sentiment_confidence = sentiment.get("confidence", 0)
    logger.info(
        f"Sentiment: {sentiment_score:.2f} (confidence: {sentiment_confidence:.2f})"
    )

    details["sentiment"] = {
        "score": sentiment_score,
        "confidence": sentiment_confidence,
        "source": sentiment.get("source", "unknown"),
    }

    # LAYER 8: Position Management (Close Check)
    # ============================================================
    logger.info("\n[LAYER 8] Position Management")
    if open_position:
        close_decision, close_confidence = evaluate_close_signal(state, open_position)
        if close_decision == "close":
            logger.info(f"✓ CLOSE DECISION (confidence: {close_confidence:.2f})")
            return {
                "decision": Decision.CLOSE.value,
                "confidence": close_confidence,
                "reason": f"Close position signal triggered",
                "details": {**details, "close_confidence": close_confidence},
            }

    # LAYER 9: Final Decision Logic
    # ============================================================
    logger.info("\n[LAYER 9] Final Decision")

    # Combine signals
    if trend_signal in ["buy", "sell"]:
        if trend_signal == multitf_signal or multitf_signal == "hold":
            # Aligned or multi-tf doesn't contradict
            decision_value = trend_signal
            confidence = calculate_signal_confidence(
                trend_signal,
                trend_strength,
                multitf_signal,
                multitf_confidence,
                sentiment_score,
                sentiment_confidence,
            )

            # Apply ABS balance confidence guard so capped weights cannot over-inflate confidence.
            confidence, balance_scale = BALANCE_CONTROLLER.apply_confidence_guard(
                confidence, balanced_weights
            )
            details["balance_scale"] = balance_scale

            # Advisory ML hints are recorded but cannot alter decisions/confidence.
            ml_conf_adj = 0.0
            details["ml_conf_adj"] = ml_conf_adj
            assert_ml_advisory_only(advisory_adjustment_applied=False)

            if confidence < EvaluatorConfig.MIN_TREND_STRENGTH:
                logger.info(
                    f"Signal too weak: {confidence:.2f} < {EvaluatorConfig.MIN_TREND_STRENGTH}"
                )
                decision_value = "hold"
        else:
            # Conflicting signals
            logger.warning(
                f"Conflicting signals: trend={trend_signal}, multitf={multitf_signal}"
            )
            decision_value = "hold"
            confidence = 0.0
    else:
        decision_value = "hold"

    # Apply high confidence requirement if needed
    if (
        require_high_confidence
        and decision_value in ["buy", "sell"]
        and confidence < 0.6
    ):
        logger.info(
            f"Confidence {confidence:.2f} below high-confidence threshold (0.6)"
        )
        decision_value = "hold"
        confidence = 0.0

    decision = Decision(decision_value)

    # Generate reason
    if decision == Decision.BUY:
        reason = f"BUY signal: {trend_regime} trend (strength: {trend_strength:.2f}), confirmed by multi-timeframe"
    elif decision == Decision.SELL:
        reason = f"SELL signal: {trend_regime} trend (strength: {trend_strength:.2f}), confirmed by multi-timeframe"
    else:
        reason = "HOLD: Insufficient signal strength or conflicting indicators"

    logger.info(f"\n{'='*60}")
    logger.info(f"DECISION: {decision.value.upper()} (confidence: {confidence:.2f})")
    logger.info(f"REASON: {reason}")
    logger.info(f"{'='*60}\n")

    details["final_confidence"] = confidence

    return {
        "decision": decision.value,
        "confidence": confidence,
        "reason": reason,
        "details": details,
    }


def evaluate_bulk(
    states: Dict[str, Dict],
    open_positions: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict]:
    """
    Evaluate multiple symbols in bulk (for multiple trading pairs).

    Args:
        states: Dict of {symbol: state_dict}
        open_positions: Optional dict of {symbol: position_dict}

    Returns:
        Dict of {symbol: evaluation_result}
    """
    results = {}
    open_positions = open_positions or {}
    for symbol, state in states.items():
        # Extract new volatility and regime states
        volatility_state = state.get("volatility_state", {})
        regime_state = state.get("regime_state", {})

        # Example: Use vol_regime and regime_transition in decision
        vol_regime = volatility_state.get("vol_regime", "NORMAL")
        regime_transition = regime_state.get("regime_transition", False)
        regime_confidence = regime_state.get("regime_confidence", 0.0)

        # Existing evaluation logic (simplified for illustration)
        # You may want to call your main evaluation function here
        result = {
            "decision": "hold",
            "confidence": regime_confidence,
            "reason": f"Volatility regime: {vol_regime}, Regime transition: {regime_transition}",
            "details": {
                "volatility_state": volatility_state,
                "regime_state": regime_state,
            },
        }
        results[symbol] = result
    return results

    for symbol, state in states.items():
        position = open_positions.get(symbol)
        results[symbol] = evaluate(state, open_position=position)

    return results


# ============================================================================
# CAUSAL EVALUATOR INTEGRATION
# ============================================================================


def evaluate_with_causal(
    state: Dict,
    causal_evaluator: Optional[Any] = None,
    market_state: Optional[Any] = None,
    policy: Optional[Any] = None,
) -> Dict:
    """
    Evaluate market state using CausalEvaluator (Stockfish-style).

    This integrates the deterministic, rule-based CausalEvaluator which combines
    8 market factors into a single evaluation score [-1, +1].

    Args:
        state: Traditional market state dictionary (legacy format)
        causal_evaluator: Initialized CausalEvaluator instance
        market_state: CausalEvaluator MarketState dataclass (preferred)
        policy: Optional policy config (PolicyConfig) for feature weights/trust

    Returns:
        Dict with decision, confidence, reason, and full causal reasoning

    Raises:
        ValueError: If causal_evaluator is None or market_state not properly configured
    """
    if causal_evaluator is None:
        raise ValueError("causal_evaluator cannot be None")

    if market_state is None:
        raise ValueError(
            "market_state (CausalEvaluator.MarketState) required for causal evaluation"
        )

    # Evaluate using CausalEvaluator
    try:
        result = causal_evaluator.evaluate(market_state)
    except Exception as e:
        logger.error(f"CausalEvaluator failed: {e}")
        raise

    def _derive_regimes(ms: Any) -> List[str]:
        regimes: List[str] = []
        try:
            session_label = getattr(ms, "session", None) or getattr(
                ms, "session_regime", None
            )
            if session_label:
                regimes.append(str(session_label))

            vol_state = getattr(ms, "volatility_state", None)
            if vol_state is not None:
                vol_reg = getattr(vol_state, "regime", None)
                if vol_reg is not None:
                    regimes.append(str(getattr(vol_reg, "name", vol_reg)))

            macro_news = getattr(ms, "macro_news_state", None)
            macro_label = (
                getattr(macro_news, "macro_news_state", None) if macro_news else None
            )
            if macro_label:
                upper_label = str(macro_label).upper()
                if "RISK_ON" in upper_label:
                    regimes.append("MACRO_ON")
                elif "RISK_OFF" in upper_label:
                    regimes.append("MACRO_OFF")

            macro_state = getattr(ms, "macro_state", None)
            sentiment = (
                getattr(macro_state, "sentiment_score", None) if macro_state else None
            )
            if sentiment is not None:
                if sentiment > 0.2:
                    regimes.append("MACRO_ON")
                elif sentiment < -0.2:
                    regimes.append("MACRO_OFF")
        except Exception:
            pass

        dedup: List[str] = []
        for r in regimes:
            if r and r not in dedup:
                dedup.append(str(r))
        return dedup

    def _extract_session_context(
        ms: Any, legacy_state: Optional[Dict]
    ) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {"session": "UNKNOWN", "modifiers": {}}
        try:
            session_label = (
                getattr(ms, "session", None)
                or getattr(ms, "session_regime", None)
                or (legacy_state or {}).get("session_regime")
                or (legacy_state or {}).get("session")
            )
            session_str = str(session_label or "UNKNOWN")
            ctx["session"] = session_str
            ctx["session_regime"] = session_str

            # Prefer explicit session_context->modifiers shape, then flat session_modifiers, then legacy
            sc = getattr(ms, "session_context", None)
            if isinstance(sc, dict) and sc.get("modifiers"):
                ctx["modifiers"] = sc.get("modifiers", {})
            else:
                mods = getattr(ms, "session_modifiers", None)
                if isinstance(mods, dict):
                    ctx["modifiers"] = mods
                else:
                    ctx["modifiers"] = (
                        (legacy_state or {})
                        .get("session_context", {})
                        .get("modifiers", {})
                    )
        except Exception:
            ctx = {"session": "UNKNOWN", "modifiers": {}}
        return ctx

    def _print_session_trace(
        session_ctx: Dict[str, Any], regimes: List[str], factors: List[Dict[str, Any]]
    ) -> None:
        try:
            mods = session_ctx.get("modifiers") or {}
            print("\n[SESSION CONTEXT]")
            print(
                f"  session_regime={session_ctx.get('session_regime', 'UNKNOWN')} | "
                f"vol_scale={mods.get('volatility_scale', 1.0):.2f} "
                f"liq_scale={mods.get('liquidity_scale', 1.0):.2f} "
                f"trade_scale={mods.get('trade_freq_scale', 1.0):.2f} "
                f"risk_scale={mods.get('risk_scale', 1.0):.2f}"
            )
            print(f"  regimes={', '.join(regimes) if regimes else '<none>'}")
            if factors:
                print("[SESSION WEIGHTS]")
                for f in factors:
                    print(
                        "  "
                        f"{f.get('factor', ''):18s} base={f.get('policy_base_weight', 1.0):.3f} "
                        f"trust={f.get('trust_score', 1.0):.3f} "
                        f"regime_mult={f.get('regime_multiplier', 1.0):.3f} "
                        f"session_mult={f.get('session_multiplier', 1.0):.3f} "
                        f"eff={f.get('policy_weight', 1.0):.3f} "
                        f"raw={f.get('raw_score', 0.0):.3f} "
                        f"weighted={f.get('weighted_score', 0.0):.3f}"
                    )
        except Exception:
            # Terminal trace is best-effort only
            pass

    # Convert causal evaluation to decision
    eval_score = result.eval_score
    confidence = result.confidence

    # Interpret evaluation as decision
    if eval_score > 0.2:
        decision = "buy"
        reason = f"CausalEvaluator BULLISH (score: {eval_score:.3f})"
    elif eval_score < -0.2:
        decision = "sell"
        reason = f"CausalEvaluator BEARISH (score: {eval_score:.3f})"
    else:
        decision = "hold"
        reason = f"CausalEvaluator NEUTRAL (score: {eval_score:.3f})"

    # Build reasoning from causal factors
    factor_explanations = []

    # Apply policy weights/trust if provided (factor names are used as keys)
    session_context = _extract_session_context(market_state, state)
    policy_applied = policy is not None
    policy_factors = []
    regimes: List[str] = _derive_regimes(market_state) if market_state else []
    if policy_applied:
        # deterministic ordering by factor name
        sorted_factors = sorted(result.scoring_factors, key=lambda f: f.factor_name)
        eval_score = 0.0
        session_mods = session_context.get("modifiers") or {}
        session_multiplier = float(session_mods.get("risk_scale", 1.0))
        for factor in sorted_factors:
            trust = policy.get_trust(factor.factor_name) if policy else 1.0
            base_weight = policy.get_base_weight(factor.factor_name) if policy else 1.0
            regime_multiplier = (
                policy.get_regime_multiplier(factor.factor_name, regimes)
                if policy
                else 1.0
            )
            policy_effective_weight = (
                base_weight * trust * regime_multiplier * session_multiplier
            )
            combined_weight = factor.weight * policy_effective_weight
            raw_score = factor.score
            weighted_score = 0.0 if trust == 0 else raw_score * combined_weight
            eval_score += weighted_score
            policy_factors.append(
                {
                    "factor": factor.factor_name,
                    "raw_score": raw_score,
                    "weight": factor.weight,
                    "policy_base_weight": base_weight,
                    "policy_weight": policy_effective_weight,
                    "regime_multiplier": regime_multiplier,
                    "session_multiplier": session_multiplier,
                    "trust_score": trust,
                    "weighted_score": weighted_score,
                    "explanation": factor.explanation,
                }
            )
        eval_score = float(np.clip(eval_score, -1.0, 1.0))
        _print_session_trace(session_context, regimes, policy_factors)
    else:
        sorted_factors = result.scoring_factors
        eval_score = result.eval_score
        for factor in sorted_factors:
            policy_factors.append(
                {
                    "factor": factor.factor_name,
                    "raw_score": factor.score,
                    "weight": factor.weight,
                    "policy_base_weight": 1.0,
                    "policy_weight": 1.0,
                    "regime_multiplier": 1.0,
                    "trust_score": 1.0,
                    "weighted_score": factor.score * factor.weight,
                    "explanation": factor.explanation,
                }
            )

    for pf in policy_factors:
        factor_explanations.append(
            {
                "factor": pf["factor"],
                "score": pf["raw_score"],
                "weight": pf["weight"],
                "policy_base_weight": pf.get("policy_base_weight", 1.0),
                "policy_weight": pf["policy_weight"],
                "regime_multiplier": pf.get("regime_multiplier", 1.0),
                "session_multiplier": pf.get("session_multiplier", 1.0),
                "trust_score": pf["trust_score"],
                "weighted_score": pf["weighted_score"],
                "explanation": pf.get("explanation"),
            }
        )

    strategy_id, entry_model_id = _extract_strategy_ids(state, market_state)

    condition_vector = None
    try:
        condition_vector = encode_conditions(market_state) if market_state else None
        if condition_vector:
            logger.debug("Condition vector attached: %s", condition_vector)
    except Exception:
        condition_vector = None

    decision_timestamp = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    decision_frame = _build_decision_frame(
        state=state,
        market_state=market_state,
        session_context=session_context,
        condition_vector=condition_vector,
        decision_timestamp=decision_timestamp,
    )
    if decision_frame:
        decision_frame.entry_signals_present = _derive_entry_signals(
            decision_frame, state, market_state
        )
        decision_frame.eligible_entry_models = get_eligible_entry_models(decision_frame)
        _attach_entry_brain_shadow(decision_frame)

    # Shadow brain lookup (read-only) and influence
    brain_info = _lookup_brain_recommendation(
        strategy_id, entry_model_id, condition_vector
    )
    brain_cfg = _load_brain_config()
    safe_mode = _safe_mode_active()
    risk_limit_hit = (
        bool((state or {}).get("risk_limit_breached"))
        if isinstance(state, dict)
        else False
    )
    policy_allowed = True  # placeholder; if policy gate fails, this remains False

    score_before_brain = eval_score
    eval_score, brain_applied = _apply_brain_weighting(
        score=eval_score,
        brain_info=brain_info,
        brain_cfg=brain_cfg,
        safe_mode=safe_mode,
        risk_limit_hit=risk_limit_hit,
        policy_allowed=policy_allowed,
    )
    if brain_applied:
        # Recompute decision from adjusted score for deterministic effect
        if eval_score > 0.2:
            decision = "buy"
            reason = f"Brain-influenced BULLISH (score: {eval_score:.3f})"
        elif eval_score < -0.2:
            decision = "sell"
            reason = f"Brain-influenced BEARISH (score: {eval_score:.3f})"
        else:
            decision = "hold"
            reason = f"Brain-influenced NEUTRAL (score: {eval_score:.3f})"
        confidence = clamp(abs(eval_score), 0.0, 1.0)
        logger.debug(
            "Brain influence: strategy_id=%s label=%s score_before=%.4f score_after=%.4f",
            strategy_id,
            brain_info.get("brain_label") if brain_info else None,
            score_before_brain,
            eval_score,
        )

    result_payload = {
        "decision": decision,
        "confidence": confidence,
        "reason": reason,
        "eval_score": eval_score,
        "causal_reasoning": factor_explanations,
        "timestamp": result.timestamp,
        "evaluator_mode": "causal",
        "policy_applied": policy_applied,
        "details": {
            "causal_eval": eval_score,
            "factors": factor_explanations,
        },
        # Phase 12 attribution scaffolding (no behavior change)
        "strategy_id": strategy_id,
        "entry_model_id": entry_model_id,
        "exit_model_id": None,
        "condition_vector": condition_vector,
        "ml_policy_version": None,
        "brain_label": brain_info.get("brain_label") if brain_info else None,
        "brain_prob_good": brain_info.get("brain_prob_good") if brain_info else None,
        "brain_expected_reward": (
            brain_info.get("brain_expected_reward") if brain_info else None
        ),
        "brain_sample_size": (
            brain_info.get("brain_sample_size") if brain_info else None
        ),
        "brain_influence_applied": bool(brain_applied),
        "brain_adjusted_score": eval_score if brain_applied else None,
        "entry_signals_present": (
            decision_frame.entry_signals_present if decision_frame else None
        ),
        "eligible_entry_models": (
            decision_frame.eligible_entry_models if decision_frame else None
        ),
        "entry_brain_labels": (
            decision_frame.entry_brain_labels if decision_frame else None
        ),
        "entry_brain_scores": (
            decision_frame.entry_brain_scores if decision_frame else None
        ),
        "decision_frame": decision_frame.to_dict() if decision_frame else None,
    }

    try:
        session_label = session_context.get("session_regime") or session_context.get(
            "session", "UNKNOWN"
        )
        macro_labels: List[str] = []
        macro_attr = getattr(market_state, "macro_regime", None)
        if macro_attr:
            macro_labels.append(str(macro_attr))
        macro_labels.extend(regimes)
        macro_labels = [str(m).upper() for m in macro_labels if m]
        macro_regimes = list(dict.fromkeys(macro_labels))

        # Effective weights keyed by factor name
        effective_weights = {
            pf.get("factor"): float(pf.get("policy_weight", 1.0))
            for pf in policy_factors
            if pf.get("factor")
        }

        feature_vector = state.get("features", {}) if isinstance(state, dict) else {}

        # Base policy components (best-effort extraction from PolicyConfig)
        base_weights = {}
        trust_map = {}
        regime_multipliers = {}
        session_multiplier_val = float(
            session_context.get("modifiers", {}).get("risk_scale", 1.0)
        )
        if policy is not None:
            base_weights = {
                k: float(v)
                for k, v in (getattr(policy, "base_weights", {}) or {}).items()
                if v is not None
            }
            trust_map = {
                k: float(v)
                for k, v in (getattr(policy, "trust_map", {}) or {}).items()
                if v is not None
            }
            raw_rm = getattr(policy, "regime_multipliers", {}) or {}
            regime_multipliers = {
                rk: {fk: float(fv) for fk, fv in (rv or {}).items() if fv is not None}
                for rk, rv in raw_rm.items()
            }

        policy_meta = getattr(policy, "data", {}) if policy is not None else {}

        feature_audit = (
            state.get("feature_audit", {}) if isinstance(state, dict) else {}
        )

        provenance = {
            "policy_version": str(
                policy_meta.get("version") or getattr(policy, "version", "unknown")
            ),
            "feature_spec_version": str(
                feature_audit.get("registry_version") or "unknown"
            ),
            "feature_audit_version": str(
                feature_audit.get("version")
                or feature_audit.get("timestamp_utc")
                or "unknown"
            ),
            "engine_version": str(
                feature_audit.get("engine_version")
                or policy_meta.get("engine_version")
                or "unknown"
            ),
        }

        # Required identifiers with fallbacks to maintain determinism
        run_id = "UNKNOWN"
        if isinstance(state, dict):
            run_id = str(
                state.get("run_id")
                or state.get("experiment_id")
                or feature_audit.get("run_id")
                or "UNKNOWN"
            )

        symbol = "UNKNOWN"
        if isinstance(state, dict):
            symbol = str(state.get("symbol") or state.get("market", "UNKNOWN"))
        timeframe = "UNKNOWN"
        if isinstance(state, dict):
            timeframe = str(
                state.get("timeframe")
                or state.get("candle_data", {}).get("timeframe")
                or state.get("candles", {}).get("timeframe")
                or "UNKNOWN"
            )

        # Map decision string to LONG/SHORT/FLAT without mutating the decision logic
        action_map = {"buy": "LONG", "sell": "SHORT"}
        action_label = action_map.get(str(decision).lower(), "FLAT")

        entry = {
            "run_id": run_id,
            "decision_id": str(uuid.uuid4()),
            "timestamp_utc": decision_timestamp,
            "symbol": symbol,
            "timeframe": timeframe,
            "session_regime": str(session_label or "UNKNOWN"),
            "macro_regimes": macro_regimes,
            "feature_vector": feature_vector,
            "effective_weights": effective_weights,
            "policy_components": {
                "base_weights": base_weights,
                "trust": trust_map,
                "regime_multipliers": regime_multipliers,
                "session_multiplier": session_multiplier_val,
            },
            "evaluation_score": float(eval_score),
            "action": action_label,
            "outcome": None,
            "provenance": provenance,
            "strategy_id": strategy_id,
            "entry_model_id": entry_model_id,
            "exit_model_id": None,
            "condition_vector": condition_vector,
            "ml_policy_version": None,
            "brain_label": brain_info.get("brain_label") if brain_info else None,
            "brain_prob_good": (
                brain_info.get("brain_prob_good") if brain_info else None
            ),
            "brain_expected_reward": (
                brain_info.get("brain_expected_reward") if brain_info else None
            ),
            "brain_sample_size": (
                brain_info.get("brain_sample_size") if brain_info else None
            ),
            "brain_influence_applied": bool(brain_applied),
            "brain_adjusted_score": eval_score if brain_applied else None,
            "entry_signals_present": (
                decision_frame.entry_signals_present if decision_frame else None
            ),
            "eligible_entry_models": (
                decision_frame.eligible_entry_models if decision_frame else None
            ),
            "entry_brain_labels": (
                decision_frame.entry_brain_labels if decision_frame else None
            ),
            "entry_brain_scores": (
                decision_frame.entry_brain_scores if decision_frame else None
            ),
            "decision_frame": decision_frame.to_dict() if decision_frame else None,
        }

        position_size = None
        exec_ctx = getattr(market_state, "execution", None) if market_state else None
        if exec_ctx is not None:
            position_size = getattr(exec_ctx, "position_size", None)
        if position_size is not None:
            try:
                entry["position_size"] = float(position_size)
            except Exception:
                pass

        _apply_shadow_brain_annotations(
            strategy_id=strategy_id,
            entry_model_id=entry_model_id,
            condition_vector=condition_vector,
            result_payload=result_payload,
            entry=entry,
        )

        logger.debug(
            "Decision attribution: strategy_id=%s entry_model_id=%s exit_model_id=%s",
            entry.get("strategy_id"),
            entry.get("entry_model_id"),
            entry.get("exit_model_id"),
        )
        _DECISION_LOGGER.log_decision(entry)
    except Exception as exc:  # pragma: no cover - logging must not break decision loop
        logger.debug("Decision logging skipped: %s", exc)

    return result_payload


def create_evaluator_factory(
    use_causal: bool = False, use_policy_engine: bool = False, **causal_kwargs
) -> Callable:
    """
    Factory function to create evaluator with selected backend.

    Args:
        use_causal: If True, use CausalEvaluator; if False, use traditional evaluator
        use_policy_engine: If True AND use_causal=True, add PolicyEngine decision layer
        **causal_kwargs: Keyword arguments for CausalEvaluator (weights, verbose, official_mode)

    Returns:
        Evaluator function that accepts (state, market_state, open_position)

    Raises:
        ImportError: If mode not supported or modules unavailable
    """
    if canonical_enforced():
        assert_causal_required(use_causal)

    # ========================================================================
    # MODE 1: Integrated CausalEvaluator + PolicyEngine (NEW)
    # ========================================================================
    if use_causal and use_policy_engine:
        if not INTEGRATION_AVAILABLE:
            raise ImportError("Integration module required for causal+policy mode")

        try:
            from engine.causal_evaluator import CausalEvaluator
            from engine.policy_engine import PolicyEngine, RiskConfig

            # Initialize evaluators
            causal_evaluator = CausalEvaluator(**causal_kwargs)
            policy_engine = PolicyEngine(verbose=causal_kwargs.get("verbose", False))

            def integrated_wrapper(
                state: Dict,
                market_state: Optional[Any] = None,
                open_position: Optional[Dict] = None,
                position_state: Optional[Any] = None,
                risk_config: Optional[Any] = None,
                daily_loss_pct: float = 0.0,
            ) -> Dict:
                """Integrated causal+policy evaluation"""

                if market_state is None:
                    if canonical_enforced():
                        raise ValueError(
                            "MarketState required in canonical/official modes; legacy evaluator fallback is blocked."
                        )
                    logger.warning(
                        "market_state not provided; falling back to traditional evaluator"
                    )
                    return evaluate(state, open_position=open_position)

                # Use defaults if not provided
                if position_state is None:
                    from engine.policy_engine import PositionSide, PositionState

                    position_state = PositionState(side=PositionSide.FLAT, size=0.0)

                if risk_config is None:
                    risk_config = RiskConfig()

                # Run integrated pipeline
                try:
                    result = evaluate_and_decide(
                        market_state=market_state,
                        position_state=position_state,
                        risk_config=risk_config,
                        causal_evaluator=causal_evaluator,
                        policy_engine=policy_engine,
                        daily_loss_pct=daily_loss_pct,
                        verbose=causal_kwargs.get("verbose", False),
                    )

                    # Convert to legacy format for compatibility
                    return {
                        "decision": (
                            result["action"].lower()
                            if isinstance(result["action"], str)
                            else result["action"].value.lower()
                        ),
                        "confidence": result["confidence"],
                        "reason": f"CausalEval+Policy: {result['decision_zone']}",
                        "eval_score": result["eval_score"],
                        "details": {
                            "causal_eval": result["eval_score"],
                            "policy_action": result["action"],
                            "target_size": result["target_size"],
                            "evaluation_zone": result["decision_zone"],
                            "causal_reasoning": result["reasoning"]["eval"],
                            "policy_reasoning": result["reasoning"]["policy"],
                        },
                        "integrated_mode": True,
                        "causal_evaluator": True,
                        "policy_engine": True,
                        "deterministic": True,
                        "lookahead_safe": True,
                    }
                except Exception as e:
                    logger.error(
                        f"Integrated pipeline failed: {e}; falling back to traditional"
                    )
                    return evaluate(state, open_position=open_position)

            return integrated_wrapper

        except ImportError as e:
            logger.error(f"Cannot import causal+policy modules: {e}")
            raise

    # ========================================================================
    # MODE 2: CausalEvaluator only
    # ========================================================================
    elif use_causal:
        try:
            from engine.causal_evaluator import CausalEvaluator

            # Initialize CausalEvaluator with provided kwargs
            causal_evaluator = CausalEvaluator(**causal_kwargs)

            def causal_eval_wrapper(
                state: Dict,
                market_state: Optional[Any] = None,
                open_position: Optional[Dict] = None,
            ) -> Dict:
                """Wrapper for causal evaluation"""
                if market_state is None:
                    # If no causal market_state provided, use traditional evaluator
                    logger.warning(
                        "market_state not provided; falling back to traditional evaluator"
                    )
                    return evaluate(state, open_position=open_position)

                return evaluate_with_causal(
                    state=state,
                    causal_evaluator=causal_evaluator,
                    market_state=market_state,
                )

            return causal_eval_wrapper

        except ImportError:
            logger.error(
                "CausalEvaluator not available; falling back to traditional evaluator"
            )
            return lambda state, market_state=None, open_position=None: evaluate(
                state, open_position=open_position
            )

    # ========================================================================
    # MODE 3: Traditional evaluator (default)
    # ========================================================================
    else:
        # Return traditional evaluator
        return lambda state, market_state=None, open_position=None: evaluate(
            state, open_position=open_position
        )


if __name__ == "__main__":
    """Example usage and testing"""

    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("\n" + "=" * 70)
    print("EVALUATOR ENGINE - TEST RUN")
    print("=" * 70)

    # Create a mock state (would come from state_builder.py)
    mock_state = {
        "timestamp": 1705441218.528,
        "symbol": "EURUSD",
        "tick": {
            "bid": 1.0850,
            "ask": 1.0852,
            "spread": 2.0,
            "last_tick_time": 1705441213,
        },
        "indicators": {
            "rsi_14": 35.0,  # Oversold = buy signal
            "sma_50": 1.0835,
            "sma_200": 1.0800,
            "atr_14": 0.0012,
            "volatility": 0.5,
        },
        "trend": {
            "regime": "uptrend",
            "strength": 0.75,
        },
        "sentiment": {
            "score": 0.2,
            "confidence": 0.3,
            "source": "placeholder",
        },
        "candles": {
            "M1": {
                "indicators": {"rsi_14": 25.0},
                "latest": {"time": 1705441210},
            },
            "M5": {
                "indicators": {"rsi_14": 32.0},
                "latest": {"time": 1705441210},
            },
            "M15": {
                "indicators": {"rsi_14": 35.0},
                "latest": {"time": 1705441210},
            },
            "H1": {
                "indicators": {"rsi_14": 38.0},
                "latest": {"time": 1705441210},
            },
        },
        "health": {
            "is_stale": False,
            "last_update": 1705441218.528,
            "errors": [],
        },
    }

    # Test 1: Basic evaluation
    print("\n[TEST 1] Basic Evaluation (Uptrend + Oversold RSI)")
    result = evaluate(mock_state)
    print(f"Decision: {result['decision'].upper()}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Reason: {result['reason']}")

    # Test 2: Evaluation with position
    print("\n[TEST 2] Evaluation with Open Position (Close Check)")
    mock_state_overbought = mock_state.copy()
    mock_state_overbought["indicators"]["rsi_14"] = 75.0

    open_pos = {"direction": "buy", "entry_price": 1.0840}
    result = evaluate(mock_state_overbought, open_position=open_pos)
    print(f"Decision: {result['decision'].upper()}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Reason: {result['reason']}")

    # Test 3: Sell signal
    print("\n[TEST 3] Sell Signal (Downtrend + Overbought)")
    mock_state_sell = mock_state.copy()
    mock_state_sell["trend"]["regime"] = "downtrend"
    mock_state_sell["indicators"]["rsi_14"] = 75.0
    mock_state_sell["sentiment"]["score"] = -0.3

    result = evaluate(mock_state_sell)
    print(f"Decision: {result['decision'].upper()}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Reason: {result['reason']}")

    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70 + "\n")
