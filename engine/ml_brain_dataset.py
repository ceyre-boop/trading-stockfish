import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .entry_features import (
    ENTRY_FEATURE_NAMES,
    ENTRY_FEATURE_TEMPLATES,
    extract_entry_features,
)
from .entry_models import ENTRY_MODELS


def _expand_condition_vector(df: pd.DataFrame) -> pd.DataFrame:
    """Expand condition_vector dict into flat columns (session, macro, vol, trend, liquidity, tod)."""

    cond_cols = ["session", "macro", "vol", "trend", "liquidity", "tod"]
    if "condition_vector" not in df.columns:
        for col in cond_cols:
            df[col] = None
        return df

    def _extract(cv: Any, key: str) -> Any:
        if isinstance(cv, dict):
            return cv.get(key)
        return None

    for col in cond_cols:
        df[col] = df["condition_vector"].apply(lambda cv, k=col: _extract(cv, k))
    return df


def _extract_cognitive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract structure_brain cognitive features from decision_frame payloads."""

    cognitive_cols = [
        "market_profile_state",
        "market_profile_confidence",
        "session_profile",
        "session_profile_confidence",
        "liquidity_bias_side",
        "nearest_target_type",
        "nearest_target_distance",
        "liquidity_sweep_flags",
        "displacement_score",
    ]

    if "decision_frame" not in df.columns:
        for col in cognitive_cols:
            df[col] = None
        return df

    def _extract(frame: Any) -> Dict[str, Any]:
        if not isinstance(frame, dict):
            return {col: None for col in cognitive_cols}

        mp_ev = frame.get("market_profile_evidence")
        if not isinstance(mp_ev, dict):
            mp_ev = {}
        liquidity = frame.get("liquidity_frame")
        if not isinstance(liquidity, dict):
            liquidity = {}
        distances = liquidity.get("distances") if isinstance(liquidity, dict) else {}
        if not isinstance(distances, dict):
            distances = {}

        nearest_target = liquidity.get("primary_target") or liquidity.get("target")
        nearest_distance = None
        if nearest_target is not None:
            nearest_distance = distances.get(nearest_target)

        return {
            "market_profile_state": frame.get("market_profile_state"),
            "market_profile_confidence": frame.get("market_profile_confidence"),
            "session_profile": frame.get("session_profile"),
            "session_profile_confidence": frame.get("session_profile_confidence"),
            "liquidity_bias_side": liquidity.get("bias"),
            "nearest_target_type": nearest_target,
            "nearest_target_distance": nearest_distance,
            "liquidity_sweep_flags": liquidity.get("swept"),
            "displacement_score": mp_ev.get("displacement_score"),
        }

    extracted = df["decision_frame"].apply(_extract)
    for col in cognitive_cols:
        df[col] = extracted.apply(lambda row, c=col: row.get(c))
    return df


def _condition_vector_as_str(cv: Any) -> Any:
    if isinstance(cv, dict):
        try:
            return json.dumps(cv, sort_keys=True)
        except Exception:
            return str(cv)
    if cv is None:
        return None
    return str(cv)


def _entry_feature_names() -> List[str]:
    if not ENTRY_FEATURE_TEMPLATES:
        return []
    template = next(iter(ENTRY_FEATURE_TEMPLATES.values()))
    return list(template.feature_names)


def _extract_entry_annotations(df: pd.DataFrame) -> pd.DataFrame:
    feature_names = _entry_feature_names()

    if "decision_frame" not in df.columns:
        df["chosen_entry_model_id"] = None
        df["eligible_entry_models"] = None
        df["entry_signals_present"] = None
        df["entry_brain_label"] = None
        df["entry_brain_prob_good"] = None
        df["entry_brain_expected_R"] = None
        for name in feature_names:
            df[f"entry_feature_{name}"] = None
        df["entry_outcome"] = df.get("outcome")
        df["entry_success"] = df["entry_outcome"].apply(
            lambda o: (
                1 if pd.notna(o) and float(o) > 0 else (0 if pd.notna(o) else None)
            )
        )
        return df

    def _extract(row: pd.Series) -> Dict[str, Any]:
        frame = row.get("decision_frame")
        entry_id = row.get("entry_model_id")
        if isinstance(frame, dict):
            entry_id = frame.get("chosen_entry_model_id", entry_id)
        annotations: Dict[str, Any] = {
            "chosen_entry_model_id": entry_id,
            "eligible_entry_models": None,
            "entry_signals_present": None,
            "entry_brain_label": None,
            "entry_brain_prob_good": None,
            "entry_brain_expected_R": None,
            "features": {},
        }
        if isinstance(frame, dict):
            annotations["eligible_entry_models"] = frame.get("eligible_entry_models")
            annotations["entry_signals_present"] = frame.get("entry_signals_present")
            labels = (
                frame.get("entry_brain_labels")
                if isinstance(frame.get("entry_brain_labels"), dict)
                else {}
            )
            scores = (
                frame.get("entry_brain_scores")
                if isinstance(frame.get("entry_brain_scores"), dict)
                else {}
            )
            if entry_id:
                annotations["entry_brain_label"] = labels.get(entry_id)
                entry_score = scores.get(entry_id) if isinstance(scores, dict) else None
                if isinstance(entry_score, dict):
                    annotations["entry_brain_prob_good"] = entry_score.get("prob_good")
                    annotations["entry_brain_expected_R"] = entry_score.get(
                        "expected_reward"
                    )

        if entry_id:
            try:
                annotations["features"] = extract_entry_features(entry_id, frame)
            except Exception:
                annotations["features"] = {}
        return annotations

    extracted = df.apply(_extract, axis=1)
    df["chosen_entry_model_id"] = extracted.apply(
        lambda ann: ann.get("chosen_entry_model_id")
    )
    df["eligible_entry_models"] = extracted.apply(
        lambda ann: ann.get("eligible_entry_models")
    )
    df["entry_signals_present"] = extracted.apply(
        lambda ann: ann.get("entry_signals_present")
    )
    df["entry_brain_label"] = extracted.apply(lambda ann: ann.get("entry_brain_label"))
    df["entry_brain_prob_good"] = extracted.apply(
        lambda ann: ann.get("entry_brain_prob_good")
    )
    df["entry_brain_expected_R"] = extracted.apply(
        lambda ann: ann.get("entry_brain_expected_R")
    )
    for name in feature_names:
        df[f"entry_feature_{name}"] = extracted.apply(
            lambda ann, n=name: ann.get("features", {}).get(n)
        )

    df["entry_outcome"] = df.get("outcome")
    df["entry_success"] = df["entry_outcome"].apply(
        lambda o: 1 if pd.notna(o) and float(o) > 0 else (0 if pd.notna(o) else None)
    )
    return df


def _compute_group_metrics(group: pd.DataFrame) -> Dict[str, Any]:
    outcomes = group["outcome"].astype(float)
    sample_size = len(outcomes)
    win_rate = float((outcomes > 0).mean()) if sample_size else 0.0
    avg_reward = float(outcomes.mean()) if sample_size else 0.0
    reward_variance = float(outcomes.var(ddof=0)) if sample_size else 0.0
    reward_std = float(outcomes.std(ddof=0)) if sample_size else 0.0

    stability_mean_5 = None
    stability_std_5 = None
    if "timestamp_utc" in group.columns:
        try:
            sorted_group = group.sort_values("timestamp_utc")
            rolling = sorted_group["outcome"].rolling(window=5, min_periods=1)
            stability_mean_5 = float(rolling.mean().iloc[-1])
            stability_std_5 = float(rolling.std(ddof=0).iloc[-1])
        except Exception:
            stability_mean_5 = None
            stability_std_5 = None

    return {
        "sample_size": sample_size,
        "win_rate": win_rate,
        "avg_reward": avg_reward,
        "reward_variance": reward_variance,
        "reward_std": reward_std,
        "stability_mean_5": stability_mean_5,
        "stability_std_5": stability_std_5,
    }


def build_brain_dataset(decisions: pd.DataFrame, min_samples: int = 1) -> pd.DataFrame:
    """Transform enriched decision logs into an ML-ready dataset.

    Deterministic, static scaffolding for Phase 12 brain dataset preparation.
    """

    if decisions is None:
        return pd.DataFrame()

    df = decisions.copy()

    # Basic filtering
    required_cols = ["strategy_id", "entry_model_id", "outcome"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    df = df.dropna(subset=["strategy_id", "entry_model_id", "outcome"], how="any")

    if df.empty:
        return pd.DataFrame()

    # Expand condition vector into flat columns
    df = _expand_condition_vector(df)
    df = _extract_cognitive_features(df)
    df = _extract_entry_annotations(df)

    if "condition_vector" in df.columns:
        df["condition_vector_str"] = df["condition_vector"].apply(
            _condition_vector_as_str
        )
    else:
        df["condition_vector_str"] = None

    # Ensure timestamp ordering is deterministic if present
    if "timestamp_utc" in df.columns:
        try:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
        except Exception:
            pass

    group_keys = [
        "strategy_id",
        "entry_model_id",
        "session",
        "macro",
        "vol",
        "trend",
        "liquidity",
        "tod",
        "market_profile_state",
        "session_profile",
        "liquidity_bias_side",
        "condition_vector_str",
    ]

    grouped = df.groupby(group_keys, dropna=False)

    records = []
    for keys, group in grouped:
        metrics = _compute_group_metrics(group)
        if metrics["sample_size"] < min_samples:
            continue

        if "timestamp_utc" in group.columns:
            try:
                group = group.sort_values("timestamp_utc")
            except Exception:
                pass

        def _pick_latest(col: str):
            series = group[col]
            non_null = series.dropna()
            if non_null.empty:
                return None
            return non_null.iloc[-1]

        record = {
            "strategy_id": keys[0],
            "entry_model_id": keys[1],
            "session": keys[2],
            "macro": keys[3],
            "vol": keys[4],
            "trend": keys[5],
            "liquidity": keys[6],
            "tod": keys[7],
            "market_profile_state": keys[8],
            "session_profile": keys[9],
            "liquidity_bias_side": keys[10],
            "condition_vector_str": keys[11],
        }
        record.update(metrics)
        record.update(
            {
                "market_profile_confidence": _pick_latest("market_profile_confidence"),
                "session_profile_confidence": _pick_latest(
                    "session_profile_confidence"
                ),
                "nearest_target_type": _pick_latest("nearest_target_type"),
                "nearest_target_distance": _pick_latest("nearest_target_distance"),
                "liquidity_sweep_flags": _pick_latest("liquidity_sweep_flags"),
                "displacement_score": _pick_latest("displacement_score"),
                "entry_outcome": _pick_latest("entry_outcome"),
                "entry_success": _pick_latest("entry_success"),
                "chosen_entry_model_id": _pick_latest("chosen_entry_model_id"),
                "eligible_entry_models": _pick_latest("eligible_entry_models"),
                "entry_signals_present": _pick_latest("entry_signals_present"),
                "entry_brain_label": _pick_latest("entry_brain_label"),
                "entry_brain_prob_good": _pick_latest("entry_brain_prob_good"),
                "entry_brain_expected_R": _pick_latest("entry_brain_expected_R"),
            }
        )

        for feature_name in _entry_feature_names():
            record[f"entry_feature_{feature_name}"] = _pick_latest(
                f"entry_feature_{feature_name}"
            )
        records.append(record)

    result = pd.DataFrame(records)
    if result.empty:
        return result

    sort_order = [
        "strategy_id",
        "entry_model_id",
        "session",
        "macro",
        "vol",
        "trend",
        "liquidity",
        "tod",
        "market_profile_state",
        "session_profile",
        "liquidity_bias_side",
        "condition_vector_str",
    ]
    result = result.sort_values(sort_order).reset_index(drop=True)
    feature_cols = [f"entry_feature_{name}" for name in _entry_feature_names()]
    ordered_cols = (
        sort_order
        + [
            "sample_size",
            "win_rate",
            "avg_reward",
            "reward_variance",
            "reward_std",
            "stability_mean_5",
            "stability_std_5",
            "market_profile_confidence",
            "session_profile_confidence",
            "nearest_target_type",
            "nearest_target_distance",
            "liquidity_sweep_flags",
            "displacement_score",
            "entry_outcome",
            "entry_success",
            "chosen_entry_model_id",
            "eligible_entry_models",
            "entry_signals_present",
            "entry_brain_label",
            "entry_brain_prob_good",
            "entry_brain_expected_R",
        ]
        + feature_cols
    )
    existing_cols = [col for col in ordered_cols if col in result.columns]
    remaining_cols = [col for col in result.columns if col not in ordered_cols]
    result = result[existing_cols + remaining_cols]
    return result


def validate_entry_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "missing_features": [],
        "invalid_entry_ids": [],
        "rows_with_missing_outcomes": 0,
        "ok": True,
    }

    if df is None or df.empty:
        summary["ok"] = False
        return summary

    invalid_ids = sorted(
        {eid for eid in df.get("entry_model_id", []) if eid and eid not in ENTRY_MODELS}
    )
    summary["invalid_entry_ids"] = invalid_ids

    required_fields = ["market_profile_state", "session_profile", "liquidity_bias_side"]
    missing_features: List[str] = []
    for field in required_fields:
        if field not in df.columns or df[field].isnull().any():
            missing_features.append(field)

    feature_cols = [f"entry_feature_{name}" for name in ENTRY_FEATURE_NAMES]
    for col in feature_cols:
        if col not in df.columns:
            missing_features.append(col)

    summary["missing_features"] = sorted(set(missing_features))

    if "chosen_entry_model_id" in df.columns:
        if "entry_outcome" not in df.columns:
            summary["rows_with_missing_outcomes"] = int(
                df[df["chosen_entry_model_id"].notna()].shape[0]
            )
        else:
            missing_outcomes = df[
                (df["chosen_entry_model_id"].notna()) & (df["entry_outcome"].isna())
            ]
            summary["rows_with_missing_outcomes"] = int(len(missing_outcomes))
    else:
        summary["rows_with_missing_outcomes"] = 0

    summary["ok"] = not (
        summary["missing_features"]
        or summary["invalid_entry_ids"]
        or summary["rows_with_missing_outcomes"]
    )

    return summary
