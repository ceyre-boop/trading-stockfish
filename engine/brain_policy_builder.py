import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from engine.decision_frame import DecisionFrame
from engine.ml_brain import NUMERIC_COLUMNS, BrainModelArtifacts, score_combo
from engine.ml_brain_entry import score_entry_models

POLICY_SORT_KEYS: List[str] = [
    "strategy_id",
    "entry_model_id",
    "session",
    "macro",
    "vol",
    "trend",
    "liquidity",
    "tod",
]

DEFAULT_POLICY_THRESHOLDS: Dict[str, Any] = {
    "min_samples": 5,
    "prob_good_min": 0.4,
    "expected_reward_min": 0.0,
    "output_dir": os.path.join("storage", "policies", "brain"),
}

POLICY_FILENAME = "brain_policy.json"
ENTRY_POLICY_FILENAME = "brain_policy_entries.json"


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in POLICY_SORT_KEYS:
        if col not in df.columns:
            df[col] = None
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].fillna(0.0)
    df[POLICY_SORT_KEYS] = df[POLICY_SORT_KEYS].fillna("UNKNOWN")
    return df


def _condition_from_row(row: pd.Series) -> Dict[str, Any]:
    condition_vector: Dict[str, Any] = {}
    for key in POLICY_SORT_KEYS[2:]:
        condition_vector[key] = row.get(key)
    for key in NUMERIC_COLUMNS:
        condition_vector[key] = row.get(key)
    return condition_vector


def _label_policy(
    sample_size: int,
    prob_good: float,
    expected_reward: float,
    thresholds: Dict[str, Any],
) -> str:
    if sample_size < thresholds.get(
        "min_samples", DEFAULT_POLICY_THRESHOLDS["min_samples"]
    ):
        return "DISABLED"
    if prob_good < thresholds.get(
        "prob_good_min", DEFAULT_POLICY_THRESHOLDS["prob_good_min"]
    ):
        return "DISCOURAGED"
    if expected_reward < thresholds.get(
        "expected_reward_min", DEFAULT_POLICY_THRESHOLDS["expected_reward_min"]
    ):
        return "ALLOWED"
    return "PREFERRED"


def _save_policy_artifact(
    policy_df: pd.DataFrame, artifacts: BrainModelArtifacts, thresholds: Dict[str, Any]
) -> str:
    output_dir = thresholds.get("output_dir", DEFAULT_POLICY_THRESHOLDS["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, POLICY_FILENAME)

    payload = {
        "metadata": {
            "version_ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "thresholds": thresholds,
            "training_metadata": artifacts.training_metadata,
            "model_types": {
                "classifier": artifacts.classifier.__class__.__name__,
                "regressor": artifacts.regressor.__class__.__name__,
            },
        },
        "policy": policy_df.to_dict(orient="records"),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, sort_keys=True, indent=2)
    return path


def _save_entry_policy_artifact(
    policy_df: pd.DataFrame, artifacts: BrainModelArtifacts, thresholds: Dict[str, Any]
) -> str:
    output_dir = thresholds.get("output_dir", DEFAULT_POLICY_THRESHOLDS["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, ENTRY_POLICY_FILENAME)

    payload = {
        "metadata": {
            "version_ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "thresholds": thresholds,
            "training_metadata": artifacts.training_metadata,
            "model_types": {
                "classifier": artifacts.classifier.__class__.__name__,
                "regressor": artifacts.regressor.__class__.__name__,
            },
        },
        "policy": policy_df.to_dict(orient="records"),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, sort_keys=True, indent=2)
    return path


def build_brain_policy(
    dataset: pd.DataFrame,
    artifacts: BrainModelArtifacts,
    thresholds: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    thresholds = {**DEFAULT_POLICY_THRESHOLDS, **(thresholds or {})}
    if dataset is None or dataset.empty:
        return pd.DataFrame(
            columns=POLICY_SORT_KEYS
            + ["prob_good", "expected_reward", "sample_size", "label"]
        )

    df = _ensure_columns(dataset)

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        sample_size = int(row.get("sample_size", 0) or 0)
        condition_vector = _condition_from_row(row)
        score = score_combo(
            row.get("strategy_id"),
            row.get("entry_model_id"),
            condition_vector,
            artifacts,
        )

        label = _label_policy(
            sample_size, score.prob_good, score.expected_reward, thresholds
        )

        record = {
            "strategy_id": row.get("strategy_id"),
            "entry_model_id": row.get("entry_model_id"),
            "session": row.get("session"),
            "macro": row.get("macro"),
            "vol": row.get("vol"),
            "trend": row.get("trend"),
            "liquidity": row.get("liquidity"),
            "tod": row.get("tod"),
            "prob_good": float(score.prob_good),
            "expected_reward": float(score.expected_reward),
            "sample_size": sample_size,
            "label": label,
        }
        records.append(record)

    policy_df = pd.DataFrame(records)
    policy_df = policy_df.sort_values(POLICY_SORT_KEYS).reset_index(drop=True)

    _save_policy_artifact(policy_df, artifacts, thresholds)
    return policy_df


def _entry_frame_from_row(row: pd.Series) -> DecisionFrame:
    liq_bias = row.get("liquidity_bias_side")
    liquidity = {"bias": liq_bias} if liq_bias is not None else {}
    frame = DecisionFrame(
        market_profile_state=row.get("market_profile_state"),
        session_profile=row.get("session_profile"),
        liquidity_frame=liquidity,
        vol_regime=row.get("vol"),
        trend_regime=row.get("trend"),
    )
    return frame


def build_entry_brain_policy(
    dataset: pd.DataFrame,
    artifacts: BrainModelArtifacts,
    thresholds: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    thresholds = {**DEFAULT_POLICY_THRESHOLDS, **(thresholds or {})}
    if dataset is None or dataset.empty:
        return pd.DataFrame(
            columns=[
                "entry_model_id",
                "market_profile_state",
                "session_profile",
                "liquidity_bias_side",
                "prob_good",
                "expected_reward",
                "sample_size",
                "label",
            ]
        )

    df = dataset.copy()
    for col in [
        "entry_model_id",
        "market_profile_state",
        "session_profile",
        "liquidity_bias_side",
    ]:
        if col not in df.columns:
            df[col] = None
    if "sample_size" not in df.columns:
        df["sample_size"] = 0

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        entry_id = row.get("entry_model_id")
        if not entry_id:
            continue
        frame = _entry_frame_from_row(row)
        scores = score_entry_models(frame, [entry_id], artifacts)
        score = scores.get(entry_id)
        if score is None:
            continue

        sample_size = int(row.get("sample_size", 0) or 0)
        label = _label_policy(
            sample_size=sample_size,
            prob_good=score.prob_good,
            expected_reward=score.expected_reward,
            thresholds=thresholds,
        )

        records.append(
            {
                "entry_model_id": entry_id,
                "market_profile_state": row.get("market_profile_state") or "UNKNOWN",
                "session_profile": row.get("session_profile") or "UNKNOWN",
                "liquidity_bias_side": row.get("liquidity_bias_side") or "UNKNOWN",
                "prob_good": float(score.prob_good),
                "expected_reward": float(score.expected_reward),
                "sample_size": sample_size,
                "label": label,
            }
        )

    policy_df = pd.DataFrame(records)
    if policy_df.empty:
        return policy_df

    policy_df = policy_df.sort_values(
        [
            "entry_model_id",
            "market_profile_state",
            "session_profile",
            "liquidity_bias_side",
        ]
    ).reset_index(drop=True)

    _save_entry_policy_artifact(policy_df, artifacts, thresholds)
    return policy_df
