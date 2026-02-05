import json
import os
from dataclasses import dataclass
from math import exp
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder

CAT_COLUMNS: List[str] = [
    "strategy_id",
    "entry_model_id",
    "session",
    "macro",
    "vol",
    "trend",
    "liquidity",
    "tod",
]

NUMERIC_COLUMNS: List[str] = [
    "sample_size",
    "win_rate",
    "avg_reward",
    "reward_variance",
    "reward_std",
    "stability_mean_5",
    "stability_std_5",
]

DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "win_rate_min": 0.5,
    "avg_reward_min": 0.0,
    "sample_size_warning": 5,
    "prob_good_warning": 0.25,
}


@dataclass
class BrainScore:
    prob_good: float
    expected_reward: float
    sample_size: int
    flags: Dict[str, Any]


@dataclass
class BrainModelArtifacts:
    classifier: Any
    regressor: Any
    thresholds: Dict[str, Any]
    training_metadata: Dict[str, Any]
    encoder: OneHotEncoder
    feature_columns: List[str]
    numeric_columns: List[str]


def _validate_and_prepare_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset is None or dataset.empty:
        raise ValueError("Dataset is empty; cannot train brain models.")

    df = dataset.copy()

    for col in CAT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].fillna(0.0)
    df[CAT_COLUMNS] = df[CAT_COLUMNS].fillna("UNKNOWN")
    return df


def _build_labels(df: pd.DataFrame, thresholds: Dict[str, Any]) -> np.ndarray:
    if "label_good" in df.columns:
        return df["label_good"].astype(int).to_numpy()

    win_rate_min = thresholds.get("win_rate_min", DEFAULT_THRESHOLDS["win_rate_min"])
    avg_reward_min = thresholds.get(
        "avg_reward_min", DEFAULT_THRESHOLDS["avg_reward_min"]
    )
    label = (
        (df["win_rate"].astype(float) >= float(win_rate_min))
        & (df["avg_reward"].astype(float) >= float(avg_reward_min))
    ).astype(int)
    return label.to_numpy()


def _fit_encoder(df: pd.DataFrame) -> OneHotEncoder:
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=float
        )
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=float)
    encoder.fit(df[CAT_COLUMNS])
    return encoder


def _encode_features(
    df: pd.DataFrame, encoder: OneHotEncoder, numeric_columns: List[str]
) -> np.ndarray:
    cat_matrix = encoder.transform(df[CAT_COLUMNS])
    numeric_matrix = df[numeric_columns].astype(float).to_numpy()
    return np.hstack([cat_matrix, numeric_matrix])


def _collect_feature_columns(
    encoder: OneHotEncoder, numeric_columns: List[str]
) -> List[str]:
    encoded_names = list(encoder.get_feature_names_out(CAT_COLUMNS))
    return encoded_names + numeric_columns


def train_brain_models(
    dataset: pd.DataFrame, thresholds: Optional[Dict[str, Any]] = None
) -> BrainModelArtifacts:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    df = _validate_and_prepare_dataset(dataset)

    encoder = _fit_encoder(df)
    feature_columns = _collect_feature_columns(encoder, NUMERIC_COLUMNS)

    labels = _build_labels(df, thresholds)
    regression_target = df["avg_reward"].astype(float).to_numpy()

    feature_matrix = _encode_features(df, encoder, NUMERIC_COLUMNS)

    classifier = GradientBoostingClassifier(random_state=42)
    classifier.fit(feature_matrix, labels)

    regressor = GradientBoostingRegressor(random_state=42)
    regressor.fit(feature_matrix, regression_target)

    metrics: Dict[str, Any] = {}
    try:
        preds = classifier.predict(feature_matrix)
        metrics["accuracy"] = float(accuracy_score(labels, preds))
        if len(np.unique(labels)) > 1:
            proba = classifier.predict_proba(feature_matrix)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(labels, proba))
    except Exception:
        metrics["accuracy"] = None

    try:
        reg_preds = regressor.predict(feature_matrix)
        metrics["mae"] = float(mean_absolute_error(regression_target, reg_preds))
        metrics["mse"] = float(mean_squared_error(regression_target, reg_preds))
    except Exception:
        metrics["mae"] = None
        metrics["mse"] = None

    training_metadata = {
        "metrics": metrics,
        "n_rows": len(df),
        "thresholds_used": thresholds,
        "feature_columns": feature_columns,
        "numeric_columns": NUMERIC_COLUMNS,
    }

    return BrainModelArtifacts(
        classifier=classifier,
        regressor=regressor,
        thresholds=thresholds,
        training_metadata=training_metadata,
        encoder=encoder,
        feature_columns=feature_columns,
        numeric_columns=NUMERIC_COLUMNS,
    )


def save_brain_models(artifacts: BrainModelArtifacts, path: str) -> None:
    os.makedirs(path, exist_ok=True)

    joblib.dump(artifacts.classifier, os.path.join(path, "classifier.joblib"))
    joblib.dump(artifacts.regressor, os.path.join(path, "regressor.joblib"))
    joblib.dump(artifacts.encoder, os.path.join(path, "encoder.joblib"))

    meta = {
        "thresholds": artifacts.thresholds,
        "training_metadata": artifacts.training_metadata,
        "feature_columns": artifacts.feature_columns,
        "numeric_columns": artifacts.numeric_columns,
    }
    with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, sort_keys=True, indent=2)


def load_brain_models(path: str) -> BrainModelArtifacts:
    classifier = joblib.load(os.path.join(path, "classifier.joblib"))
    regressor = joblib.load(os.path.join(path, "regressor.joblib"))
    encoder = joblib.load(os.path.join(path, "encoder.joblib"))

    with open(os.path.join(path, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    return BrainModelArtifacts(
        classifier=classifier,
        regressor=regressor,
        thresholds=meta.get("thresholds", DEFAULT_THRESHOLDS),
        training_metadata=meta.get("training_metadata", {}),
        encoder=encoder,
        feature_columns=meta.get("feature_columns", []),
        numeric_columns=meta.get("numeric_columns", NUMERIC_COLUMNS),
    )


def _extract_condition(condition_vector: Optional[Dict[str, Any]], key: str) -> Any:
    if isinstance(condition_vector, dict):
        return condition_vector.get(key)
    return None


def _prepare_scoring_frame(
    strategy_id: Any, entry_model_id: Any, condition_vector: Optional[Dict[str, Any]]
) -> pd.DataFrame:
    condition_vector = condition_vector or {}
    row = {
        "strategy_id": strategy_id,
        "entry_model_id": entry_model_id,
        "session": _extract_condition(condition_vector, "session") or "UNKNOWN",
        "macro": _extract_condition(condition_vector, "macro") or "UNKNOWN",
        "vol": _extract_condition(condition_vector, "vol") or "UNKNOWN",
        "trend": _extract_condition(condition_vector, "trend") or "UNKNOWN",
        "liquidity": _extract_condition(condition_vector, "liquidity") or "UNKNOWN",
        "tod": _extract_condition(condition_vector, "tod") or "UNKNOWN",
    }
    for col in NUMERIC_COLUMNS:
        row[col] = _extract_condition(condition_vector, col)
    return pd.DataFrame([row])


def _compute_probability(classifier: Any, features: np.ndarray) -> float:
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(features)
        if proba.shape[1] > 1:
            return float(proba[0, 1])
        return float(proba[0, 0])
    if hasattr(classifier, "decision_function"):
        score = classifier.decision_function(features)
        return float(1.0 / (1.0 + exp(-float(score))))
    return float(classifier.predict(features)[0])


def score_combo(
    strategy_id: Any,
    entry_model_id: Any,
    condition_vector: Optional[Dict[str, Any]],
    artifacts: BrainModelArtifacts,
) -> BrainScore:
    df = _prepare_scoring_frame(strategy_id, entry_model_id, condition_vector)

    if any(col not in df.columns for col in artifacts.numeric_columns):
        for col in artifacts.numeric_columns:
            if col not in df.columns:
                df[col] = 0.0

    df[artifacts.numeric_columns] = df[artifacts.numeric_columns].fillna(0.0)

    feature_matrix = _encode_features(df, artifacts.encoder, artifacts.numeric_columns)

    prob_good = _compute_probability(artifacts.classifier, feature_matrix)
    expected_reward = float(artifacts.regressor.predict(feature_matrix)[0])

    sample_size = int(df.iloc[0].get("sample_size", 0) or 0)
    flags: Dict[str, Any] = {}
    if sample_size < artifacts.thresholds.get("sample_size_warning", 0):
        flags["low_sample"] = True
    if prob_good < artifacts.thresholds.get("prob_good_warning", 0.0):
        flags["low_confidence"] = True

    return BrainScore(
        prob_good=float(prob_good),
        expected_reward=expected_reward,
        sample_size=sample_size,
        flags=flags,
    )
