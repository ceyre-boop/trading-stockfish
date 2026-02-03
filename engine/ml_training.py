from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from engine import research_api

# --------------------------- configs ---------------------------

MODEL_VERSION = "0.1.0"
DEFAULT_MODELS_DIR = Path("models")


@dataclass(frozen=True)
class TrainingConfig:
    models_dir: Path = DEFAULT_MODELS_DIR
    lookahead: int = 1  # number of rows ahead (time-ordered) for label
    volatility_threshold: float = 1.0
    risk_on_token: str = "RISK_ON"
    train_frac: float = 0.8  # deterministic time-based split
    start_date: Optional[date] = None
    end_date: Optional[date] = None


# --------------------------- helpers ---------------------------


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Simple, deterministic AUC implementation (no ties handling beyond stable ordering).
    order = np.argsort(y_score)
    y_true_sorted = y_true[order]
    cum_pos = np.cumsum(y_true_sorted)
    total_pos = cum_pos[-1] if len(cum_pos) else 0
    total_neg = len(y_true_sorted) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5
    auc = (
        cum_pos[y_true_sorted == 0].sum()
        - (np.arange(len(y_true_sorted))[y_true_sorted == 0] * total_pos).sum()
    ) / (total_pos * total_neg)
    return float(auc)


def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2)) if len(y_true) else 0.0


def _train_logistic(
    X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 200
) -> np.ndarray:
    # Deterministic gradient descent with zero init
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        logits = X @ w
        preds = _sigmoid(logits)
        grad = X.T @ (preds - y) / len(y)
        w -= lr * grad
    return w


def _predict_logistic(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return _sigmoid(X @ w)


def _time_split(df: pd.DataFrame, frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = max(1, int(math.ceil(n * frac)))
    train = df.iloc[:cut]
    val = df.iloc[cut:]
    return train, val


def _one_hot(values: pd.Series, prefix: str) -> pd.DataFrame:
    if values.empty:
        return pd.DataFrame()
    return pd.get_dummies(values.fillna(""), prefix=prefix)


def _multi_hot(series: pd.Series, prefix: str) -> pd.DataFrame:
    rows: List[Dict[str, int]] = []
    for entry in series.fillna([]):
        row: Dict[str, int] = {}
        if isinstance(entry, list):
            for val in entry:
                key = f"{prefix}_{val}"
                row[key] = 1
        rows.append(row)
    return pd.DataFrame(rows).fillna(0).astype(int)


def _stats_summary() -> Dict[str, float]:
    stats_df = research_api.load_stats(research_api.StatsFilter())
    if stats_df.empty:
        return {}
    vals = {
        "stats_variance_mean": float(
            stats_df.get("variance", pd.Series(dtype=float)).mean(skipna=True)
        ),
        "stats_stability_mean": float(
            stats_df.get("stability", pd.Series(dtype=float)).mean(skipna=True)
        ),
    }
    return {k: (0.0 if math.isnan(v) else v) for k, v in vals.items()}


def _assemble_features(
    df: pd.DataFrame,
    include_lags: bool = True,
    stats_summary: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    # Base numeric features from key_features if present
    feature_cols: List[str] = []
    frames: List[pd.DataFrame] = []

    if "key_features" in df.columns:
        # expand dict columns
        expanded = pd.json_normalize(
            df["key_features"].apply(lambda x: x if isinstance(x, dict) else {})
        )
        frames.append(expanded)
        feature_cols.extend(list(expanded.columns))

    # Regime/session/timeframe one-hots
    for col, prefix in [("session_regime", "session"), ("timeframe", "tf")]:
        if col in df.columns:
            oh = _one_hot(df[col], prefix)
            frames.append(oh)
            feature_cols.extend(list(oh.columns))

    if "macro_regimes" in df.columns:
        mh = _multi_hot(df["macro_regimes"], "macro")
        frames.append(mh)
        feature_cols.extend(list(mh.columns))

    # Lagged outcomes (optional)
    if include_lags:
        for col in ["outcome_pnl", "outcome_hit", "outcome_max_drawdown"]:
            if col in df.columns:
                frames.append(df[[col]].rename(columns={col: f"lag_{col}"}))
                feature_cols.append(f"lag_{col}")

    if stats_summary:
        stats_block = pd.DataFrame([stats_summary] * len(df))
        frames.append(stats_block)
        feature_cols.extend(list(stats_block.columns))

    if not frames:
        return pd.DataFrame(), []

    feat_df = pd.concat(frames, axis=1).fillna(0)
    # Ensure deterministic column ordering
    feat_df = feat_df[sorted(feat_df.columns)]
    feature_cols = list(feat_df.columns)
    return feat_df, feature_cols


# --------------------------- label generation ---------------------------


def _shifted_label(
    df: pd.DataFrame, col: str, lookahead: int, default: float = 0.0
) -> pd.Series:
    shifted = df[col].shift(-lookahead)
    return shifted.fillna(default)


def _build_labels(df: pd.DataFrame, prior: str, cfg: TrainingConfig) -> pd.Series:
    if prior == "macro_up_prob":
        # Label 1 if next macro_regime contains risk_on_token
        future_macro = df["macro_regimes"].shift(-cfg.lookahead)
        return future_macro.apply(
            lambda x: 1.0 if isinstance(x, list) and cfg.risk_on_token in x else 0.0
        )

    if prior == "volatility_spike_prob":
        future_dd = _shifted_label(
            df, "outcome_max_drawdown", cfg.lookahead, default=0.0
        )
        future_var = _shifted_label(df, "outcome_pnl", cfg.lookahead, default=0.0).abs()
        return (
            (future_dd.abs() > cfg.volatility_threshold)
            | (future_var > cfg.volatility_threshold)
        ).astype(float)

    if prior == "regime_transition_prob":
        future_session = df["session_regime"].shift(-cfg.lookahead)
        return (future_session != df["session_regime"]).fillna(False).astype(float)

    if prior == "directional_confidence":
        future_pnl = _shifted_label(df, "outcome_pnl", cfg.lookahead, default=0.0)
        return (future_pnl > 0).astype(float)

    raise ValueError(f"Unknown prior: {prior}")


# --------------------------- training pipeline ---------------------------


def _prepare_dataset(
    prior: str, cfg: TrainingConfig
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    decisions = research_api.load_decisions(
        research_api.DecisionsFilter(start_date=cfg.start_date, end_date=cfg.end_date)
    )
    if decisions.empty:
        raise ValueError("No decision data available for training")

    # Sort by timestamp to ensure deterministic lookahead
    decisions = decisions.sort_values("timestamp_utc").reset_index(drop=True)

    y = _build_labels(decisions, prior, cfg)

    stats_summary = _stats_summary()
    X_df, feature_cols = _assemble_features(decisions, stats_summary=stats_summary)
    if X_df.empty:
        raise ValueError("No features available for training")

    # Align lengths
    X_df = X_df.iloc[: len(y)]
    y = y.iloc[: len(X_df)]

    # Drop rows with NaN labels resulting from lookahead at tail
    mask = y.notna()
    X_df = X_df[mask]
    y = y[mask]

    return X_df.reset_index(drop=True), y.reset_index(drop=True), feature_cols


def _train_and_evaluate(prior: str, cfg: TrainingConfig) -> Dict[str, object]:
    X_df, y_series, feature_cols = _prepare_dataset(prior, cfg)
    # Time-based split
    df_all = X_df.copy()
    df_all["label"] = y_series.values
    train_df, val_df = _time_split(df_all, cfg.train_frac)

    X_train = train_df.drop(columns=["label"]).to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=float)
    X_val = (
        val_df.drop(columns=["label"]).to_numpy(dtype=float)
        if len(val_df)
        else np.empty((0, X_train.shape[1]))
    )
    y_val = val_df["label"].to_numpy(dtype=float) if len(val_df) else np.empty((0,))

    # Add intercept
    X_train_aug = np.hstack([np.ones((len(X_train), 1)), X_train])
    X_val_aug = (
        np.hstack([np.ones((len(X_val), 1)), X_val])
        if len(X_val)
        else np.empty((0, X_train_aug.shape[1]))
    )

    weights = _train_logistic(X_train_aug, y_train)
    train_probs = _predict_logistic(X_train_aug, weights)
    val_probs = (
        _predict_logistic(X_val_aug, weights) if len(X_val_aug) else np.array([])
    )

    def _metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
        if len(y_true) == 0:
            return {"count": 0, "accuracy": 0.0, "brier": 0.0, "auc": 0.5}
        preds = (probs >= 0.5).astype(float)
        accuracy = float((preds == y_true).mean())
        return {
            "count": int(len(y_true)),
            "accuracy": accuracy,
            "brier": _brier_score(y_true, probs),
            "auc": _binary_auc(y_true, probs),
        }

    metrics = {
        "train": _metrics(y_train, train_probs),
        "val": _metrics(y_val, val_probs),
    }

    return {
        "weights": weights.tolist(),
        "feature_cols": feature_cols,
        "metrics": metrics,
    }


def _write_artifacts(
    model_name: str, result: Dict[str, object], cfg: TrainingConfig
) -> Path:
    ts = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    model_dir = cfg.models_dir / model_name
    _ensure_dir(model_dir)

    weights_path = model_dir / f"{model_name}_weights.pkl"
    with weights_path.open("wb") as f:
        pickle.dump(
            {"weights": result["weights"], "feature_cols": result["feature_cols"]}, f
        )

    meta = {
        "model_name": model_name,
        "version": MODEL_VERSION,
        "training_window": {
            "lookahead": cfg.lookahead,
        },
        "features_used": result.get("feature_cols", []),
        "metrics": result.get("metrics", {}),
        "timestamp_utc": ts,
    }
    meta_path = model_dir / f"{model_name}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return weights_path


# --------------------------- public entrypoints ---------------------------


def train_macro_up_model(cfg: Optional[TrainingConfig] = None) -> Dict[str, object]:
    cfg = cfg or TrainingConfig()
    result = _train_and_evaluate("macro_up_prob", cfg)
    path = _write_artifacts("macro_up_prob", result, cfg)
    return {**result, "artifact_path": str(path)}


def train_volatility_spike_model(
    cfg: Optional[TrainingConfig] = None,
) -> Dict[str, object]:
    cfg = cfg or TrainingConfig()
    result = _train_and_evaluate("volatility_spike_prob", cfg)
    path = _write_artifacts("volatility_spike_prob", result, cfg)
    return {**result, "artifact_path": str(path)}


def train_regime_transition_model(
    cfg: Optional[TrainingConfig] = None,
) -> Dict[str, object]:
    cfg = cfg or TrainingConfig()
    result = _train_and_evaluate("regime_transition_prob", cfg)
    path = _write_artifacts("regime_transition_prob", result, cfg)
    return {**result, "artifact_path": str(path)}


def train_directional_confidence_model(
    cfg: Optional[TrainingConfig] = None,
) -> Dict[str, object]:
    cfg = cfg or TrainingConfig()
    result = _train_and_evaluate("directional_confidence", cfg)
    path = _write_artifacts("directional_confidence", result, cfg)
    return {**result, "artifact_path": str(path)}
