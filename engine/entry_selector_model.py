from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


@dataclass
class EntrySelectorArtifacts:
    classifier: Any
    regressor: Any
    encoders: Dict[str, Any]
    metadata: Dict[str, Any]


_CAT_COLS = [
    "market_profile_state",
    "session_profile",
    "liquidity_bias_side",
    "liquidity_sweep_state",
    "vol_regime",
    "trend_regime",
    "risk_mae_bucket",
    "risk_mfe_bucket",
    "risk_horizon",
    "risk_aggressiveness",
    "entry_model_id",
]
_NUM_COLS = [
    "liquidity_nearest_target_distance",
    "displacement_score",
    "sweep_flag",
    "fvg_flag",
    "ob_flag",
    "ifvg_flag",
    "risk_expected_R",
]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for col in _CAT_COLS:
        if col not in data.columns:
            data[col] = "UNKNOWN"
        data[col] = data[col].fillna("UNKNOWN").astype(str)
    for col in _NUM_COLS:
        if col not in data.columns:
            data[col] = 0.0
        data[col] = data[col].astype(float).fillna(0.0)
    if "chosen_flag" not in data.columns:
        data["chosen_flag"] = 0
    if "entry_outcome_R" not in data.columns:
        data["entry_outcome_R"] = np.nan
    return data


def _encode_features(df: pd.DataFrame, encoder: OneHotEncoder) -> np.ndarray:
    cat_matrix = encoder.transform(df[_CAT_COLS])
    num_matrix = df[_NUM_COLS].to_numpy(dtype=float)
    return np.hstack([cat_matrix, num_matrix])


def train_entry_selector_model(
    df: pd.DataFrame,
    random_state: int = 42,
    sample_weight: Optional[Iterable[float]] = None,
    use_priors: bool = False,
) -> EntrySelectorArtifacts:
    if df is None or df.empty:
        raise ValueError("Dataset is empty; cannot train entry selector.")

    data = _prepare(df)

    label = data["entry_model_id"].where(data["chosen_flag"] == 1, other="NO_TRADE")

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=float
        )
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=float)
    encoder.fit(data[_CAT_COLS])

    X = _encode_features(data, encoder)

    clf_sample_weight = None
    if sample_weight is not None:
        clf_sample_weight = pd.Series(sample_weight).to_numpy(dtype=float)
        if len(clf_sample_weight) != len(data):
            raise ValueError("sample_weight length must match dataset length")

    # Older sklearn versions shipped in this project do not support the multi_class kwarg;
    # fall back to the default constructor when the signature rejects it.
    try:
        clf = LogisticRegression(
            max_iter=500, multi_class="ovr", random_state=random_state
        )
    except TypeError:
        clf = LogisticRegression(max_iter=500, random_state=random_state)
    if clf_sample_weight is not None:
        clf.fit(X, label, sample_weight=clf_sample_weight)
    else:
        clf.fit(X, label)

    reg_data = data.dropna(subset=["entry_outcome_R"])
    reg = GradientBoostingRegressor(random_state=random_state)
    if reg_data.empty:
        reg.fit(np.zeros((1, X.shape[1])), [0.0])
    else:
        X_reg = _encode_features(reg_data, encoder)
        y_reg = reg_data["entry_outcome_R"].astype(float).to_numpy()
        reg_sample_weight = None
        if clf_sample_weight is not None:
            reg_sample_weight = clf_sample_weight[reg_data.index]
        reg.fit(X_reg, y_reg, sample_weight=reg_sample_weight)

    metadata = {
        "classes": list(clf.classes_),
        "cat_columns": list(_CAT_COLS),
        "num_columns": list(_NUM_COLS),
        "random_state": random_state,
        "use_priors": bool(use_priors),
        "sample_weight_sum": (
            float(clf_sample_weight.sum()) if clf_sample_weight is not None else None
        ),
    }

    return EntrySelectorArtifacts(
        classifier=clf,
        regressor=reg,
        encoders={"categorical": encoder},
        metadata=metadata,
    )
