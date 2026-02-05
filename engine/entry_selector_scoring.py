from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .decision_frame import DecisionFrame
from .entry_features import extract_entry_features
from .entry_selector_model import EntrySelectorArtifacts

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


def _prepare_row(entry_id: str, frame: DecisionFrame) -> pd.DataFrame:
    feats = extract_entry_features(entry_id, frame)
    row = {col: feats.get(col) for col in _CAT_COLS + _NUM_COLS}
    row["entry_model_id"] = entry_id
    for col in _CAT_COLS:
        row[col] = str(row.get(col, "UNKNOWN") or "UNKNOWN")
    for col in _NUM_COLS:
        val = row.get(col)
        try:
            row[col] = float(val)
        except Exception:
            row[col] = 0.0
    return pd.DataFrame([row])


def score_entry_selector(
    frame: DecisionFrame,
    eligible_models: List[str],
    artifacts: EntrySelectorArtifacts,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    encoder = artifacts.encoders.get("categorical")
    classes = list(artifacts.metadata.get("classes", []))

    for entry_id in eligible_models:
        row_df = _prepare_row(entry_id, frame)
        cat_matrix = encoder.transform(row_df[_CAT_COLS])
        num_matrix = row_df[_NUM_COLS].to_numpy(dtype=float)
        X = np.hstack([cat_matrix, num_matrix])

        prob_select = 0.0
        if hasattr(artifacts.classifier, "predict_proba"):
            proba = artifacts.classifier.predict_proba(X)
            if entry_id in classes:
                idx = classes.index(entry_id)
                prob_select = float(proba[0][idx])
            elif "NO_TRADE" in classes:
                prob_select = float(0.0)
        expected_R = (
            float(artifacts.regressor.predict(X)[0])
            if hasattr(artifacts.regressor, "predict")
            else 0.0
        )

        results[entry_id] = {"prob_select": prob_select, "expected_R": expected_R}

    return results
