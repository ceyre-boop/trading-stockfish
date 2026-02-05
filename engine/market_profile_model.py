from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from sklearn.ensemble import GradientBoostingClassifier

from .market_profile_features import MarketProfileFeatures

CLASSES = ["ACCUMULATION", "MANIPULATION", "DISTRIBUTION", "UNKNOWN"]


@dataclass
class TrainedMarketProfileModel:
    model: GradientBoostingClassifier
    feature_order: List[str]
    categorical_columns: List[str]


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "coarse_label" not in df.columns:
        raise ValueError("coarse_label column required for training")
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    df = df.fillna(0)
    return df


def train_market_profile_model(
    df: pd.DataFrame, random_state: int = 42
) -> TrainedMarketProfileModel:
    df = _prepare_df(df)
    feature_cols = [c for c in df.columns if c not in {"coarse_label", "human_label"}]
    target = df["coarse_label"].fillna("UNKNOWN")
    feature_df = df[feature_cols].copy()
    # Treat both object and pandas string dtype as categorical
    categorical = [
        c
        for c in feature_df.columns
        if is_object_dtype(feature_df[c]) or is_string_dtype(feature_df[c])
    ]
    X = pd.get_dummies(feature_df, columns=categorical, dummy_na=False)

    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X, target)
    return TrainedMarketProfileModel(
        model=model,
        feature_order=list(X.columns),
        categorical_columns=categorical,
    )


def predict_market_profile_proba(
    model: TrainedMarketProfileModel, features: MarketProfileFeatures
) -> Dict[str, float]:
    vec = features.to_vector()
    df_row = pd.DataFrame([vec])
    if model.categorical_columns:
        df_row = pd.get_dummies(
            df_row, columns=model.categorical_columns, dummy_na=False
        )
    df_row = df_row.reindex(columns=model.feature_order, fill_value=0)
    proba_arr = model.model.predict_proba(df_row)
    # Align classes; classifier might order differently, so map using model.classes_
    probs = {cls: 0.0 for cls in CLASSES}
    for cls, p in zip(model.model.classes_, proba_arr[0]):
        probs[str(cls)] = float(p)
    missing = set(CLASSES) - set(model.model.classes_)
    if missing:
        # redistribute missing to UNKNOWN proportionally 0
        pass
    # Normalize to sum 1 for determinism
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    return probs
