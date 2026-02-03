"""
Registry-driven stats harness.
Selects features by role/tag from config/feature_registry.json and computes
basic distributions/correlations deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from config.feature_registry import FeatureSpec, load_registry


@dataclass(frozen=True)
class FeatureSelection:
    features: List[str]
    specs: List[FeatureSpec]


def select_features(
    tags: Optional[Iterable[str]] = None, roles: Optional[Iterable[str]] = None
) -> FeatureSelection:
    registry = load_registry()
    tag_set = set(tags or [])
    role_set = set(roles or [])

    def _selected(spec: FeatureSpec) -> bool:
        if tag_set and not tag_set.intersection(spec.tags):
            return False
        if role_set and not role_set.intersection(spec.role):
            return False
        return True

    specs = [spec for spec in registry.specs.values() if _selected(spec)]
    names = [spec.name for spec in specs]
    return FeatureSelection(features=names, specs=specs)


def compute_distributions(
    df: pd.DataFrame, selection: FeatureSelection
) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for name in selection.features:
        if name not in df.columns:
            continue
        series = df[name].dropna()
        if series.empty:
            continue
        stats[name] = {
            "count": int(series.count()),
            "mean": (
                float(series.mean()) if pd.api.types.is_numeric_dtype(series) else None
            ),
            "std": (
                float(series.std()) if pd.api.types.is_numeric_dtype(series) else None
            ),
            "min": (
                float(series.min()) if pd.api.types.is_numeric_dtype(series) else None
            ),
            "max": (
                float(series.max()) if pd.api.types.is_numeric_dtype(series) else None
            ),
        }
    return stats


def compute_correlations(df: pd.DataFrame, selection: FeatureSelection) -> pd.DataFrame:
    cols = [c for c in selection.features if c in df.columns]
    if not cols:
        return pd.DataFrame()
    numeric = df[cols].select_dtypes(include=["number"])
    if numeric.empty:
        return pd.DataFrame()
    return numeric.corr()


def run_stats(
    df: pd.DataFrame,
    tags: Optional[Iterable[str]] = None,
    roles: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    selection = select_features(tags=tags, roles=roles)
    distributions = compute_distributions(df, selection)
    correlations = compute_correlations(df, selection)
    return {
        "features": selection.features,
        "distributions": distributions,
        "correlations": correlations.to_dict(),
    }
