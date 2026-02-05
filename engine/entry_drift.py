from typing import Any, Dict, List

import pandas as pd


def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    if series is None or weights is None or series.empty:
        return 0.0
    valid = ~(series.isna() | weights.isna())
    if not valid.any():
        return 0.0
    s = series[valid].astype(float)
    w = weights[valid].astype(float)
    if w.sum() == 0:
        return 0.0
    return float((s * w).sum() / w.sum())


def detect_entry_drift(
    stress_df: pd.DataFrame, thresholds: Dict[str, Any]
) -> pd.DataFrame:
    if stress_df is None or stress_df.empty:
        return pd.DataFrame(
            columns=["entry_model_id", "drift_flags", "drift_score", "ok"]
        )

    required_keys = [
        "regret",
        "score_drift",
        "eligibility_drift",
        "winrate_min",
        "min_samples",
    ]
    for key in required_keys:
        if key not in thresholds:
            thresholds[key] = 0.0

    grouped = stress_df.groupby("entry_model_id", dropna=False, sort=True)
    records: List[Dict[str, Any]] = []

    for entry_id, group in grouped:
        sample_weights = group.get("sample_size") if "sample_size" in group else None
        if sample_weights is None:
            sample_weights = pd.Series([1] * len(group), index=group.index)

        regret_mean = _weighted_mean(group.get("mean_regret"), sample_weights)
        score_drift_mean = _weighted_mean(group.get("score_drift_mean"), sample_weights)
        eligibility_rate = _weighted_mean(
            group.get("eligibility_drift_rate"), sample_weights
        )
        winrate = _weighted_mean(group.get("winrate"), sample_weights)
        total_samples = float(sample_weights.fillna(0).sum())

        flags = {
            "regret": bool(regret_mean > thresholds.get("regret", 0.0)),
            "score_drift": bool(score_drift_mean > thresholds.get("score_drift", 0.0)),
            "eligibility_drift": bool(
                eligibility_rate > thresholds.get("eligibility_drift", 0.0)
            ),
            "winrate": bool(winrate < thresholds.get("winrate_min", 0.0)),
            "sample_size": bool(total_samples < thresholds.get("min_samples", 0)),
        }
        drift_score = float(sum(1 for v in flags.values() if v))
        ok = bool(not any(flags.values()))

        records.append(
            {
                "entry_model_id": entry_id,
                "drift_flags": flags,
                "drift_score": drift_score,
                "ok": ok,
            }
        )

    result = pd.DataFrame(records)
    result = result.sort_values(["entry_model_id"], na_position="last").reset_index(
        drop=True
    )
    return result
