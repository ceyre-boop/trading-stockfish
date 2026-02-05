import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd

_LABEL_ORDER = ["DISABLED", "DISCOURAGED", "ALLOWED", "PREFERRED"]

_DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "winrate_min": 0.4,
    "expected_R_promote": 0.5,
    "winrate_promote": 0.55,
    "min_samples": 5,
    "min_samples_promote": 8,
    "min_samples_stable": 3,
}


def _label_rank(label: Any) -> int:
    if label not in _LABEL_ORDER:
        return 0
    return _LABEL_ORDER.index(label)


def _downgrade(label: str) -> str:
    rank = _label_rank(label)
    return _LABEL_ORDER[max(0, rank - 1)]


def _upgrade(label: str) -> str:
    rank = _label_rank(label)
    return _LABEL_ORDER[min(len(_LABEL_ORDER) - 1, rank + 1)]


def _prepare_stress(stress_df: pd.DataFrame) -> pd.DataFrame:
    if stress_df is None:
        return pd.DataFrame()
    required_cols = [
        "entry_model_id",
        "market_profile_state",
        "session_profile",
        "liquidity_bias_side",
        "mean_regret",
        "mean_expected_R",
        "winrate",
        "sample_size",
    ]
    df = stress_df.copy()
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    return df


def _drift_lookup(drift_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if drift_df is None or drift_df.empty:
        return {}
    lookup: Dict[str, Dict[str, Any]] = {}
    for _, row in drift_df.iterrows():
        entry_id = row.get("entry_model_id")
        if not entry_id:
            continue
        lookup[entry_id] = row.to_dict()
    return lookup


def _current_label(row: pd.Series, current_policy_entries: pd.DataFrame) -> str:
    if current_policy_entries is None or current_policy_entries.empty:
        return "DISABLED"
    key_mask = (
        (current_policy_entries.get("entry_model_id") == row.get("entry_model_id"))
        & (
            current_policy_entries.get("market_profile_state")
            == row.get("market_profile_state")
        )
        & (current_policy_entries.get("session_profile") == row.get("session_profile"))
        & (
            current_policy_entries.get("liquidity_bias_side")
            == row.get("liquidity_bias_side")
        )
    )
    matches = current_policy_entries[key_mask]
    if matches.empty:
        return "DISABLED"
    lbl = matches.iloc[0].get("label")
    return lbl if isinstance(lbl, str) else "DISABLED"


def propose_entry_policy_updates(
    stress_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    current_policy_entries: pd.DataFrame,
    thresholds: Dict[str, Any],
) -> pd.DataFrame:
    thresholds = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    stress_df = _prepare_stress(stress_df)
    drift_lookup = _drift_lookup(drift_df)

    records: List[Dict[str, Any]] = []

    if stress_df.empty:
        return pd.DataFrame(
            columns=[
                "entry_model_id",
                "market_profile_state",
                "session_profile",
                "liquidity_bias_side",
                "current_label",
                "proposed_label",
                "reason",
                "winrate",
                "mean_expected_R",
                "mean_regret",
                "sample_size",
            ]
        )

    for _, row in stress_df.iterrows():
        entry_id = row.get("entry_model_id")
        if not entry_id:
            continue
        current_label = _current_label(row, current_policy_entries)

        drift_flags = drift_lookup.get(entry_id, {}).get("drift_flags")
        has_drift = False
        if isinstance(drift_flags, dict):
            has_drift = any(bool(v) for v in drift_flags.values())

        winrate = row.get("winrate")
        mean_expected_R = row.get("mean_expected_R")
        mean_regret = row.get("mean_regret")
        sample_size = row.get("sample_size")
        try:
            sample_size = int(sample_size)
        except Exception:
            sample_size = 0

        proposed_label = current_label
        reason = None

        if sample_size < thresholds.get("min_samples_stable", 0):
            proposed_label = current_label
            reason = "low_sample"
        else:
            low_winrate = (winrate is not None) and (
                winrate < thresholds.get("winrate_min")
            )
            if (has_drift or low_winrate) and sample_size >= thresholds.get(
                "min_samples", 0
            ):
                proposed_label = _downgrade(current_label)
                reason = "drift_detected" if has_drift else "low_winrate"
            else:
                promote = (
                    mean_expected_R is not None
                    and winrate is not None
                    and mean_expected_R >= thresholds.get("expected_R_promote")
                    and winrate >= thresholds.get("winrate_promote")
                    and sample_size >= thresholds.get("min_samples_promote", 0)
                    and not has_drift
                )
                if promote:
                    proposed_label = _upgrade(current_label)
                    reason = "high_expected_R"
                else:
                    proposed_label = current_label
                    reason = reason or "stable"

        records.append(
            {
                "entry_model_id": entry_id,
                "market_profile_state": row.get("market_profile_state"),
                "session_profile": row.get("session_profile"),
                "liquidity_bias_side": row.get("liquidity_bias_side"),
                "current_label": current_label,
                "proposed_label": proposed_label,
                "reason": reason,
                "winrate": winrate,
                "mean_expected_R": mean_expected_R,
                "mean_regret": mean_regret,
                "sample_size": sample_size,
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    result = result.sort_values(
        [
            "entry_model_id",
            "market_profile_state",
            "session_profile",
            "liquidity_bias_side",
        ],
        na_position="last",
    ).reset_index(drop=True)
    return result
