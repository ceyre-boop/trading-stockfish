import json
import os
from typing import Any, Dict, List

import pandas as pd

_ARTIFACT_DIR = os.path.join("storage", "policies", "brain")
_PROPOSED_PATH = os.path.join(_ARTIFACT_DIR, "brain_policy_entries.proposed.json")
_DIFF_PATH = os.path.join(_ARTIFACT_DIR, "brain_policy_entries.diff.json")


def _ensure_dir():
    os.makedirs(_ARTIFACT_DIR, exist_ok=True)


def _key(df: pd.DataFrame) -> pd.Series:
    return (
        df.get("entry_model_id").astype(str)
        + "|"
        + df.get("market_profile_state").astype(str)
        + "|"
        + df.get("session_profile").astype(str)
        + "|"
        + df.get("liquidity_bias_side").astype(str)
    )


def _to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return df.to_dict(orient="records")


def diff_entry_policies(
    current_policy: pd.DataFrame, proposed_policy: pd.DataFrame
) -> pd.DataFrame:
    current_policy = (
        current_policy.copy() if current_policy is not None else pd.DataFrame()
    )
    proposed_policy = (
        proposed_policy.copy() if proposed_policy is not None else pd.DataFrame()
    )

    # Normalize proposed policy: prefer proposed_label if present
    if "proposed_label" in proposed_policy.columns:
        proposed_policy = proposed_policy.drop(columns=["label"], errors="ignore")
        proposed_policy = proposed_policy.rename(columns={"proposed_label": "label"})

    for df in (current_policy, proposed_policy):
        for col in [
            "entry_model_id",
            "market_profile_state",
            "session_profile",
            "liquidity_bias_side",
            "label",
            "reason",
            "winrate",
            "mean_expected_R",
            "mean_regret",
            "sample_size",
        ]:
            if col not in df.columns:
                df[col] = None

    current_policy["__key"] = _key(current_policy)
    proposed_policy["__key"] = _key(proposed_policy)

    merged = proposed_policy.merge(
        current_policy[["__key", "label"]],
        on="__key",
        how="left",
        suffixes=("", "_current"),
    )

    diffs: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        cur = row.get("label_current")
        new = row.get("label")
        cur_val = None if pd.isna(cur) else cur
        new_val = None if pd.isna(new) else new
        if cur_val == new_val:
            continue
        parts = row.get("__key", "||||").split("|")
        diffs.append(
            {
                "entry_model_id": parts[0],
                "market_profile_state": parts[1],
                "session_profile": parts[2],
                "liquidity_bias_side": parts[3],
                "from_label": cur_val,
                "to_label": new_val,
                "reason": row.get("reason"),
                "winrate": row.get("winrate"),
                "mean_expected_R": row.get("mean_expected_R"),
                "mean_regret": row.get("mean_regret"),
                "sample_size": row.get("sample_size"),
            }
        )

    diff_df = pd.DataFrame(diffs)
    diff_df = diff_df.sort_values(
        [
            "entry_model_id",
            "market_profile_state",
            "session_profile",
            "liquidity_bias_side",
        ],
        na_position="last",
    ).reset_index(drop=True)

    _ensure_dir()
    metadata = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_window": {"start": None, "end": None},
        "thresholds": None,
    }
    with open(_PROPOSED_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": metadata, "policy": _to_records(proposed_policy)},
            f,
            ensure_ascii=True,
            indent=2,
        )
    with open(_DIFF_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": metadata, "diff": _to_records(diff_df)},
            f,
            ensure_ascii=True,
            indent=2,
        )

    return diff_df
