import json
import os
from typing import Any, Dict, List

import pandas as pd

_ARTIFACT_DIR = os.path.join("storage", "policies", "brain")
_REPORT_PATH = os.path.join(_ARTIFACT_DIR, "brain_policy_entries.report.json")


def _ensure_dir():
    os.makedirs(_ARTIFACT_DIR, exist_ok=True)


def build_entry_policy_report(
    diff_df: pd.DataFrame, performance_df: pd.DataFrame
) -> Dict[str, Any]:
    diff_df = diff_df.copy() if diff_df is not None else pd.DataFrame()
    performance_df = (
        performance_df.copy() if performance_df is not None else pd.DataFrame()
    )

    summary = {
        "num_upgrades": 0,
        "num_downgrades": 0,
        "num_disables": 0,
        "num_enables": 0,
    }

    label_order = ["DISABLED", "DISCOURAGED", "ALLOWED", "PREFERRED"]

    def _rank(label: Any) -> int:
        try:
            return label_order.index(label)
        except ValueError:
            return -1

    if diff_df.empty:
        report = {"summary": summary, "by_entry_model": []}
        _ensure_dir()
        with open(_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=True, indent=2)
        return report

    # Count summary metrics
    for _, row in diff_df.iterrows():
        from_label = row.get("from_label")
        to_label = row.get("to_label")
        if from_label is None or to_label is None:
            continue
        if from_label == to_label:
            continue
        if from_label == "DISABLED" and to_label != "DISABLED":
            summary["num_enables"] += 1
        if from_label != "DISABLED" and to_label == "DISABLED":
            summary["num_disables"] += 1

        if _rank(to_label) > _rank(from_label):
            summary["num_upgrades"] += 1
        if _rank(to_label) < _rank(from_label):
            summary["num_downgrades"] += 1

    # Build per-entry changes
    grouped = diff_df.groupby("entry_model_id", dropna=False, sort=True)
    by_entry: List[Dict[str, Any]] = []

    perf_lookup: Dict[tuple, Dict[str, Any]] = {}
    if not performance_df.empty:
        perf_df = performance_df.set_index(
            ["entry_model_id", "market_profile_state", "session_profile"], drop=False
        )
        for idx, row in perf_df.iterrows():
            perf_lookup[idx] = row.to_dict()

    for entry_id, group in grouped:
        changes: List[Dict[str, Any]] = []
        for _, row in group.iterrows():
            regime = {
                "market_profile_state": row.get("market_profile_state"),
                "session_profile": row.get("session_profile"),
                "liquidity_bias_side": row.get("liquidity_bias_side"),
            }
            metrics = {
                "winrate": row.get("winrate"),
                "mean_expected_R": row.get("mean_expected_R"),
                "mean_regret": row.get("mean_regret"),
                "sample_size": row.get("sample_size"),
            }
            perf_key = (
                entry_id,
                row.get("market_profile_state"),
                row.get("session_profile"),
            )
            if perf_key in perf_lookup:
                metrics.update(
                    {
                        "winrate": metrics.get("winrate")
                        or perf_lookup[perf_key].get("winrate"),
                        "mean_expected_R": metrics.get("mean_expected_R")
                        or perf_lookup[perf_key].get("mean_expected_R"),
                    }
                )

            changes.append(
                {
                    "from_label": row.get("from_label"),
                    "to_label": row.get("to_label"),
                    "regime": regime,
                    "reason": row.get("reason"),
                    "metrics": metrics,
                }
            )
        by_entry.append({"entry_model_id": entry_id, "changes": changes})

    report = {"summary": summary, "by_entry_model": by_entry}

    _ensure_dir()
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)

    return report
