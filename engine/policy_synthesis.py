import json
from pathlib import Path
from typing import Dict

import pandas as pd


def _compute_stats(df: pd.DataFrame) -> Dict[str, float]:
    realized = df["label_realized_R"].dropna()
    sample_count = int(realized.shape[0])
    mean_ev = float(realized.mean()) if sample_count > 0 else 0.0
    median_ev = float(realized.median()) if sample_count > 0 else 0.0
    winrate = float((realized > 0).mean()) if sample_count > 0 else 0.0
    variance = float(realized.var()) if sample_count > 1 else 0.0
    return {
        "mean_EV": mean_ev,
        "median_EV": median_ev,
        "winrate": winrate,
        "variance": variance,
        "sample_count": sample_count,
    }


def _label_for_stats(stats: Dict[str, float], min_samples: int) -> str:
    if stats["sample_count"] < min_samples:
        return "DISABLED"
    mean_ev = stats["mean_EV"]
    if mean_ev < 0:
        return "DISCOURAGED"
    if mean_ev < 0.2:
        return "ALLOWED"
    return "PREFERRED"


def synthesize_entry_policy(
    ev_dataset: pd.DataFrame,
    *,
    version: str = "v1",
    min_samples: int = 50,
    output_path: str | Path = "brain_policy_entries.learned.json",
) -> Dict[str, str]:
    if ev_dataset is None or ev_dataset.empty:
        return {}

    # Filter to open actions only
    open_df = ev_dataset[ev_dataset["action_type"].isin(["OPEN_LONG", "OPEN_SHORT"])]
    if open_df.empty:
        return {}

    labels: Dict[str, str] = {}
    for entry_id, group in open_df.groupby("entry_model_id"):
        stats = _compute_stats(group)
        label = _label_for_stats(stats, min_samples)
        labels[str(entry_id)] = label

    ordered = dict(sorted(labels.items(), key=lambda kv: kv[0]))

    artifact = {
        "version": version,
        "policy": [{"entry_model_id": k, "label": v} for k, v in ordered.items()],
    }
    try:
        Path(output_path).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    except Exception:
        # If writing fails, still return the labels
        return ordered

    return ordered
