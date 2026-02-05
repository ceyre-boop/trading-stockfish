from typing import Any, Dict, List, Optional

import pandas as pd


def _mode_by_metric(
    group: pd.DataFrame, key: str, metric: str, pick_max: bool
) -> Optional[Any]:
    if key not in group.columns or metric not in group.columns or group.empty:
        return None
    agg = group.groupby(key)[metric].mean()
    if agg.empty:
        return None
    if pick_max:
        best_val = agg.max()
        candidates = sorted([k for k, v in agg.items() if v == best_val])
    else:
        best_val = agg.min()
        candidates = sorted([k for k, v in agg.items() if v == best_val])
    return candidates[0] if candidates else None


def attribute_entry_performance(replay_df: pd.DataFrame) -> pd.DataFrame:
    if replay_df is None or replay_df.empty:
        return pd.DataFrame(
            columns=[
                "entry_model_id",
                "total_trades",
                "total_R",
                "avg_R",
                "winrate",
                "best_regime",
                "worst_regime",
                "best_session_profile",
                "worst_session_profile",
            ]
        )

    df = replay_df.copy()
    for col in [
        "entry_model_id",
        "expected_R",
        "entry_success",
        "market_profile_state",
        "session_profile",
    ]:
        if col not in df.columns:
            df[col] = None

    grouped = df.groupby("entry_model_id", dropna=False, sort=True)
    records: List[Dict[str, Any]] = []

    for entry_id, group in grouped:
        total_trades = len(group)
        total_R = float(
            group.get("expected_R", pd.Series(dtype=float)).astype(float).sum()
        )
        avg_R = (
            float(group.get("expected_R", pd.Series(dtype=float)).astype(float).mean())
            if total_trades
            else 0.0
        )
        winrate = None
        if "entry_success" in group:
            col = group["entry_success"].dropna()
            if not col.empty:
                winrate = float(col.astype(float).mean())

        best_regime = _mode_by_metric(
            group, "market_profile_state", "expected_R", pick_max=True
        )
        worst_regime = _mode_by_metric(
            group, "market_profile_state", "expected_R", pick_max=False
        )
        best_session = _mode_by_metric(
            group, "session_profile", "expected_R", pick_max=True
        )
        worst_session = _mode_by_metric(
            group, "session_profile", "expected_R", pick_max=False
        )

        records.append(
            {
                "entry_model_id": entry_id,
                "total_trades": total_trades,
                "total_R": total_R,
                "avg_R": avg_R,
                "winrate": winrate,
                "best_regime": best_regime,
                "worst_regime": worst_regime,
                "best_session_profile": best_session,
                "worst_session_profile": worst_session,
            }
        )

    result = pd.DataFrame(records)
    result = result.sort_values(["entry_model_id"], na_position="last").reset_index(
        drop=True
    )
    return result
