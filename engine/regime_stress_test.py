from typing import Any, Dict, List

import pandas as pd


def stress_test_entries_by_regime(replay_df: pd.DataFrame) -> pd.DataFrame:
    if replay_df is None or replay_df.empty:
        return pd.DataFrame(
            columns=[
                "market_profile_state",
                "session_profile",
                "liquidity_bias_side",
                "entry_model_id",
                "mean_regret",
                "mean_expected_R",
                "winrate",
                "eligibility_drift_rate",
                "score_drift_mean",
                "sample_size",
            ]
        )

    cols = [
        "market_profile_state",
        "session_profile",
        "liquidity_bias_side",
        "entry_model_id",
        "regret",
        "expected_R",
        "entry_success",
        "eligibility_drift",
        "score_drift",
    ]
    df = replay_df.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = None

    grouped = df.groupby(
        [
            "market_profile_state",
            "session_profile",
            "liquidity_bias_side",
            "entry_model_id",
        ],
        dropna=False,
        sort=True,
    )

    records: List[Dict[str, Any]] = []
    for keys, group in grouped:
        mp_state, session, liq_bias, entry_id = keys
        sample_size = len(group)
        mean_regret = (
            float(group["regret"].astype(float).mean())
            if not group["regret"].isna().all()
            else 0.0
        )
        mean_expected = (
            float(group["expected_R"].astype(float).mean())
            if not group["expected_R"].isna().all()
            else 0.0
        )
        winrate = None
        if not group["entry_success"].isna().all():
            winrate = float(group["entry_success"].astype(float).mean())
        eligibility_rate = None
        if not group["eligibility_drift"].isna().all():
            eligibility_rate = float(group["eligibility_drift"].astype(float).mean())
        score_drift_mean = None
        if not group["score_drift"].isna().all():
            score_drift_mean = float(group["score_drift"].astype(float).mean())

        records.append(
            {
                "market_profile_state": mp_state,
                "session_profile": session,
                "liquidity_bias_side": liq_bias,
                "entry_model_id": entry_id,
                "mean_regret": mean_regret,
                "mean_expected_R": mean_expected,
                "winrate": winrate,
                "eligibility_drift_rate": eligibility_rate,
                "score_drift_mean": score_drift_mean,
                "sample_size": sample_size,
            }
        )

    result = pd.DataFrame(records)
    result = result.sort_values(
        [
            "market_profile_state",
            "entry_model_id",
            "session_profile",
            "liquidity_bias_side",
        ],
        na_position="last",
    ).reset_index(drop=True)
    return result
