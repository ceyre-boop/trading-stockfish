from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from .entry_selector_dataset import build_entry_selector_dataset
from .entry_selector_model import train_entry_selector_model


def _coerce_decision_logs(replay_df: pd.DataFrame) -> pd.DataFrame:
    # If replay_df already looks like decision logs, return as-is
    required = {"entry_model_id", "decision_frame"}
    if required.issubset(set(replay_df.columns)):
        return replay_df

    records = []
    for _, row in replay_df.iterrows():
        entry_id = row.get("entry_model_id")
        if not entry_id:
            continue
        structural = (
            row.get("structural_context")
            if isinstance(row.get("structural_context"), dict)
            else {}
        )
        frame_dict = {
            "market_profile_state": row.get("market_profile_state")
            or structural.get("market_profile_state"),
            "session_profile": row.get("session_profile"),
            "liquidity_frame": {
                "bias": row.get("liquidity_bias_side")
                or structural.get("liquidity_bias_side")
            },
            "vol_regime": row.get("vol_regime") or structural.get("vol_regime"),
            "trend_regime": row.get("trend_regime") or structural.get("trend_regime"),
        }
        records.append(
            {
                "timestamp_utc": row.get("timestamp_utc"),
                "entry_model_id": entry_id,
                "entry_outcome": row.get("entry_outcome_R") or row.get("expected_R"),
                "eligible_entry_models": [
                    e for e in {entry_id, row.get("best_entry_model")} if e
                ],
                "decision_frame": frame_dict,
            }
        )
    return pd.DataFrame(records)


def retrain_entry_selector_from_replay(
    replay_df: pd.DataFrame,
    save_path: str,
    priors: Optional[Dict[str, Dict[str, Any]]] = None,
    use_priors: bool = False,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Any:
    """Retrain entry selector artifacts from replay-derived data.

    Builds a dataset, trains classifier/regressor, saves artifacts to save_path (joblib) and returns artifacts.
    Deterministic ordering enforced. Prior-driven weighting is optional and disabled by default.
    """
    decision_logs = _coerce_decision_logs(replay_df)
    if "timestamp_utc" in decision_logs.columns:
        decision_logs = decision_logs.sort_values("timestamp_utc").reset_index(
            drop=True
        )
    dataset = build_entry_selector_dataset(decision_logs)

    weights = None
    if use_priors and priors:
        weight_map = {}
        for entry_id, prior in priors.items():
            if not isinstance(prior, dict):
                continue
            conf = float(prior.get("confidence", 0.0) or 0.0)
            exp_r = float(prior.get("base_expected_R_prior", 0.0) or 0.0)
            weight_map[entry_id] = 1.0 + alpha * conf + beta * exp_r
        weights = dataset["entry_model_id"].map(weight_map).fillna(1.0).astype(float)

    artifacts = train_entry_selector_model(
        dataset,
        sample_weight=weights.to_numpy() if weights is not None else None,
        use_priors=use_priors,
    )

    payload = {
        "classifier": artifacts.classifier,
        "regressor": artifacts.regressor,
        "encoders": artifacts.encoders,
        "metadata": artifacts.metadata,
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, save_path)
    return artifacts
