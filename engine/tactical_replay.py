import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .adversarial_replay import adversarial_replay
from .decision_frame import DecisionFrame
from .entry_consistency import validate_entry_consistency
from .entry_eligibility import get_eligible_entry_models
from .entry_features import extract_entry_features
from .entry_models import ENTRY_MODELS


def _load_logs_between(start_date, end_date, logs_dir: str) -> pd.DataFrame:
    paths = sorted(Path(logs_dir).glob("*.jsonl"))
    records = []
    for p in paths:
        try:
            with p.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
        except Exception:
            continue
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        if getattr(df["timestamp_utc"].dtype, "tz", None) is not None:
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
        df = df[(df["timestamp_utc"] >= start_ts) & (df["timestamp_utc"] <= end_ts)]
    return df


def run_replay_for_window(
    start_date, end_date, logs_dir: str, artifacts, brain_policy_entries: pd.DataFrame
) -> pd.DataFrame:
    """Convenience wrapper to load decision logs for a window and run adversarial replay.

    Deterministic ordering and no side-effects. Returns the replay dataframe.
    """
    decision_logs = _load_logs_between(start_date, end_date, logs_dir)
    replay_df = adversarial_replay(decision_logs, artifacts, brain_policy_entries)
    if not replay_df.empty:
        replay_df = replay_df.sort_values(
            ["timestamp_utc", "entry_model_id"], na_position="last"
        ).reset_index(drop=True)
    return replay_df


def _frame_from_dict(raw: Optional[Dict[str, Any]]) -> DecisionFrame:
    frame = DecisionFrame()
    if not isinstance(raw, dict):
        return frame
    for key, value in raw.items():
        if hasattr(frame, key):
            setattr(frame, key, value)
    return frame


def _match_policy(
    entry_id: str, frame: DecisionFrame, policy_df: pd.DataFrame
) -> Optional[pd.Series]:
    if policy_df is None or policy_df.empty:
        return None
    candidates = policy_df[policy_df.get("entry_model_id") == entry_id]
    if candidates.empty:
        return None
    mp_state = frame.market_profile_state or "UNKNOWN"
    session_profile = frame.session_profile or "UNKNOWN"
    liq_bias = None
    if isinstance(frame.liquidity_frame, dict):
        liq_bias = frame.liquidity_frame.get("bias")
    liq_bias = liq_bias or "UNKNOWN"

    exact = candidates[
        (candidates.get("market_profile_state") == mp_state)
        & (candidates.get("session_profile") == session_profile)
        & (candidates.get("liquidity_bias_side") == liq_bias)
    ]
    if not exact.empty:
        return exact.iloc[0]
    return candidates.iloc[0]


def replay_tactical_brain(
    decision_logs: pd.DataFrame,
    brain_policy_entries: pd.DataFrame,
) -> pd.DataFrame:
    if decision_logs is None or decision_logs.empty:
        return pd.DataFrame(
            columns=[
                "entry_id",
                "eligibility_match",
                "policy_label_match",
                "score_drift",
                "consistency_report",
            ]
        )

    records: List[Dict[str, Any]] = []

    for _, row in decision_logs.iterrows():
        frame_raw = row.get("decision_frame")
        frame = _frame_from_dict(frame_raw)

        logged_eligible = row.get("eligible_entry_models") or []
        if not logged_eligible and isinstance(frame, DecisionFrame):
            logged_eligible = frame.eligible_entry_models or []
        logged_labels = {}
        logged_scores = {}
        if isinstance(frame, DecisionFrame):
            logged_labels = frame.entry_brain_labels or {}
            logged_scores = frame.entry_brain_scores or {}
        elif isinstance(frame_raw, dict):
            logged_labels = frame_raw.get("entry_brain_labels", {}) or {}
            logged_scores = frame_raw.get("entry_brain_scores", {}) or {}

        recomputed_eligible = (
            set(get_eligible_entry_models(frame))
            if isinstance(frame, DecisionFrame)
            else set()
        )
        consistency_report = validate_entry_consistency(frame, brain_policy_entries)

        candidate_ids = (
            set(logged_eligible)
            | set(logged_labels.keys())
            | {row.get("entry_model_id")}
        )
        candidate_ids = {cid for cid in candidate_ids if cid}
        for entry_id in sorted(candidate_ids):
            policy_row = _match_policy(entry_id, frame, brain_policy_entries)
            policy_label = policy_row.get("label") if policy_row is not None else None
            logged_label = logged_labels.get(entry_id)
            policy_label_match = policy_label == logged_label

            eligibility_match = (entry_id in recomputed_eligible) == (
                entry_id in logged_eligible
            )

            logged_score = (
                logged_scores.get(entry_id) if isinstance(logged_scores, dict) else None
            )
            recomputed_score = None
            if policy_row is not None:
                recomputed_score = {
                    "prob_good": policy_row.get("prob_good"),
                    "expected_reward": policy_row.get("expected_reward"),
                }
            score_drift = None
            if isinstance(logged_score, dict) and isinstance(recomputed_score, dict):
                score_drift = {
                    "prob_good_diff": None,
                    "expected_reward_diff": None,
                }
                if (
                    logged_score.get("prob_good") is not None
                    and recomputed_score.get("prob_good") is not None
                ):
                    score_drift["prob_good_diff"] = float(
                        recomputed_score["prob_good"]
                    ) - float(logged_score.get("prob_good"))
                if (
                    logged_score.get("expected_reward") is not None
                    and recomputed_score.get("expected_reward") is not None
                ):
                    score_drift["expected_reward_diff"] = float(
                        recomputed_score["expected_reward"]
                    ) - float(logged_score.get("expected_reward"))

            features = None
            try:
                features = extract_entry_features(entry_id, frame)
            except Exception:
                features = None

            records.append(
                {
                    "entry_id": entry_id,
                    "eligibility_match": eligibility_match,
                    "policy_label_match": policy_label_match,
                    "score_drift": score_drift,
                    "consistency_report": consistency_report,
                    "features": features,
                }
            )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    result = result.sort_values(["entry_id"]).reset_index(drop=True)
    return result
