from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .decision_actions import DecisionRecord
from .decision_frame import DecisionFrame
from .decision_logger import build_decision_record
from .entry_brain import BrainPolicy
from .entry_eligibility import get_eligible_entry_models
from .entry_features import extract_entry_features
from .entry_models import ENTRY_MODELS
from .entry_selector_scoring import score_entry_selector


def _frame_from_dict(raw: Optional[Dict[str, Any]]) -> DecisionFrame:
    frame = DecisionFrame()
    if not isinstance(raw, dict):
        return frame
    for key, value in raw.items():
        if hasattr(frame, key):
            setattr(frame, key, value)
    return frame


def _preferred_entries(policy_df: pd.DataFrame) -> set:
    if policy_df is None or policy_df.empty:
        return set()
    preferred = policy_df[policy_df.get("label") == "PREFERRED"]
    return set(preferred.get("entry_model_id").dropna().tolist())


def adversarial_replay(
    decision_logs: pd.DataFrame,
    entry_selector_artifacts,
    brain_policy_entries: Optional[pd.DataFrame] = None,
    candidate_path: Optional[str | Path] = None,
    outcome_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    if decision_logs is None or decision_logs.empty:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "market_profile_state",
                "session_profile",
                "liquidity_bias_side",
                "entry_model_id",
                "best_entry_model",
                "regret",
                "eligibility_drift",
                "score_drift",
                "policy_alignment",
                "expected_R",
                "entry_success",
                "structural_context",
            ]
        )

    preferred_ids = _preferred_entries(brain_policy_entries)
    policy_map: Dict[str, Any] = {}
    if brain_policy_entries is not None:
        policy_map = {
            row.get("entry_model_id"): row.get("label")
            for _, row in brain_policy_entries.iterrows()
        }
    multipliers = BrainPolicy.DEFAULT_MULTIPLIERS

    records: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    outcome_rows: List[Dict[str, Any]] = []

    for _, row in decision_logs.iterrows():
        frame_raw = row.get("decision_frame")
        frame = _frame_from_dict(frame_raw)

        logged_eligible = row.get("eligible_entry_models") or []
        if not logged_eligible and isinstance(frame, DecisionFrame):
            logged_eligible = frame.eligible_entry_models or []
        logged_scores = {}
        if isinstance(frame_raw, dict):
            logged_scores = frame_raw.get("entry_brain_scores") or {}
        chosen_entry = row.get("entry_model_id") or frame.chosen_entry_model_id

        if logged_eligible:
            recomputed_eligible = list(logged_eligible)
        else:
            recomputed_eligible = get_eligible_entry_models(frame)
        selector_scores = score_entry_selector(
            frame, recomputed_eligible, entry_selector_artifacts
        )

        best_entry = None
        best_expected = float("-inf")
        for entry_id, payload in selector_scores.items():
            exp_r = float(payload.get("expected_R", 0.0))
            if exp_r > best_expected or (
                exp_r == best_expected and entry_id < (best_entry or entry_id)
            ):
                best_entry = entry_id
                best_expected = exp_r

        chosen_score = selector_scores.get(chosen_entry, {}) if chosen_entry else {}
        chosen_expected = float(chosen_score.get("expected_R", 0.0))
        logged_score = None
        if isinstance(logged_scores, dict) and chosen_entry:
            logged_score = logged_scores.get(chosen_entry)
        expected_logged = None
        if isinstance(logged_score, dict):
            expected_logged = logged_score.get("expected_R")
            if expected_logged is None:
                expected_logged = logged_score.get("expected_reward")
            if expected_logged is not None:
                try:
                    expected_logged = float(expected_logged)
                except Exception:
                    expected_logged = None

        score_drift = None
        if expected_logged is not None:
            score_drift = abs(chosen_expected - expected_logged)

        regret = (
            (best_expected - chosen_expected) if best_expected != float("-inf") else 0.0
        )

        liq_bias = None
        if isinstance(frame.liquidity_frame, dict):
            liq_bias = frame.liquidity_frame.get("bias")

        outcome_val = row.get("entry_outcome")
        if outcome_val is None:
            outcome_val = row.get("outcome")
        entry_success = None
        if outcome_val is not None:
            try:
                entry_success = 1 if float(outcome_val) > 0 else 0
            except Exception:
                entry_success = None

        structural_context = {
            "market_profile_state": frame.market_profile_state,
            "liquidity_bias_side": liq_bias,
            "vol_regime": frame.vol_regime,
            "trend_regime": frame.trend_regime,
        }

        records.append(
            {
                "timestamp_utc": row.get("timestamp_utc"),
                "market_profile_state": frame.market_profile_state,
                "session_profile": frame.session_profile,
                "liquidity_bias_side": liq_bias,
                "entry_model_id": chosen_entry,
                "best_entry_model": best_entry,
                "regret": regret,
                "eligibility_drift": set(recomputed_eligible) != set(logged_eligible),
                "score_drift": score_drift,
                "policy_alignment": (
                    chosen_entry in preferred_ids if chosen_entry else False
                ),
                "expected_R": chosen_expected,
                "entry_success": entry_success,
                "structural_context": structural_context,
            }
        )

        # Candidate rows per entry model
        ts_val = row.get("timestamp_utc")
        bar_index = row.get("bar_index") or row.get("index")
        for entry_id, model in ENTRY_MODELS.items():
            eligible = entry_id in recomputed_eligible
            payload = selector_scores.get(entry_id, {}) if eligible else {}
            raw_score = (
                float(payload.get("expected_R", 0.0) or 0.0) if payload else None
            )
            policy_label = policy_map.get(entry_id, "DISABLED")
            multiplier = multipliers.get(policy_label, 0.0)
            adjusted_score = (
                float((raw_score or 0.0) * multiplier) if raw_score is not None else 0.0
            )
            risk_snapshot = (
                dict(model.risk_profile) if hasattr(model, "risk_profile") else {}
            )
            features = extract_entry_features(entry_id, frame)

            candidate_rows.append(
                {
                    "timestamp_utc": ts_val,
                    "bar_index": bar_index,
                    "entry_model_id": entry_id,
                    "eligible": bool(eligible),
                    "raw_score": raw_score,
                    "adjusted_score": adjusted_score,
                    "policy_label": policy_label,
                    "risk_profile": risk_snapshot,
                    "risk_expected_R": features.get("risk_expected_R"),
                    "risk_mae_bucket": features.get("risk_mae_bucket"),
                    "risk_mfe_bucket": features.get("risk_mfe_bucket"),
                    "risk_time_horizon": features.get("risk_horizon"),
                    "risk_aggressiveness": features.get("risk_aggressiveness"),
                    "structural_context": structural_context,
                }
            )

        # Outcome rows for executed entries
        if chosen_entry:
            outcome_rows.append(
                {
                    "timestamp_utc": row.get("timestamp_utc"),
                    "bar_index": bar_index,
                    "entry_model_id": chosen_entry,
                    "realized_R": outcome_val,
                    "max_adverse_excursion": row.get("max_adverse_excursion"),
                    "max_favorable_excursion": row.get("max_favorable_excursion"),
                    "time_to_outcome": row.get("time_to_outcome"),
                    "fill_price": row.get("fill_price"),
                    "exit_price": row.get("exit_price"),
                }
            )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    if "timestamp_utc" in result.columns:
        try:
            result["timestamp_utc"] = pd.to_datetime(
                result["timestamp_utc"], errors="coerce"
            )
        except Exception:
            pass

    result = result.sort_values(
        ["timestamp_utc", "entry_model_id"], na_position="last"
    ).reset_index(drop=True)
    if candidate_rows:
        candidate_df = pd.DataFrame(candidate_rows)
        candidate_df = candidate_df.sort_values(
            [
                col
                for col in ["timestamp_utc", "bar_index", "entry_model_id"]
                if col in candidate_df.columns
            ],
            na_position="last",
        ).reset_index(drop=True)
        target = (
            Path(candidate_path)
            if candidate_path
            else Path("storage/reports/entry_candidates.parquet")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            candidate_df.to_parquet(target, index=False)
        except Exception:
            candidate_df.to_json(
                target.with_suffix(".json"), orient="records", lines=True
            )

    if outcome_rows:
        outcome_df = pd.DataFrame(outcome_rows)
        outcome_df = outcome_df.sort_values(
            [
                col
                for col in ["timestamp_utc", "bar_index", "entry_model_id"]
                if col in outcome_df.columns
            ],
            na_position="last",
        ).reset_index(drop=True)
        target = (
            Path(outcome_path)
            if outcome_path
            else Path("storage/reports/entry_outcomes.parquet")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            outcome_df.to_parquet(target, index=False)
        except Exception:
            outcome_df.to_json(
                target.with_suffix(".json"), orient="records", lines=True
            )

    return result


def collect_decision_records(decision_logs: pd.DataFrame) -> List[DecisionRecord]:
    """Thin adapter to convert raw decision logs into DecisionRecord objects for EV dataset building.

    Does not modify replay behavior; purely transforms existing logged decisions.
    """

    records: List[DecisionRecord] = []
    if decision_logs is None or decision_logs.empty:
        return records
    for _, row in decision_logs.iterrows():
        try:
            records.append(build_decision_record(row.to_dict()))
        except Exception:
            continue
    return records
