import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from .entry_selector_dataset import build_entry_selector_dataset
from .entry_selector_model import EntrySelectorArtifacts, train_entry_selector_model
from .entry_selector_scoring import score_entry_selector
from .tactical_replay import _load_logs_between

_DEFAULT_POLICY_PATH = Path("storage/policies/brain/brain_policy_entries.active.json")


def load_active_policy(policy_path: str | Path = _DEFAULT_POLICY_PATH) -> pd.DataFrame:
    """Load the active entry policy pointed to by the active pointer or direct JSON.

    Returns an empty DataFrame on errors; ordering is deterministic for reproducibility.
    """
    path = Path(policy_path)
    if not path.exists():
        return pd.DataFrame()

    def _load_policy(target: Path) -> pd.DataFrame:
        try:
            with target.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return pd.DataFrame()
        policy_payload = payload.get("policy") if isinstance(payload, dict) else payload
        try:
            df = pd.DataFrame(policy_payload or [])
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            return df
        sort_cols = [
            "entry_model_id",
            "market_profile_state",
            "session_profile",
            "liquidity_bias_side",
        ]
        present = [c for c in sort_cols if c in df.columns]
        if not present:
            return df.reset_index(drop=True)
        return df.sort_values(present, na_position="last").reset_index(drop=True)

    pointer_payload = None
    try:
        pointer_payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pointer_payload = None

    if isinstance(pointer_payload, dict) and pointer_payload.get("path"):
        target = Path(pointer_payload.get("path"))
        if target.exists():
            return _load_policy(target)

    return _load_policy(path)


def _coerce_artifacts(obj: Any) -> Optional[EntrySelectorArtifacts]:
    if isinstance(obj, EntrySelectorArtifacts):
        return obj
    if isinstance(obj, dict):
        if {"classifier", "regressor", "encoders", "metadata"}.issubset(
            set(obj.keys())
        ):
            return EntrySelectorArtifacts(
                classifier=obj.get("classifier"),
                regressor=obj.get("regressor"),
                encoders=obj.get("encoders"),
                metadata=obj.get("metadata"),
            )
    return None


def load_selector_artifacts(path: str | Path) -> Optional[EntrySelectorArtifacts]:
    resolved = Path(path)
    if not resolved.exists():
        return None
    try:
        payload = joblib.load(resolved)
    except Exception:
        return None
    return _coerce_artifacts(payload)


def _write_selector_artifacts(
    artifacts: EntrySelectorArtifacts, path: str | Path
) -> None:
    payload = {
        "classifier": artifacts.classifier,
        "regressor": artifacts.regressor,
        "encoders": artifacts.encoders,
        "metadata": artifacts.metadata,
    }
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path_obj)


def _frame_from_row(row: Dict[str, Any]):
    from .decision_frame import DecisionFrame

    frame = DecisionFrame()
    structural = (
        row.get("structural_context")
        if isinstance(row.get("structural_context"), dict)
        else {}
    )
    for key in [
        "market_profile_state",
        "session_profile",
        "vol_regime",
        "trend_regime",
    ]:
        val = row.get(key)
        if val is None and structural:
            val = structural.get(key)
        if hasattr(frame, key):
            setattr(frame, key, val)
    liq_bias = row.get("liquidity_bias_side")
    if liq_bias is None and structural:
        liq_bias = structural.get("liquidity_bias_side")
    frame.liquidity_frame = {"bias": liq_bias} if liq_bias is not None else {}
    frame.eligible_entry_models = (
        [row.get("entry_model_id")] if row.get("entry_model_id") else []
    )
    return frame


def build_entry_belief_map(
    artifacts: EntrySelectorArtifacts, replay_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    if replay_df is None or replay_df.empty:
        return []

    rows: List[Dict[str, Any]] = []
    for _, row in replay_df.iterrows():
        entry_id = row.get("entry_model_id")
        if not entry_id:
            continue
        frame = _frame_from_row(row)
        scores = score_entry_selector(frame, [entry_id], artifacts) or {}
        payload = scores.get(entry_id, {}) if isinstance(scores, dict) else {}
        rows.append(
            {
                "entry_model_id": entry_id,
                "prob_select": float(payload.get("prob_select", 0.0)),
                "expected_R": float(payload.get("expected_R", 0.0)),
                "entry_success": row.get("entry_success"),
                "regret": row.get("regret"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return []

    belief_records: List[Dict[str, Any]] = []
    grouped = df.groupby("entry_model_id", sort=True, dropna=True)
    for entry_id, group in grouped:
        success_mean = None
        if group["entry_success"].notna().any():
            success_mean = float(group["entry_success"].dropna().mean())
        regret_mean = None
        if group["regret"].notna().any():
            regret_mean = float(group["regret"].dropna().mean())
        belief_records.append(
            {
                "entry_model_id": entry_id,
                "avg_prob_select": float(group["prob_select"].mean()),
                "avg_expected_R": float(group["expected_R"].mean()),
                "empirical_success_rate": success_mean,
                "avg_regret": regret_mean,
                "count": int(len(group)),
            }
        )

    return sorted(belief_records, key=lambda r: r.get("entry_model_id") or "")


def run_tactical_cycle(
    start_date: str,
    end_date: str,
    logs_dir: str,
    selector_artifact_path: str,
    belief_map_path: str,
    policy_path: str,
    brain_policy_entries: pd.DataFrame,
    entry_selector_artifacts: Optional[EntrySelectorArtifacts] = None,
) -> Dict[str, Any]:
    if brain_policy_entries is None or brain_policy_entries.empty:
        brain_policy_entries = load_active_policy(policy_path)

    decision_logs = _load_logs_between(start_date, end_date, logs_dir)
    if not decision_logs.empty and "timestamp_utc" in decision_logs.columns:
        try:
            decision_logs["timestamp_utc"] = pd.to_datetime(
                decision_logs["timestamp_utc"], errors="coerce"
            )
            decision_logs = decision_logs.sort_values(
                ["timestamp_utc", "entry_model_id"], na_position="last"
            ).reset_index(drop=True)
        except Exception:
            decision_logs = decision_logs

    if decision_logs.empty:
        Path(belief_map_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(belief_map_path).open("w", encoding="utf-8") as handle:
            json.dump([], handle, ensure_ascii=True, indent=2)
        return {
            "rows_replayed": 0,
            "dataset_rows": 0,
            "artifact_path": selector_artifact_path,
            "belief_map_path": belief_map_path,
        }

    artifacts_for_replay = _coerce_artifacts(entry_selector_artifacts)
    if artifacts_for_replay is None:
        bootstrap_dataset = build_entry_selector_dataset(decision_logs)
        if bootstrap_dataset.empty:
            return {
                "rows_replayed": 0,
                "dataset_rows": 0,
                "artifact_path": selector_artifact_path,
                "belief_map_path": belief_map_path,
            }
        artifacts_for_replay = train_entry_selector_model(bootstrap_dataset)

    from .adversarial_replay import adversarial_replay

    replay_df = adversarial_replay(
        decision_logs, artifacts_for_replay, brain_policy_entries
    )
    if not replay_df.empty and "timestamp_utc" in replay_df.columns:
        replay_df = replay_df.sort_values(
            ["timestamp_utc", "entry_model_id"], na_position="last"
        ).reset_index(drop=True)

    dataset = build_entry_selector_dataset(replay_df)
    if not dataset.empty:
        unique_labels = dataset["chosen_flag"].unique().tolist()
        if unique_labels == [1]:
            dummy = dataset.iloc[0].copy()
            dummy["chosen_flag"] = 0
            dummy["entry_outcome_R"] = None
            dummy["entry_success"] = None
            dataset = pd.concat([dataset, pd.DataFrame([dummy])], ignore_index=True)
    artifacts = train_entry_selector_model(dataset)
    _write_selector_artifacts(artifacts, selector_artifact_path)

    belief_map = build_entry_belief_map(artifacts, replay_df)
    belief_path = Path(belief_map_path)
    belief_path.parent.mkdir(parents=True, exist_ok=True)
    with belief_path.open("w", encoding="utf-8") as handle:
        json.dump(belief_map, handle, ensure_ascii=True, indent=2, sort_keys=True)

    return {
        "rows_replayed": len(replay_df),
        "dataset_rows": len(dataset),
        "artifact_path": selector_artifact_path,
        "belief_map_path": belief_map_path,
    }
