import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd

_REQUIRED_COLUMNS = [
    "entry_model_id",
    "market_profile_state",
    "session_profile",
    "liquidity_bias_side",
    "label",
]


def load_proposed_policy(path: str) -> pd.DataFrame:
    return _load_policy(Path(path))


def load_current_policy(path: str) -> pd.DataFrame:
    return _load_policy(Path(path))


def _load_policy(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=_REQUIRED_COLUMNS)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        policy = payload.get("policy") if isinstance(payload, dict) else payload
        df = pd.DataFrame(policy or [])
    except Exception:
        df = pd.DataFrame(columns=_REQUIRED_COLUMNS)
    return df


def _validate_policy(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Policy missing required columns: {missing}")
    if df["entry_model_id"].isna().any():
        raise ValueError("Policy contains missing entry_model_id")


def _active_pointer_path(storage_dir: str) -> Path:
    return Path(storage_dir) / "brain_policy_entries.active.json"


def apply_entry_policy_update(
    proposed_policy: pd.DataFrame,
    current_policy: pd.DataFrame,
    approval: bool,
    version_tag: str,
    storage_dir: str,
) -> Dict[str, Any]:
    storage_path = Path(storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)
    active_pointer = _active_pointer_path(storage_dir)

    current_version = None
    if active_pointer.exists():
        try:
            payload = json.loads(active_pointer.read_text(encoding="utf-8"))
            current_version = payload.get("active_version")
        except Exception:
            current_version = None

    if not approval:
        return {
            "applied": False,
            "reason": "not approved",
            "current_version": current_version,
        }

    proposed_policy = (
        proposed_policy.copy() if proposed_policy is not None else pd.DataFrame()
    )
    # Normalize column name if using proposed_label/current_label layout
    if (
        "proposed_label" in proposed_policy.columns
        and "label" not in proposed_policy.columns
    ):
        proposed_policy["label"] = proposed_policy["proposed_label"]

    _validate_policy(proposed_policy)

    # Deterministic ordering
    proposed_policy = proposed_policy.sort_values(
        [
            "entry_model_id",
            "market_profile_state",
            "session_profile",
            "liquidity_bias_side",
        ],
        na_position="last",
    ).reset_index(drop=True)

    version_filename = f"brain_policy_entries.{version_tag}.json"
    version_path = storage_path / version_filename
    payload = {
        "metadata": {
            "version_tag": version_tag,
            "created_ts": pd.Timestamp.now(tz="UTC").isoformat(),
        },
        "policy": proposed_policy.to_dict(orient="records"),
    }
    with version_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)

    pointer_payload = {"active_version": version_tag, "path": str(version_path)}
    with active_pointer.open("w", encoding="utf-8") as handle:
        json.dump(pointer_payload, handle, ensure_ascii=True, indent=2)

    return {"applied": True, "new_version": version_tag, "path": str(version_path)}
