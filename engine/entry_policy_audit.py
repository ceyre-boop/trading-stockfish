import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def write_policy_audit_record(
    action: str,
    version_tag: str,
    operator: str,
    reason: str,
    metadata: Dict[str, Any],
    storage_dir: str,
):
    storage_path = Path(storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)
    audit_path = storage_path / "brain_policy_audit.jsonl"

    previous_active = None
    active_pointer = storage_path / "brain_policy_entries.active.json"
    if active_pointer.exists():
        try:
            payload = json.loads(active_pointer.read_text(encoding="utf-8"))
            previous_active = payload.get("active_version")
        except Exception:
            previous_active = None

    record = {
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "action": action,
        "version_tag": version_tag,
        "operator": operator,
        "reason": reason,
        "metadata": metadata or {},
        "previous_active_version": previous_active,
        "new_active_version": (
            metadata.get("new_active_version") if isinstance(metadata, dict) else None
        ),
    }

    ordered_keys = [
        "timestamp_utc",
        "action",
        "version_tag",
        "operator",
        "reason",
        "metadata",
        "previous_active_version",
        "new_active_version",
    ]
    ordered_record = {k: record.get(k) for k in ordered_keys}

    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(ordered_record, ensure_ascii=True) + "\n")
