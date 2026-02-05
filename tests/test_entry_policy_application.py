import json
import os
from pathlib import Path

import pandas as pd

from engine.entry_policy_apply import apply_entry_policy_update
from engine.entry_policy_audit import write_policy_audit_record
from engine.entry_policy_versions import list_policy_versions, rollback_policy_version
from engine.evaluator import (
    _ENTRY_BRAIN_POLICY_CACHE,
    _ENTRY_BRAIN_POLICY_CACHE_KEY,
    _load_entry_brain_policy,
)


def _policy_df(label: str):
    return pd.DataFrame(
        [
            {
                "entry_model_id": "ENTRY_OB_CONTINUATION",
                "market_profile_state": "ACCUMULATION",
                "session_profile": "PROFILE_1A",
                "liquidity_bias_side": "UP",
                "label": label,
            }
        ]
    )


def test_apply_policy_when_approved(tmp_path):
    storage = tmp_path / "policies" / "brain"
    storage.mkdir(parents=True)

    proposed = _policy_df("PREFERRED")
    current = _policy_df("ALLOWED")

    result = apply_entry_policy_update(
        proposed_policy=proposed,
        current_policy=current,
        approval=True,
        version_tag="v1",
        storage_dir=str(storage),
    )

    version_file = storage / "brain_policy_entries.v1.json"
    active_pointer = storage / "brain_policy_entries.active.json"
    assert result["applied"] is True
    assert version_file.exists()
    assert active_pointer.exists()
    pointer = json.loads(active_pointer.read_text(encoding="utf-8"))
    assert pointer["active_version"] == "v1"
    assert Path(pointer["path"]).name == version_file.name


def test_apply_policy_when_not_approved(tmp_path):
    storage = tmp_path / "policies" / "brain"
    storage.mkdir(parents=True)
    # Seed active pointer
    active_pointer = storage / "brain_policy_entries.active.json"
    active_pointer.write_text(
        json.dumps({"active_version": "old", "path": "old_path"}), encoding="utf-8"
    )

    proposed = _policy_df("PREFERRED")
    current = _policy_df("ALLOWED")

    result = apply_entry_policy_update(
        proposed_policy=proposed,
        current_policy=current,
        approval=False,
        version_tag="v2",
        storage_dir=str(storage),
    )

    pointer_after = json.loads(active_pointer.read_text(encoding="utf-8"))
    assert result["applied"] is False
    assert pointer_after["active_version"] == "old"


def test_policy_version_listing(tmp_path):
    storage = tmp_path / "policies" / "brain"
    storage.mkdir(parents=True)
    for tag in ["v1", "v3", "v2"]:
        (storage / f"brain_policy_entries.{tag}.json").write_text(
            "{}", encoding="utf-8"
        )
    (storage / "brain_policy_entries.active.json").write_text("{}", encoding="utf-8")

    versions = list_policy_versions(str(storage))
    assert versions == ["v1", "v2", "v3"]


def test_policy_rollback(tmp_path):
    storage = tmp_path / "policies" / "brain"
    storage.mkdir(parents=True)
    version_file = storage / "brain_policy_entries.v1.json"
    version_file.write_text("{}", encoding="utf-8")

    result = rollback_policy_version("v1", str(storage))
    pointer = json.loads(
        (storage / "brain_policy_entries.active.json").read_text(encoding="utf-8")
    )

    assert result["rolled_back"] is True
    assert pointer["active_version"] == "v1"
    assert Path(pointer["path"]).name == version_file.name


def test_policy_audit_record_written(tmp_path):
    storage = tmp_path / "policies" / "brain"
    storage.mkdir(parents=True)
    audit_path = storage / "brain_policy_audit.jsonl"

    write_policy_audit_record(
        action="apply",
        version_tag="v1",
        operator="tester",
        reason="unit test",
        metadata={"new_active_version": "v1"},
        storage_dir=str(storage),
    )

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["action"] == "apply"
    assert payload["version_tag"] == "v1"
    assert payload["operator"] == "tester"
    assert payload["new_active_version"] == "v1"


def test_evaluator_loads_active_policy(tmp_path, monkeypatch):
    storage = tmp_path / "policies" / "brain"
    storage.mkdir(parents=True)
    version_file = storage / "brain_policy_entries.v1.json"
    policy_payload = {"policy": _policy_df("PREFERRED").to_dict(orient="records")}
    version_file.write_text(json.dumps(policy_payload), encoding="utf-8")

    active_pointer = storage / "brain_policy_entries.active.json"
    active_pointer.write_text(
        json.dumps({"active_version": "v1", "path": str(version_file)}),
        encoding="utf-8",
    )

    monkeypatch.setattr("engine.evaluator._ENTRY_BRAIN_POLICY_PATH", active_pointer)
    monkeypatch.setattr(
        "engine.evaluator._ENTRY_BRAIN_POLICY_CACHE", None, raising=False
    )
    monkeypatch.setattr(
        "engine.evaluator._ENTRY_BRAIN_POLICY_CACHE_KEY", None, raising=False
    )

    df = _load_entry_brain_policy()
    assert not df.empty
    assert df.iloc[0]["label"] == "PREFERRED"
