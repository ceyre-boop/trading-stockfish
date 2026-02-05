import json
import os
from pathlib import Path
from typing import Dict, List


def list_policy_versions(storage_dir: str) -> List[str]:
    storage_path = Path(storage_dir)
    if not storage_path.exists():
        return []
    versions: List[str] = []
    for path in storage_path.iterdir():
        name = path.name
        if not name.startswith("brain_policy_entries."):
            continue
        if (
            name.endswith("active.json")
            or name.endswith("diff.json")
            or name.endswith("proposed.json")
            or name.endswith("report.json")
        ):
            continue
        if path.suffix.lower() != ".json":
            continue
        parts = name.split(".")
        if len(parts) < 3:
            continue
        versions.append(parts[-2])
    return sorted(versions)


def rollback_policy_version(version_tag: str, storage_dir: str) -> Dict[str, str]:
    storage_path = Path(storage_dir)
    version_file = storage_path / f"brain_policy_entries.{version_tag}.json"
    if not version_file.exists():
        raise FileNotFoundError(f"Version {version_tag} not found at {version_file}")

    active_pointer = storage_path / "brain_policy_entries.active.json"
    pointer_payload = {"active_version": version_tag, "path": str(version_file)}
    storage_path.mkdir(parents=True, exist_ok=True)
    with active_pointer.open("w", encoding="utf-8") as handle:
        json.dump(pointer_payload, handle, ensure_ascii=True, indent=2)

    return {"rolled_back": True, "active_version": version_tag}
