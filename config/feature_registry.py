import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    type: str
    source: str
    description: str
    live: bool
    shape: str
    role: List[str]
    transform: Dict[str, Any]
    encoding: Dict[str, Any]
    dependencies: List[str]
    tags: List[str]
    alias: str
    constraints: Dict[str, Any]


class FeatureRegistry:
    def __init__(
        self, specs: Dict[str, FeatureSpec], version: str, defaults: Dict[str, Any]
    ):
        self.specs = specs
        self.version = version
        self.defaults = defaults

    @classmethod
    def load(cls, path: str | Path) -> "FeatureRegistry":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        version = obj.get("version", "0.0.0")
        defaults = obj.get("defaults", {})
        specs: Dict[str, FeatureSpec] = {}
        for name, meta in obj.get("features", {}).items():
            merged = {}
            merged.update(defaults)
            merged.update(meta)
            specs[name] = FeatureSpec(
                name=name,
                type=merged["type"],
                source=merged["source"],
                description=merged.get("description", ""),
                live=bool(merged.get("live", True)),
                shape=merged.get("shape", "scalar"),
                role=list(merged.get("role", [])),
                transform=merged.get("transform", {"kind": "none", "params": {}}),
                encoding=merged.get("encoding", {"kind": "none", "params": {}}),
                dependencies=list(merged.get("dependencies", [])),
                tags=list(merged.get("tags", [])),
                alias=str(merged.get("alias", name)),
                constraints=merged.get("constraints", {}),
            )
        return cls(specs=specs, version=version, defaults=defaults)

    def get(self, name: str) -> FeatureSpec:
        return self.specs[name]

    def list_by_tag(self, tag: str) -> List[FeatureSpec]:
        return [spec for spec in self.specs.values() if tag in spec.tags]

    def list_for_role(self, role: str) -> List[FeatureSpec]:
        return [spec for spec in self.specs.values() if role in spec.role]


def load_registry(
    default_path: str | Path = Path("config/feature_registry.json"),
) -> FeatureRegistry:
    return FeatureRegistry.load(default_path)
