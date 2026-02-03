import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class PolicyConfig:
    def __init__(self, data: Dict[str, Any]):
        self.data = data or {}
        self.features = (
            self.data.get("features", {}) if isinstance(self.data, dict) else {}
        )
        self.base_weights = (
            self.data.get("base_weights", {}) if isinstance(self.data, dict) else {}
        )
        self.trust_map = (
            self.data.get("trust", {}) if isinstance(self.data, dict) else {}
        )
        self.regime_multipliers = (
            self.data.get("regime_multipliers", {})
            if isinstance(self.data, dict)
            else {}
        )

    def get_weight(self, feature: str) -> float:
        # Backward-compatible alias for base weight
        if isinstance(self.base_weights, dict) and feature in self.base_weights:
            try:
                return float(self.base_weights.get(feature, 1.0))
            except Exception:
                return 1.0
        entry = self.features.get(feature) if isinstance(self.features, dict) else None
        if isinstance(entry, dict):
            return float(entry.get("weight", 1.0))
        return 1.0

    def get_trust(self, feature: str) -> float:
        if isinstance(self.trust_map, dict) and feature in self.trust_map:
            try:
                return float(self.trust_map.get(feature, 1.0))
            except Exception:
                return 1.0
        entry = self.features.get(feature) if isinstance(self.features, dict) else None
        if isinstance(entry, dict):
            try:
                return float(entry.get("trust_score", 1.0))
            except Exception:
                return 1.0
        return 1.0

    def get_regime_multiplier(self, feature: str, regimes: Iterable[str]) -> float:
        multiplier = 1.0
        if not isinstance(self.regime_multipliers, dict):
            return multiplier
        for regime in regimes or []:
            try:
                rm = self.regime_multipliers.get(str(regime), {})
                if isinstance(rm, dict):
                    multiplier *= float(rm.get(feature, 1.0))
            except Exception:
                continue
        return multiplier

    def get_base_weight(self, feature: str) -> float:
        if isinstance(self.base_weights, dict) and feature in self.base_weights:
            try:
                return float(self.base_weights.get(feature, 1.0))
            except Exception:
                return 1.0
        return self.get_weight(feature)

    def effective_weight(self, feature: str, regimes: Iterable[str]) -> float:
        base = self.get_base_weight(feature)
        trust = self.get_trust(feature)
        regime_mult = self.get_regime_multiplier(feature, regimes)
        return base * trust * regime_mult


def load_policy(path: Path | str) -> Optional[PolicyConfig]:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    return PolicyConfig(obj)


def get_default_policy_path(run_id: Optional[str] = None) -> Path:
    """Return default policy path in logs/policy.

    Args:
        run_id: Optional identifier (e.g., timestamp or UUID) to embed in the file name.

    Returns:
        Path to logs/policy/policy_config_<run_id_or_timestamp>.json
    """

    timestamp = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return Path("logs/policy") / f"policy_config_{timestamp}.json"
