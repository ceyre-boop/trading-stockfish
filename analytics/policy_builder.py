"""Policy builder.

Derives per-feature trust scores and weights from registry, feature_stats.json,
and drift_report.json. Outputs policy_config.json deterministically.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.feature_registry import load_registry
from engine.regime_multipliers import (
    MultiplierConfig,
    StatsResult,
    compute_regime_multipliers,
)

MISSING_FRAC_THRESHOLD = 0.5
LOW_VARIANCE_EPSILON = 1e-6


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_policy(
    registry_path: Path,
    stats_path: Path,
    drift_path: Path,
    out_path: Path,
    run_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    missing_frac_threshold: float = MISSING_FRAC_THRESHOLD,
    low_variance_epsilon: float = LOW_VARIANCE_EPSILON,
) -> Dict[str, Any]:
    registry = load_registry(registry_path)
    stats = load_json(stats_path) or {}
    drift = load_json(drift_path) or {"findings": {}}

    features_policy: Dict[str, Dict[str, Any]] = {}
    base_weights: Dict[str, float] = {}
    trust_map: Dict[str, float] = {}
    regime_multipliers: Dict[str, Dict[str, float]] = {}

    drift_flags = {k: v for k, v in (drift.get("findings") or {}).items()}

    stats_features = (
        stats.get("features", {}) if isinstance(stats.get("features", {}), dict) else {}
    )

    for name in sorted(registry.specs.keys()):
        spec = registry.specs[name]
        reasons: List[str] = []
        trust = 1.0
        weight = 1.0
        m = stats_features.get(name, {})
        missing_frac = m.get("missing_frac") if isinstance(m, dict) else None
        std = m.get("std") if isinstance(m, dict) else None

        if missing_frac is not None and missing_frac > missing_frac_threshold:
            trust = 0.0
            reasons.append("high_missing_frac")

        if name in drift_flags and drift_flags[name]:
            trust = 0.0
            reasons.append("drift_flagged")

        if trust > 0.0 and std is not None and std <= low_variance_epsilon:
            trust = 0.3
            reasons.append("low_variance")

        weight = trust

        # Populate v0.3.0 schema fields
        base_weights[name] = weight
        trust_map[name] = trust

        # Legacy per-feature block retained for backward compatibility and reasons
        features_policy[name] = {
            "trust_score": trust,
            "weight": weight,
            "regime_multipliers": {},
            "reasons": reasons,
            "role": spec.role,
            "tags": spec.tags,
            "live": spec.live,
        }

    policy = {
        "version": "0.3.0",
        "run_id": run_id or stats.get("run_id"),
        "experiment_id": experiment_id or stats.get("experiment_id"),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "registry_version": getattr(registry, "version", None),
        "engine_version": stats.get("engine_version"),
        "base_weights": base_weights,
        "trust": trust_map,
        "regime_multipliers": regime_multipliers,
        "features": features_policy,
        "params": {
            "missing_frac_threshold": missing_frac_threshold,
            "low_variance_epsilon": low_variance_epsilon,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")
    return policy


def _extract_feature_regime_multipliers(
    regime_multipliers: Dict[str, Dict[str, float]], feature: str
) -> Dict[str, float]:
    feature_mults: Dict[str, float] = {}
    for regime, fmap in (regime_multipliers or {}).items():
        if not isinstance(fmap, dict):
            continue
        if feature in fmap:
            try:
                feature_mults[regime] = float(fmap.get(feature, 1.0))
            except Exception:
                feature_mults[regime] = 1.0
    return feature_mults


def _clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def build_policy_from_stats(
    stats: StatsResult,
    config: Optional[Dict[str, Any]] = None,
    multiplier_config: Optional[MultiplierConfig] = None,
) -> Dict[str, Any]:
    """Build a policy_config dictionary from StatsResult.

    Args:
        stats: StatsResult containing feature importance, stability, and regime metrics.
        config: Optional overrides (run_id, experiment_id, registry_path, out_path).
        multiplier_config: Optional MultiplierConfig to bound multipliers.

    Returns:
        Dict suitable for serialization to policy_config.json.
    """

    cfg = config or {}
    registry_path = cfg.get("registry_path", "config/feature_registry.json")
    registry = None
    try:
        registry = load_registry(registry_path)
    except Exception:
        registry = None

    run_id = cfg.get("run_id") or (stats.metadata or {}).get("run_id")
    experiment_id = cfg.get("experiment_id") or (stats.metadata or {}).get(
        "experiment_id"
    )
    engine_version = (stats.metadata or {}).get("engine_version")
    registry_version = getattr(registry, "version", None)

    # Base weights from long-horizon importance (normalized for stability)
    raw_importance: Dict[str, float] = {}
    for feat, weight in (stats.feature_importance or {}).items():
        try:
            raw_importance[str(feat)] = float(weight)
        except Exception:
            continue

    base_weights: Dict[str, float] = {}
    if raw_importance:
        max_abs = max(abs(v) for v in raw_importance.values()) or 1.0
        for feat, val in raw_importance.items():
            base_weights[feat] = val / max_abs

    # Trust from stability metrics (bounded)
    trust_map: Dict[str, float] = {}
    for feat, stability in (stats.feature_stability or {}).items():
        try:
            trust_map[str(feat)] = _clamp(float(stability), 0.0, 1.0)
        except Exception:
            continue

    mcfg = multiplier_config
    if mcfg is None:
        candidate_cfg = cfg.get("multiplier_config")
        if isinstance(candidate_cfg, MultiplierConfig):
            mcfg = candidate_cfg
        elif isinstance(candidate_cfg, dict):
            try:
                mcfg = MultiplierConfig(**candidate_cfg)
            except Exception:
                mcfg = None

    regime_multipliers = compute_regime_multipliers(stats, mcfg)

    # Build per-feature block for compatibility
    features_policy: Dict[str, Dict[str, Any]] = {}
    registry_features = set(registry.specs.keys()) if registry else set()
    feature_names = sorted(
        set(base_weights.keys()) | set(trust_map.keys()) | registry_features
    )
    for name in feature_names:
        feature_regime_mult = _extract_feature_regime_multipliers(
            regime_multipliers, name
        )
        bw = base_weights.get(name, 1.0)
        trust_val = trust_map.get(name, 1.0)
        features_policy[name] = {
            "trust_score": trust_val,
            "weight": bw,
            "regime_multipliers": feature_regime_mult,
            "reasons": [],
            "role": (
                getattr(registry.specs.get(name), "role", [])
                if registry and name in registry.specs
                else []
            ),
            "tags": (
                getattr(registry.specs.get(name), "tags", [])
                if registry and name in registry.specs
                else []
            ),
            "live": (
                getattr(registry.specs.get(name), "live", True)
                if registry and name in registry.specs
                else True
            ),
        }

    policy = {
        "version": "0.3.0",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "registry_version": registry_version,
        "engine_version": engine_version,
        "base_weights": base_weights,
        "trust": trust_map,
        "regime_multipliers": regime_multipliers,
        "features": features_policy,
        "params": {
            "source": "stats_feedback_loop",
            "multiplier_bounds": {
                "min": (mcfg or MultiplierConfig()).min_multiplier,
                "max": (mcfg or MultiplierConfig()).max_multiplier,
            },
        },
    }

    out_path = cfg.get("out_path")
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8"
        )

    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy builder")
    parser.add_argument("--registry", type=str, default="config/feature_registry.json")
    parser.add_argument("--stats", type=str, default="logs/feature_stats.json")
    parser.add_argument(
        "--drift", type=str, default="logs/drift_reports/drift_report.json"
    )
    parser.add_argument("--out", type=str, default="logs/policy_config.json")
    parser.add_argument(
        "--missing-frac-threshold", type=float, default=MISSING_FRAC_THRESHOLD
    )
    parser.add_argument(
        "--low-variance-epsilon", type=float, default=LOW_VARIANCE_EPSILON
    )
    args = parser.parse_args()

    policy = build_policy(
        registry_path=Path(args.registry),
        stats_path=Path(args.stats),
        drift_path=Path(args.drift),
        out_path=Path(args.out),
        missing_frac_threshold=args.missing_frac_threshold,
        low_variance_epsilon=args.low_variance_epsilon,
    )
    print(
        f"Policy config written to {args.out} with {len(policy.get('features', {}))} features"
    )


if __name__ == "__main__":
    main()
