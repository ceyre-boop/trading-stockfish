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
        "timestamp_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat()
        + "Z",
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
