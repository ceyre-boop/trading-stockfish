"""Feature statistics writer.

Reads audit artifacts with feature snapshots and emits feature_stats.json with
per-feature metrics and a numeric correlation matrix. Designed to be deterministic
and registry-driven.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.feature_registry import load_registry


def _is_number(x: Any) -> bool:
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def _flatten_numeric(values: List[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        if _is_number(v):
            out.append(float(v))
    return out


def _shannon_entropy(values: List[Any]) -> Optional[float]:
    if not values:
        return None
    counts: Dict[Any, int] = defaultdict(int)
    for v in values:
        try:
            key = (
                v
                if isinstance(v, (str, int, float, bool))
                else json.dumps(v, sort_keys=True, default=str)
            )
        except Exception:
            key = str(v)
        counts[key] += 1
    total = sum(counts.values())
    if total == 0:
        return None
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p + 1e-12, 2)
    return ent


def _mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _std(vals: List[float]) -> Optional[float]:
    if len(vals) < 2:
        return 0.0 if vals else None
    m = _mean(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return math.sqrt(var)


def _corr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    mx = _mean(x)
    my = _mean(y)
    if mx is None or my is None:
        return None
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    denom = dx * dy
    if denom == 0:
        return None
    return num / denom


def load_audits_with_snapshots(audit_dir: Path) -> List[Dict[str, Any]]:
    audits: List[Dict[str, Any]] = []
    for path in sorted(audit_dir.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "features_snapshot" not in obj:
            continue
        obj["_path"] = str(path)
        audits.append(obj)
    return audits


def collect_feature_values(audits: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    values: Dict[str, List[Any]] = defaultdict(list)
    for audit in audits:
        snap = audit.get("features_snapshot", {})
        for k, v in snap.items():
            values[k].append(v)
    return values


def compute_feature_metrics(
    values: Dict[str, List[Any]], window: int
) -> Dict[str, Dict[str, Any]]:
    metrics: Dict[str, Dict[str, Any]] = {}
    for feature in sorted(values.keys()):
        vals = values[feature]
        total = len(vals)
        missing = sum(1 for v in vals if v is None)
        present_vals = [v for v in vals if v is not None]
        nums = _flatten_numeric(present_vals)
        mean = _mean(nums)
        std = _std(nums)
        rolling_vals = nums[-window:] if window > 0 else nums
        rolling_std = _std(rolling_vals) if rolling_vals else None
        zscores = []
        if mean is not None and std not in (None, 0):
            zscores = [abs((v - mean) / std) for v in nums]
        zscore_stability = _mean(zscores) if zscores else None
        entropy = None
        if not nums and present_vals:
            entropy = _shannon_entropy(present_vals)
        metrics[feature] = {
            "count": len(present_vals),
            "missing_frac": (missing / total) if total else 0.0,
            "mean": mean,
            "std": std,
            "min": min(nums) if nums else None,
            "max": max(nums) if nums else None,
            "entropy": entropy,
            "rolling_std": rolling_std,
            "zscore_stability": zscore_stability,
            "regime": {},
            "ml": {"shap": None, "mi": None},
        }
    return metrics


def compute_correlation_matrix(
    metrics: Dict[str, Dict[str, Any]], values: Dict[str, List[Any]]
) -> Dict[str, Any]:
    numeric_features = [
        f for f in sorted(values.keys()) if any(_is_number(v) for v in values[f])
    ]
    numeric_values = {f: _flatten_numeric(values[f]) for f in numeric_features}
    # align lengths by truncating to min length per pair
    matrix: List[List[Optional[float]]] = []
    for i, fi in enumerate(numeric_features):
        row: List[Optional[float]] = []
        for j, fj in enumerate(numeric_features):
            if i == j:
                row.append(1.0)
                continue
            xi = numeric_values[fi]
            yj = numeric_values[fj]
            mlen = min(len(xi), len(yj))
            if mlen < 2:
                row.append(None)
                continue
            row.append(_corr(xi[:mlen], yj[:mlen]))
        matrix.append(row)
    return {"features": numeric_features, "matrix": matrix}


def write_feature_stats(
    audits: List[Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
    corr: Dict[str, Any],
    out_path: Path,
    registry_version: Optional[str],
    engine_version: Optional[str],
    run_id: Optional[str],
    experiment_id: Optional[str],
    params: Dict[str, Any],
) -> None:
    latest = audits[-1] if audits else {}
    out = {
        "run_id": run_id or latest.get("run_id"),
        "experiment_id": experiment_id or latest.get("experiment_id"),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "registry_version": registry_version,
        "engine_version": engine_version,
        "features": metrics,
        "correlation_matrix": corr,
        "params": params,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature stats writer")
    parser.add_argument(
        "--audit-dir",
        type=str,
        default="logs/feature_audits",
        help="Directory with audit JSON files",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="logs/feature_stats.json",
        help="Output feature stats path",
    )
    parser.add_argument(
        "--window", type=int, default=50, help="Rolling window for stability metrics"
    )
    parser.add_argument(
        "--roles",
        nargs="*",
        default=None,
        help="Optional roles to include (eval/ml/diagnostic)",
    )
    parser.add_argument(
        "--tags", nargs="*", default=None, help="Optional tags to include"
    )
    args = parser.parse_args()

    registry = load_registry()
    audits = load_audits_with_snapshots(Path(args.audit_dir))
    if not audits:
        print("No audit files with feature snapshots found; no stats written.")
        return

    values = collect_feature_values(audits)

    # Filter by role/tags if provided
    if args.roles or args.tags:
        allowed = set()
        for name, spec in registry.specs.items():
            if args.roles and not any(r in spec.role for r in args.roles):
                continue
            if args.tags and not any(t in spec.tags for t in args.tags):
                continue
            allowed.add(name)
        values = {k: v for k, v in values.items() if k in allowed}

    metrics = compute_feature_metrics(values, window=max(1, args.window))
    corr = compute_correlation_matrix(metrics, values)

    params = {
        "window": max(1, args.window),
        "roles": args.roles or [],
        "tags": args.tags or [],
        "audit_dir": args.audit_dir,
    }

    write_feature_stats(
        audits=audits,
        metrics=metrics,
        corr=corr,
        out_path=Path(args.out),
        registry_version=getattr(registry, "version", None),
        engine_version=None,
        run_id=audits[-1].get("run_id") if audits else None,
        experiment_id=audits[-1].get("experiment_id") if audits else None,
        params=params,
    )
    print(f"Feature stats written to {args.out}")


if __name__ == "__main__":
    main()
