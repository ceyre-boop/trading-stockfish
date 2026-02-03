"""Feature drift detector.

Reads feature audit artifacts and flags drift conditions using simple, deterministic
rules (spikes, new categories, persistent missingness).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AuditRecord:
    path: Path
    run_id: Optional[str]
    experiment_id: Optional[str]
    timestamp_utc: Optional[str]
    summary: Dict[str, Any]
    issues: List[Dict[str, Any]]


def load_audits(audit_dir: Path) -> List[AuditRecord]:
    records: List[AuditRecord] = []
    for path in sorted(audit_dir.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        records.append(
            AuditRecord(
                path=path,
                run_id=obj.get("run_id"),
                experiment_id=obj.get("experiment_id"),
                timestamp_utc=obj.get("timestamp_utc"),
                summary=obj.get("summary", {}),
                issues=obj.get("issues", []),
            )
        )
    return records


def _series_from_issues(records: List[AuditRecord]) -> Dict[str, Dict[str, List[int]]]:
    """Build per-feature time-series of issue counts ordered by audit file name."""
    series: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        per_feature_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for issue in rec.issues:
            feature = str(issue.get("feature"))
            kind = str(issue.get("issue"))
            per_feature_counts[feature][kind] += 1
        # push counts for every feature seen so far; if absent in this run, push 0
        all_features = set(series.keys()) | set(per_feature_counts.keys())
        for feat in sorted(all_features):
            if feat not in per_feature_counts:
                for kind in series[feat].keys():
                    series[feat][kind].append(0)
                continue
            for kind, count in per_feature_counts[feat].items():
                series[feat][kind].append(count)
            # ensure kinds seen historically but not this run get 0
            for kind in series[feat].keys():
                if kind not in per_feature_counts[feat]:
                    series[feat][kind].append(0)
    return series


def detect_drifts(
    records: List[AuditRecord],
    window: int,
    spike_factor: float,
    abs_threshold: int,
    missing_frac_threshold: float,
) -> Dict[str, List[str]]:
    findings: Dict[str, List[str]] = defaultdict(list)
    series = _series_from_issues(records)

    def _spike(vals: List[int]) -> bool:
        if not vals:
            return False
        recent = vals[-1]
        baseline = vals[-window - 1 : -1] if len(vals) > 1 else []
        if not baseline:
            return False
        baseline_max = max(baseline)
        return recent >= abs_threshold and recent > baseline_max * spike_factor

    # New categories and missing frac are derived directly from issues arrays.
    # Build last-run mappings for category violations and missingness.
    last_rec = records[-1] if records else None
    last_issues = last_rec.issues if last_rec else []
    category_violations: Dict[str, List[str]] = defaultdict(list)
    missing_hits: Dict[str, int] = defaultdict(int)

    for issue in last_issues:
        feature = str(issue.get("feature"))
        kind = str(issue.get("issue"))
        if kind == "constraint_violation":
            val = issue.get("value")
            allowed = issue.get("allowed") or []
            if allowed and val is not None:
                sval = str(val)
                if sval not in {str(a) for a in allowed}:
                    category_violations[feature].append(sval)
        if kind == "missing_value":
            missing_hits[feature] += 1

    for feature, kinds in series.items():
        # Spike checks
        for kind in ("missing_alias", "constraint_violation", "type_mismatch"):
            vals = kinds.get(kind, [])
            if _spike(vals):
                findings[feature].append(
                    f"spike_{kind}: prev_max={max(vals[:-1])} recent={vals[-1]}"
                )

        # Persistent missingness / collapse
        miss_vals = kinds.get("missing_value", [])
        if miss_vals:
            total_runs = len(miss_vals)
            missing_runs = sum(1 for v in miss_vals if v > 0)
            if total_runs > 0 and missing_runs / total_runs >= missing_frac_threshold:
                findings[feature].append(
                    f"persistent_missing: {missing_runs}/{total_runs} runs >= threshold {missing_frac_threshold}"
                )

    # New categories in last run
    for feature, vals in category_violations.items():
        joined = ", ".join(sorted(set(vals)))
        findings[feature].append(f"new_category: {joined}")

    return findings


def write_report(
    findings: Dict[str, List[str]],
    latest: Optional[AuditRecord],
    out_path: Path,
    params: Dict[str, Any],
) -> None:
    out = {
        "run_id": latest.run_id if latest else None,
        "experiment_id": latest.experiment_id if latest else None,
        "timestamp_utc": latest.timestamp_utc if latest else None,
        "findings": {k: findings[k] for k in sorted(findings)},
        "aggregates": {
            "features_flagged": len(findings),
            "messages": sum(len(v) for v in findings.values()),
        },
        "params": params,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature drift detector")
    parser.add_argument(
        "--audit-dir",
        type=str,
        default="logs/feature_audits",
        help="Directory with audit JSON files",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="logs/drift_reports/drift_report.json",
        help="Output report path",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Baseline window size (number of prior runs)",
    )
    parser.add_argument(
        "--spike-factor", type=float, default=3.0, help="Spike multiplier threshold"
    )
    parser.add_argument(
        "--abs-threshold",
        type=int,
        default=3,
        help="Minimum absolute count to flag a spike",
    )
    parser.add_argument(
        "--missing-frac-threshold",
        type=float,
        default=0.8,
        help="Fraction of runs missing to flag collapse",
    )
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir)
    out_path = Path(args.out)

    records = load_audits(audit_dir)
    if not records:
        print("No audit files found; no report written.")
        return

    findings = detect_drifts(
        records=records,
        window=max(1, args.window),
        spike_factor=max(1.0, args.spike_factor),
        abs_threshold=max(1, args.abs_threshold),
        missing_frac_threshold=max(0.0, min(1.0, args.missing_frac_threshold)),
    )

    params = {
        "window": max(1, args.window),
        "spike_factor": max(1.0, args.spike_factor),
        "abs_threshold": max(1, args.abs_threshold),
        "missing_frac_threshold": max(0.0, min(1.0, args.missing_frac_threshold)),
        "audit_dir": str(audit_dir),
    }

    write_report(findings, records[-1], out_path, params)
    print(f"Drift report written to {out_path}")


if __name__ == "__main__":
    main()
