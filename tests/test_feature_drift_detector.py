import json
from pathlib import Path

import pytest

from analytics.feature_drift_detector import AuditRecord, detect_drifts, load_audits


def _record(tmp_path, name, issues, summary=None):
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "run_id": name,
                "experiment_id": "exp",
                "timestamp_utc": "2026-01-31T00:00:00Z",
                "summary": summary or {},
                "issues": issues,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_spike_detection(tmp_path):
    # Baseline low, recent spike
    p1 = _record(
        tmp_path,
        "run1.json",
        [
            {"feature": "f1", "issue": "missing_alias"},
        ],
    )
    p2 = _record(
        tmp_path,
        "run2.json",
        [
            {"feature": "f1", "issue": "missing_alias"},
        ],
    )
    p3 = _record(
        tmp_path,
        "run3.json",
        [
            {"feature": "f1", "issue": "missing_alias"},
            {"feature": "f1", "issue": "missing_alias"},
            {"feature": "f1", "issue": "missing_alias"},
            {"feature": "f1", "issue": "missing_alias"},
        ],
    )
    records = load_audits(tmp_path)
    findings = detect_drifts(
        records, window=2, spike_factor=2.0, abs_threshold=2, missing_frac_threshold=0.8
    )
    assert "f1" in findings
    assert any("spike_missing_alias" in msg for msg in findings["f1"])


def test_persistent_missingness(tmp_path):
    paths = [
        _record(tmp_path, f"run{i}.json", [{"feature": "f2", "issue": "missing_value"}])
        for i in range(4)
    ]
    records = load_audits(tmp_path)
    findings = detect_drifts(
        records,
        window=2,
        spike_factor=2.0,
        abs_threshold=2,
        missing_frac_threshold=0.75,
    )
    assert "f2" in findings
    assert any("persistent_missing" in msg for msg in findings["f2"])


def test_new_category_detection(tmp_path):
    _record(tmp_path, "run1.json", [])
    _record(
        tmp_path,
        "run2.json",
        [
            {
                "feature": "growth_event_flag",
                "issue": "constraint_violation",
                "value": "Housing Starts",
                "allowed": ["GDP", "PMI", "ISM"],
            }
        ],
    )
    records = load_audits(tmp_path)
    findings = detect_drifts(
        records, window=2, spike_factor=2.0, abs_threshold=1, missing_frac_threshold=0.5
    )
    assert "growth_event_flag" in findings
    assert any("new_category" in msg for msg in findings["growth_event_flag"])


def test_no_spike_when_baseline_high(tmp_path):
    # High baseline, small recent increase should not flag with factor 2
    _record(
        tmp_path,
        "run1.json",
        [{"feature": "f3", "issue": "constraint_violation"} for _ in range(5)],
    )
    _record(
        tmp_path,
        "run2.json",
        [{"feature": "f3", "issue": "constraint_violation"} for _ in range(5)],
    )
    _record(
        tmp_path,
        "run3.json",
        [{"feature": "f3", "issue": "constraint_violation"} for _ in range(6)],
    )
    records = load_audits(tmp_path)
    findings = detect_drifts(
        records, window=2, spike_factor=2.0, abs_threshold=1, missing_frac_threshold=0.5
    )
    assert "f3" not in findings or not any(
        "spike_constraint_violation" in msg for msg in findings.get("f3", [])
    )


def test_detect_drifts_orders_features(tmp_path):
    _record(tmp_path, "a.json", [{"feature": "b", "issue": "missing_alias"}])
    _record(tmp_path, "b.json", [{"feature": "a", "issue": "missing_alias"}])
    records = load_audits(tmp_path)
    findings = detect_drifts(
        records, window=1, spike_factor=2.0, abs_threshold=1, missing_frac_threshold=0.5
    )
    # Ensure deterministic ordering via sorted keys in findings
    assert list(findings.keys()) == sorted(findings.keys())
