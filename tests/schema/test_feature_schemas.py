import json
import pathlib

import pytest

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "schemas"


def _load_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _skip_if_no_jsonschema():
    if jsonschema is None:
        pytest.skip("jsonschema not installed")


def test_feature_spec_example_validates():
    _skip_if_no_jsonschema()
    schema = _load_json(SCHEMAS / "feature_spec.schema.json")
    example = {
        "version": "1.0.0",
        "registry_version": "2026.02",
        "features": {
            "macro_pressure": {
                "source": "news",
                "path": "news.macro.pressure",
                "dtype": "float",
                "role": ["eval", "ml"],
                "tags": ["macro", "risk"],
                "live": True,
                "constraints": {"range": {"min": -5, "max": 5}},
                "session_modifiers": {"risk_scale": 1.0, "liquidity_scale": 1.0},
            },
            "growth_event_flag": {
                "source": "ontology",
                "path": "macro.events.growth_flag",
                "dtype": "category",
                "role": ["eval", "ml"],
                "tags": ["macro", "ontology"],
                "live": True,
                "constraints": {"allowed_values": ["GDP", "PMI", "ISM", "Retail"]},
            },
        },
    }
    jsonschema.validate(instance=example, schema=schema)


def test_feature_audit_example_validates():
    _skip_if_no_jsonschema()
    schema = _load_json(SCHEMAS / "feature_audit.schema.json")
    example = {
        "version": "1.0.0",
        "run_id": "run_20260202_A",
        "timestamp_utc": "2026-02-02T12:30:00Z",
        "issues": [
            {
                "feature": "growth_event_flag",
                "issue": "constraint_violation",
                "value": "Housing Starts",
                "allowed": ["GDP", "PMI", "ISM", "Retail"],
                "message": "Unexpected category",
                "path": "macro.events.growth_flag",
            }
        ],
        "summary": {
            "features_total": 6,
            "issues_total": 1,
            "issues_by_type": {"constraint_violation": 1},
        },
    }
    jsonschema.validate(instance=example, schema=schema)


def test_round_trip_write_and_validate(tmp_path: pathlib.Path):
    _skip_if_no_jsonschema()
    spec_schema = _load_json(SCHEMAS / "feature_spec.schema.json")
    audit_schema = _load_json(SCHEMAS / "feature_audit.schema.json")

    feature_spec = {
        "version": "1.0.0",
        "registry_version": "2026.02",
        "features": {
            "session_bias": {
                "source": "derived",
                "path": "session.bias.score",
                "dtype": "float",
                "role": ["eval"],
                "tags": ["session", "risk"],
                "live": True,
                "session_modifiers": {"risk_scale": 1.1, "trade_freq_scale": 1.2},
            }
        },
    }
    feature_audit = {
        "version": "1.0.0",
        "run_id": "run_test",
        "timestamp_utc": "2026-02-02T00:00:00Z",
        "issues": [],
        "summary": {"features_total": 1, "issues_total": 0, "issues_by_type": {}},
        "snapshot": {"session_bias": {"mean": 0.0, "std": 0.0}},
    }

    spec_path = tmp_path / "feature_spec.json"
    audit_path = tmp_path / "feature_audit.json"
    spec_path.write_text(json.dumps(feature_spec, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(feature_audit, indent=2), encoding="utf-8")

    jsonschema.validate(instance=_load_json(spec_path), schema=spec_schema)
    jsonschema.validate(instance=_load_json(audit_path), schema=audit_schema)
