import json
import pathlib
import pytest

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None

BASE = pathlib.Path(__file__).resolve().parent.parent


def _load_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _skip_if_no_jsonschema():
    if jsonschema is None:
        pytest.skip("jsonschema not installed")


def test_feature_spec_schema_example_validates():
    _skip_if_no_jsonschema()
    schema = _load_json(BASE / "schemas" / "feature_spec.schema.json")
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


def test_feature_audit_schema_example_validates():
    _skip_if_no_jsonschema()
    schema = _load_json(BASE / "schemas" / "feature_audit.schema.json")
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
