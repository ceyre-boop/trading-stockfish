# Feature Audit Schema (v1.0.0)

Canonical schema for per-run feature audits emitted by the extractor. Versioned at **v1.0.0**; bump major on breaking changes. Downstream consumers: drift detector, stats harness, policy builder (via stats), ML pipelines.

## Top-level
- `version` (string, required): must be `"1.0.0"` for this schema.
- `run_id` (string, optional): audit run identifier.
- `experiment_id` (string, optional): experiment/provenance tag.
- `timestamp_utc` (string, date-time, required): ISO8601 UTC when the audit was produced.
- `registry_version` (string, optional): registry version used.
- `engine_version` (string, optional): engine build/revision.
- `summary` (object, optional): counts and quick stats.
  - `features_total` (integer)
  - `issues_total` (integer)
  - `issues_by_type` (object of issue_type → integer)
- `issues` (array, required): ordered list of issue records (deterministic ordering recommended).
- `snapshot` (object, optional): lightweight per-feature snapshot or reduced state.
- `params` (object, optional): parameters used by the auditor (thresholds, window sizes, etc.).

## Issue entry
- `feature` (string, required): feature name.
- `issue` (string, required): one of `missing_alias`, `constraint_violation`, `type_mismatch`, `missing_value` (extensible with versioned release notes).
- `message` (string, optional): human-readable detail.
- `path` (string, optional): alias/path checked.
- `value` (any, optional): offending value.
- `allowed` (array, optional): allowed values when `constraint_violation` is raised.
- `expected_dtype` (string, optional): expected dtype.
- `observed_dtype` (string, optional): observed dtype.
- `stats` (object, optional): small stats blob (e.g., missing fraction) if available at audit time.

## Summary semantics
- `features_total`: number of features audited.
- `issues_total`: total issues across all features.
- `issues_by_type`: counts by issue type for dashboards/CI.

## Snapshot (optional)
- Intended for lightweight, reproducible slices (e.g., recent values, categorical histograms).
- Keep payload small; include only what downstream drift/stats need.

## Consumption contracts
- Drift detector: consumes `issues` to build time-series per feature/issue type (spikes, missingness, new categories via constraint violations).
- Stats harness: consumes `issues` and optional `snapshot` to compute missing fractions, std/variance, and categorical distributions.
- Policy builder: consumes stats outputs derived from audits/drift; stable audit schema ensures drift detector consistency.
- ML pipelines: may consume `snapshot` for data quality gating.

## Example (abridged)
```json
{
  "version": "1.0.0",
  "run_id": "run_20260202_A",
  "experiment_id": "exp_macro_v3",
  "timestamp_utc": "2026-02-02T12:30:00Z",
  "registry_version": "2026.02",
  "engine_version": "1.3.0",
  "summary": {
    "features_total": 6,
    "issues_total": 4,
    "issues_by_type": {
      "missing_alias": 1,
      "constraint_violation": 2,
      "type_mismatch": 1
    }
  },
  "issues": [
    {
      "feature": "growth_event_flag",
      "issue": "constraint_violation",
      "value": "Housing Starts",
      "allowed": ["GDP", "PMI", "ISM", "Retail"],
      "message": "Unexpected category",
      "path": "macro.events.growth_flag"
    },
    {
      "feature": "macro_pressure",
      "issue": "missing_value",
      "message": "Missing in current run"
    }
  ],
  "params": {
    "missing_frac_threshold": 0.5,
    "abs_threshold": 3,
    "spike_factor": 3.0
  }
}
```

## JSON Schema
See `schemas/feature_audit.schema.json` (JSON Schema 2020-12) for the machine-validatable contract.
