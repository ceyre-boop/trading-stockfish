# Feature Stats Artifact Schema

A deterministic, registry-driven artifact emitted by the stats module as `feature_stats.json`.

## Top-level fields
- `run_id`: optional string
- `experiment_id`: optional string
- `timestamp_utc`: ISO8601 UTC
- `registry_version`: version from the registry
- `engine_version`: optional string
- `features`: map of feature name → metrics (sorted by feature name)
- `correlation_matrix`: { "features": [names], "matrix": [[float]] }
- `params`: configuration used for the run (e.g., window, role/tags filters, audit_dir)

## Per-feature metrics
- `count`: number of observed (non-missing) samples
- `missing_frac`: fraction of missing samples over total
- `mean`: mean of numeric values (null if non-numeric or insufficient data)
- `std`: standard deviation (population) of numeric values (null if insufficient)
- `min`: min (numeric) or null
- `max`: max (numeric) or null
- `entropy`: Shannon entropy for categorical values (null if not applicable)
- `rolling_std`: rolling std over the provided window (null if insufficient)
- `zscore_stability`: mean absolute z-score over window (null if insufficient)
- `regime`: placeholder object for regime-specific metrics (empty map)
- `ml`: placeholder object with `shap` and `mi` fields (null by default)

## Correlation matrix
- Computed over numeric features with at least 2 non-missing samples.
- Stored as a dense matrix aligned with `correlation_matrix.features` ordering.

## Example (truncated)
```
{
  "run_id": "run_20260131_A",
  "experiment_id": "exp_macro_v2",
  "timestamp_utc": "2026-01-31T12:34:56Z",
  "registry_version": "0.2.0",
  "engine_version": null,
  "features": {
    "growth_event_flag": {
      "count": 10,
      "missing_frac": 0.1,
      "mean": 0.6,
      "std": 0.49,
      "min": 0.0,
      "max": 1.0,
      "entropy": 0.97,
      "rolling_std": 0.48,
      "zscore_stability": 0.4,
      "regime": {},
      "ml": {"shap": null, "mi": null}
    }
  },
  "correlation_matrix": {
    "features": ["growth_event_flag", "macro_pressure_score"],
    "matrix": [
      [1.0, 0.12],
      [0.12, 1.0]
    ]
  },
  "params": {
    "window": 50,
    "roles": ["eval", "ml"],
    "tags": [],
    "audit_dir": "logs/feature_audits"
  }
}
```

## Ingestion notes
- Stats harness produces this artifact; downstream consumers (policy, monitoring) can read per-feature metrics and correlations.
- Placeholders (`regime`, `ml.shap`, `ml.mi`) are reserved for future extensions; current writer fills null/empty.
- Output is deterministic (sorted feature keys and correlation feature order).
