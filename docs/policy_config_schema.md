# Policy Config Schema

A deterministic, registry-driven policy artifact (`policy_config.json`) that assigns per-feature trust and weights for downstream evaluators, including regime-conditioned multipliers.

## Top-level fields
- `run_id`: optional string
- `experiment_id`: optional string
- `timestamp_utc`: ISO8601 UTC
- `registry_version`: version from the registry
- `engine_version`: optional string
- `features`: (optional) legacy map of feature name → policy entry (sorted by feature name)
- `base_weights`: map of feature → base weight (regime-agnostic)
- `trust`: map of feature → trust score [0,1] derived from audits/stats
- `regime_multipliers`: map of regime_label → { feature → multiplier }
- `params`: configuration used to derive the policy (e.g., thresholds)

## Per-feature entry (legacy `features` map)
- `trust_score`: float in [0,1]
- `weight`: float (default mirrors trust_score in this version)
- `regime_multipliers`: object (placeholder for back-compat)
- `reasons`: array of strings explaining adjustments (e.g., "high_missing_frac", "drift_flagged", "low_variance")

## Effective weight computation (v0.3.0)
- `effective_weight(feature, regimes)` = `base_weight * trust * ∏ regime_multipliers[regime].get(feature, 1.0)`
- Regimes are derived at evaluation time (e.g., `HIGH_VOL`, `LOW_VOL`, `MACRO_ON`, `MACRO_OFF`).

## Deterministic rules (current version)
- If `missing_frac` > 0.5 → `trust_score = 0.0`, reason `high_missing_frac`
- If feature is drift-flagged in `drift_report.json` → `trust_score = 0.0`, reason `drift_flagged`
- Else if variance (std) is near-zero (<= 1e-6) → `trust_score = 0.3`, reason `low_variance`
- Else → `trust_score = 1.0`
- `weight` (legacy) = `trust_score` (subject to change in future versions)
- `base_weights` are populated from long-horizon feature importance; `trust` from stability/drift; `regime_multipliers` from regime-conditioned studies.

## Example (v0.3.0, truncated)
```
{
  "version": "0.3.0",
  "run_id": "run_20260202_A",
  "experiment_id": "exp_macro_v3",
  "timestamp_utc": "2026-02-02T12:40:00Z",
  "registry_version": "0.3.0",
  "engine_version": null,
  "base_weights": {
    "macro_pressure": 1.0,
    "macro_bias_num": 0.8,
    "rsi_14": 0.6,
    "growth_event_flag": 0.7
  },
  "trust": {
    "macro_pressure": 0.9,
    "macro_bias_num": 0.8,
    "rsi_14": 0.7,
    "growth_event_flag": 0.6
  },
  "regime_multipliers": {
    "HIGH_VOL": {
      "macro_pressure": 1.5,
      "macro_bias_num": 1.3,
      "rsi_14": 0.5
    },
    "LOW_VOL": {
      "macro_pressure": 0.7,
      "macro_bias_num": 0.8,
      "rsi_14": 1.2
    },
    "MACRO_ON": {
      "macro_pressure": 1.8,
      "growth_event_flag": 1.4
    },
    "MACRO_OFF": {
      "macro_pressure": 0.4
    }
  },
  "params": {
    "missing_frac_threshold": 0.5,
    "low_variance_epsilon": 1e-6
  }
}
```
