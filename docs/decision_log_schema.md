# Decision Log Schema (v1.0.0)

Newline-delimited JSON (JSONL); each line is one decision record.

## Required fields
- `run_id` (string): run identifier (UUID or run hash).
- `decision_id` (string): unique decision identifier (UUID or monotonic counter).
- `timestamp_utc` (string, ISO 8601, suffixed with `Z`): decision timestamp.
- `symbol` (string): instrument symbol.
- `timeframe` (string): bar timeframe, e.g., `M5`, `H1`.
- `session_regime` (string): session label (e.g., `ASIA`, `LONDON`, `NY`, `LONDON_NY_OVERLAP`, `PRE_SESSION`, `POST_SESSION`).
- `macro_regimes` (array[string]): macro/regime flags (e.g., `MACRO_ON`, `HIGH_VOL`).
- `feature_vector` (object): canonical feature values (post-transform/encoding), keyed by feature name.
- `effective_weights` (object): effective per-feature weights (base × trust × regime/session multipliers), keyed by feature name.
- `policy_components` (object):
  - `base_weights` (object): per-feature base weights.
  - `trust` (object): per-feature trust scores.
  - `regime_multipliers` (object): map of regime → {feature → multiplier}.
- `evaluation_score` (number): overall score used to choose the action.
- `action` (string): chosen action (e.g., `LONG`, `SHORT`, `FLAT`).
- `provenance` (object):
  - `policy_version` (string)
  - `feature_spec_version` (string)
  - `feature_audit_version` (string)
  - `engine_version` (string)

## Optional fields
- `position_size` (number): position size at decision time.
- `outcome` (object|null): may be null at decision time; can be filled later.
  - `pnl` (number|null)
  - `max_drawdown` (number|null)
  - `holding_period_bars` (integer|null)

## Example (single JSON object / line)
```json
{
  "run_id": "run_20260202_A",
  "decision_id": "dec_00001",
  "timestamp_utc": "2026-02-02T12:00:00Z",
  "symbol": "ES",
  "timeframe": "M5",
  "session_regime": "LONDON_NY_OVERLAP",
  "macro_regimes": ["MACRO_ON", "HIGH_VOL"],
  "feature_vector": {
    "session_high": 5025.5,
    "session_low": 4988.0,
    "session_range": 37.5,
    "macro_pressure": 0.4
  },
  "effective_weights": {
    "session_high": 0.9,
    "session_low": 1.0,
    "macro_pressure": 1.2
  },
  "policy_components": {
    "base_weights": {"session_high": 1.0, "session_low": 1.0, "macro_pressure": 1.0},
    "trust": {"session_high": 0.9, "session_low": 1.0, "macro_pressure": 1.0},
    "regime_multipliers": {"MACRO_ON": {"macro_pressure": 1.2}}
  },
  "evaluation_score": 0.37,
  "action": "LONG",
  "position_size": 0.5,
  "outcome": {
    "pnl": null,
    "max_drawdown": null,
    "holding_period_bars": null
  },
  "provenance": {
    "policy_version": "0.3.0",
    "feature_spec_version": "1.0.0",
    "feature_audit_version": "1.0.0",
    "engine_version": "v4.0"
  }
}
```