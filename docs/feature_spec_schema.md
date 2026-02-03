# Feature Spec Schema (v1.0.0)

Canonical contract for feature definitions used by registry → extractor → audit → drift → stats → policy → evaluator → runner. Versioned at **v1.0.0**; bump major on breaking changes.

## Top-level
- `version` (string, required): must be `"1.0.0"` for this schema.
- `registry_version` (string, optional): registry version tag.
- `run_id` (string, optional): provenance for generation.
- `timestamp_utc` (string, date-time, optional): ISO8601 UTC when emitted.
- `features` (object, required): map of feature name → feature spec (keys sorted for determinism).

## Feature entry (per feature)
- `name` (string, optional): feature name (defaults to the key).
- `description` (string, optional): human-readable summary.
- `source` (string, required): upstream source identifier (e.g., `market_data`, `news`, `ml`, `derived`).
- `path` (string, required): alias/path for extraction (dot/bracket notation allowed).
- `dtype` (string, required): one of `float`, `int`, `string`, `bool`, `category`, `array`, `object`.
- `role` (array[string], required): roles, e.g., `eval`, `ml`, `analytics`, `monitoring`.
- `tags` (array[string], required): ontology tags (domain, factor type, etc.).
- `required` (bool, default false): whether missing is an error at extraction.
- `live` (bool, required): whether enabled for production pipelines.
- `default` (any, optional): fallback when missing and not required.
- `constraints` (object, optional):
  - `allowed_values` (array): enumerated categories (strings/numbers).
  - `range` (object): `min` (number), `max` (number), inclusive unless otherwise noted.
  - `regex` (string): pattern for string values.
  - `dtype_strict` (bool, default true): enforce type exactly.
- `encodings` (object, optional): e.g., label maps, scaling info; key/value per encoding.
- `transform` (array[string], optional): ordered transforms applied during extraction (e.g., `clip`, `zscore`, `one_hot`).
- `dependencies` (array[string], optional): other features or raw fields required.
- `null_ok` (bool, default false): whether null/None is permitted.
- `sensitivity` (string, optional): classification (e.g., `public`, `internal`, `restricted`).
- `provenance` (object, optional): author, ticket, rationale.
- `version` (string, optional): per-feature semantic version if independently versioned.

## Validation rules
- `features` must be an object; keys are unique feature names.
- Each feature must include `source`, `path`, `dtype`, `role`, `tags`, and `live`.
- `role` and `tags` must be non-empty arrays of strings.
- `dtype` must be in the allowed set; `constraints.range` only for numeric dtypes; `constraints.allowed_values` recommended for `category`.
- If `required` is true and `default` is absent, extractor must emit `missing_alias` when not present.
- `dependencies` must be resolvable feature names or raw field identifiers.

## Allowed values and guidance
- Dtypes: `float`, `int`, `string`, `bool`, `category`, `array`, `object`.
- Roles: `eval`, `ml`, `analytics`, `monitoring`, `debug` (extendable; document additions in release notes).
- Tags: free-form but should map to ontology facets (factor class, domain, timeframe, risk type).
- Transforms: document order; transforms must be deterministic and side-effect free.
- Encodings: store label maps or scaler params if needed for reproducibility.

## Alias resolution
- `path` is resolved by the extractor against the raw state; dotted or bracket notation allowed.
- Aliases must resolve deterministically; unresolved paths trigger `missing_alias` issues in audits.

## Constraints enforcement
- `constraints.allowed_values` checked for categorical/string fields; violations emit `constraint_violation`.
- `constraints.range` checked for numeric fields; out-of-range emits `constraint_violation`.
- `dtype_strict` governs whether coercion is allowed; if true, mismatches emit `type_mismatch`.

## Dependencies
- `dependencies` declare upstream fields/features required; extractor should fetch/compute these first and order operations accordingly.

## Forward compatibility
- Additive, non-breaking fields should bump minor version; breaking changes bump major.
- Unknown fields should be ignored by tolerant consumers but flagged by strict validation in CI.

## Example (abridged)
```json
{
  "version": "1.0.0",
  "registry_version": "2026.02",
  "timestamp_utc": "2026-02-02T12:00:00Z",
  "features": {
    "macro_pressure": {
      "source": "news",
      "path": "news.macro.pressure",
      "dtype": "float",
      "role": ["eval", "ml"],
      "tags": ["macro", "risk", "news"],
      "live": true,
      "constraints": {"range": {"min": -5, "max": 5}},
      "transform": ["clip"],
      "dependencies": ["news.raw"],
      "description": "Macro pressure score from news feed"
    },
    "growth_event_flag": {
      "source": "ontology",
      "path": "macro.events.growth_flag",
      "dtype": "category",
      "role": ["eval", "ml"],
      "tags": ["macro", "ontology", "events"],
      "live": true,
      "constraints": {"allowed_values": ["GDP", "PMI", "ISM", "Retail"]},
      "null_ok": false
    }
  }
}
```

## JSON Schema
See `schemas/feature_spec.schema.json` for the machine-validatable contract.
