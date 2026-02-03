# Research Harness for Parquet Storage

## Purpose
- Stable, deterministic Python API to query long-term Parquet history (decisions, audits, stats, policies).
- Support research use cases: regime performance studies, feature drift analysis, session behavior comparisons, policy version benchmarking, and safety/safe-mode incidence tracking.
- Keep reads reproducible and schema-aligned; no mutation of source artifacts.

## Core Queries (examples)
- Per-regime performance over time: PnL, Sharpe, hit rate, drawdown slices by macro/session regime and timeframe.
- Feature importance/stability drift: track feature weights, trust, and stability metrics over rolling windows.
- Session behavior: compare sessions (e.g., LONDON vs NY vs OVERLAP) for action mix, hit rate, and drawdown distribution.
- Policy version performance: correlate decision outcomes and stats with policy_version and engine_version.
- Safety mode incidence: counts and timelines of safe-mode activations (if logged alongside decisions/audits).

## API Surface (pure functions)
- `load_decisions(filter: DecisionsFilter) -> pd.DataFrame`
- `load_stats(filter: StatsFilter) -> pd.DataFrame`
- `load_policies(filter: PolicyFilter) -> pd.DataFrame`
- `compute_regime_performance(df_decisions, by=["session_regime", "macro_regimes"], metrics=["pnl", "sharpe", "hit_rate"], freq="D") -> pd.DataFrame`
- `compute_feature_drift_over_time(df_stats, feature: str, freq="W") -> pd.DataFrame`
- `compute_policy_version_performance(df_decisions, df_stats=None, by=["policy_version"], freq="W") -> pd.DataFrame`
- `compute_safety_mode_incidence(df_decisions_or_audits, freq="D") -> pd.DataFrame` (optional, if safety events are encoded)

## Filtering Model
Filters are light dataclasses (or TypedDicts) passed to loaders; all fields optional and combined with logical AND:
- `date_range`: start/end UTC dates (inclusive)
- `symbols`: list[str]
- `session_regimes`: list[str]
- `macro_regimes`: list[str]
- `policy_versions`: list[str]
- `timeframes`: list[str]
- `run_ids`: list[str] (optional)
- `limit`: optional row cap for bounded reads in notebooks

## Determinism Requirements
- Pure functions only: loaders read Parquet; compute_* functions derive new DataFrames without side effects.
- No randomness; no sampling. Any row limiting is explicit and stable (e.g., head(n) after sort).
- No writes to storage; research harness lives read-only relative to storage/.
- Schema alignment: columns match `schemas/*_storage.schema.json`; coercions are explicit and logged if needed.
- Time handling: all timestamps treated as UTC; use ISO-8601 parsing; avoid local time assumptions.
- Engine parameters (paths, filters) passed explicitly—no implicit globals or env-dependent defaults.
