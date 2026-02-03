# Research Workflow (Daily/Weekly/Monthly)

## Daily Workflow
- Run engine and emit logs (decisions, audits, stats, policy snapshots as applicable).
- Update storage for the run: `python -m engine.jobs.update_storage_for_run --run-id <run>` (deterministic append-only Parquet).
- Lightweight research checks via `research_api`:
  - Regime performance snapshot (session + macro) on the latest day.
  - Feature drift quick check from recent stats (stability/variance deltas).
  - Safety mode events: count/inspect any SAFE_MODE activations if captured in decisions/audits.

## Weekly Workflow
- Run feedback loop to refresh policy candidates; archive outcomes.
- Update storage for the week (date-range backfill if needed): `python -m engine.jobs.backfill_storage --from YYYY-MM-DD --to YYYY-MM-DD`.
- Deeper research queries (Parquet-only):
  - Per-regime performance over the week (PnL, Sharpe, hit rate) via `compute_regime_performance`.
  - Feature drift over time using stats: `compute_feature_drift_over_time`.
  - Policy version performance: join decisions with policies and evaluate.
  - Safety mode incidence over the week (counts/timeline).
- Review candidate policy behavior with gated outcomes; keep notes in feedback summaries.

## Monthly Workflow
- Full storage backfill for any gaps (deterministic append, no mutation of prior partitions).
- Long-horizon research:
  - Session behavior comparisons (LONDON vs NY vs OVERLAP) using decisions storage.
  - Macro regime transitions and persistence from decisions/stats.
  - Long-term drift curves for key features from stats storage.
- Retrain ML priors: `python -m engine.jobs.retrain_priors --days 30` (or explicit `--from/--to` window).
- Review model metrics and promotion decisions (metadata under `models/`), and record policy evolution plus any safety events.

## Determinism
- All workflows operate on Parquet storage produced by append-only writers; no live data dependencies.
- Research queries and training use `research_api` and `ml_training` with fixed lookahead windows; no randomness, no shuffling.
- CLI jobs are parameterized by explicit dates/ranges to ensure reproducibility of backfills, queries, and retraining.
