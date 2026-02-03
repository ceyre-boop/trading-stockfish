# Phase 9 Completion Report

## Storage Layer
- `docs/storage_layout.md` present and describes Parquet layout/partitioning.
- Schemas present for all tables: `schemas/decisions_storage.schema.json`, `schemas/audits_storage.schema.json`, `schemas/stats_storage.schema.json`, `schemas/policies_storage.schema.json`.
- `engine/storage_writers.py` implemented for deterministic, de-duped Parquet writes.
- Storage jobs implemented: `engine/jobs/storage_jobs.py` with backfill/update CLIs (`backfill_storage.py`, `update_storage_for_run.py`).
- Sample Parquet production verified via new tests (`test_storage_writers.py`) and passing suite.

## Research API
- `docs/research_harness.md` present (API surface, queries, determinism).
- `engine/research_api.py` implemented for loading/filtering decisions/stats/policies and computing regime/performance/drift metrics.
- Tests added and passing: `tests/test_research_api.py` (filters, regime performance, policy version performance).

## ML Priors
- `docs/ml_priors.md` present with targets, labels, features, determinism.
- `engine/ml_training.py` implemented for label generation, feature assembly, deterministic training, and artifact writing.
- `engine/jobs/retrain_priors.py` implemented for periodic retrain/promotion with deterministic comparisons.
- Model artifacts versioned under `models/<model>/<version_tag>/` with metadata including promotion flag and metrics.
- Promotion logic deterministic (accuracy primary, Brier tie-break) and metadata updates.

## Research Workflows
- `docs/research_workflow.md` documents daily/weekly/monthly cadence using storage jobs, research API, and retrain job.

## Tests
- Full suite passing: `pytest -q` → all tests green (includes new storage writer and research API tests).
- No regressions observed during Phase 9 additions.
