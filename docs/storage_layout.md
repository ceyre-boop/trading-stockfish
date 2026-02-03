# Long-Term Storage Layout (Parquet)

Primary goal: deterministic, queryable history for decisions, audits, stats, and policies using Parquet. All writes respect schemas; no ad-hoc CSVs.

## Format
- Primary format: Parquet (columnar, compressed, schema-aware).
- All ingests/exports flow through validated schemas; no free-form CSVs.

## Directory Structure
```
storage/
  decisions/        # partitioned by date (primary), optional symbol/regime/timeframe
    decisions_YYYYMMDD.parquet
  audits/           # partitioned by date (primary)
    audits_YYYYMMDD.parquet
  stats/            # partitioned by date (primary)
    stats_YYYYMMDD.parquet
  policies/         # versioned rows in a single table
    policies.parquet
```

## File Naming Conventions
- decisions: `decisions_YYYYMMDD.parquet`
- audits: `audits_YYYYMMDD.parquet`
- stats: `stats_YYYYMMDD.parquet`
- policies: `policies.parquet` (append-only, version columns present)

## Partitioning Strategy
- Primary partition: by date (UTC day). Each daily file holds that day’s rows.
- Optional secondary partitioning (pros/cons must be weighed per volume):
  - By symbol: faster symbol slices; more small files if many symbols.
  - By regime (session/macro): accelerates regime queries; may duplicate small partitions.
  - By timeframe: helpful for multi-timeframe research; increases partition fan-out.
- Guidance:
  - Keep date as the mandatory partition to enable time-window backfills and pruning.
  - Add symbol or regime partitions only if file counts stay manageable; otherwise, push those filters to query engines.

### How partitions support queries
- Per-regime analysis: filter on regime columns; if regime-partitioned, prunes IO, else predicate pushdown on Parquet.
- Per-symbol analysis: symbol partitioning prunes to relevant files; otherwise rely on Parquet stats.
- Time-window backfills: daily partitioning enables idempotent re-writes for specific dates without touching other days.

## Retention & Compaction
- Retain months of data; append-only semantics.
- Old data may be compacted (e.g., merge many daily files into monthly Parquet) but never mutated in-place; compaction writes new files and can tombstone old ones explicitly.
- Reproducibility: historical files are immutable once written; backfills write only missing partitions or replace entire daily partitions deterministically.

## Notes
- All writers must de-duplicate on stable keys (e.g., run_id + decision_id for decisions; policy_version for policies) before append/compaction.
- Schemas live under `schemas/` and drive both validation and Parquet column ordering.
