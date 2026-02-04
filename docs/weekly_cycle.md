# Weekly Cycle

Steps (deterministic, offline-friendly):
- Run feedback loop over the last N days.
- Backfill storage for the week if needed (date-windowed).
- Generate research report using `research_api`:
  - Per-regime performance
  - Feature drift over time
  - Policy version performance
  - Safety events/incidence
- Optional: retrain ML priors on the weekly window.
- Produce `reports/weekly_report_<date>.md` (or .html) with findings and metrics.

Execution: `python -m engine.jobs.weekly_cycle --mode SIMULATION --days 7`

No promotions or live changes occur unless explicitly enabled; defaults are read-only and deterministic.
