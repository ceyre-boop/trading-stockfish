# Daily Run

Command:
```
python -m engine.jobs.daily_run --mode PAPER --run-id <uuid>
```

Responsibilities:
- Load active policy (respecting SAFE_MODE if applicable).
- Run engine for the session/day in the chosen mode (SIMULATION/PAPER/LIVE).
- Emit artifacts: `logs/decision_log.jsonl`, feature audits, stats snapshot.
- Update storage for the run (`update_storage_for_run`).
- Optionally compute a lightweight research summary (regime performance for the day).

Modes:
- SIMULATION: no live orders.
- PAPER: live data, simulated fills.
- LIVE: real orders; requires explicit confirmation flag.

Outputs are deterministic and logged under the run’s working directory.
