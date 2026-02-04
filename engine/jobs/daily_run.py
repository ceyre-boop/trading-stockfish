from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from engine import research_api
from engine.jobs import storage_jobs
from engine.modes import Mode, get_adapter, resolve_mode


@contextmanager
def _working_dir(path: Path):
    prev = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev)


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def run_daily(mode_str: str, run_id: str, base_dir: Path = Path("logs"), compute_research_summary: bool = False, confirm_live: bool = False) -> Dict[str, object]:
    mode = resolve_mode(mode_str)
    if mode == Mode.LIVE and not confirm_live:
        raise ValueError("LIVE mode requires confirm_live=True")

    adapter = get_adapter(mode)

    base_dir.mkdir(parents=True, exist_ok=True)
    decision_log = base_dir / "decision_log.jsonl"
    audit_dir = base_dir / "feature_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"audit_{run_id}.json"
    stats_path = base_dir / "feature_stats.json"

    # Simulate engine outputs deterministically
    decision_entry = {
        "run_id": run_id,
        "decision_id": f"{run_id}_d1",
        "timestamp_utc": _timestamp(),
        "symbol": "TEST",
        "timeframe": "M5",
        "session_regime": "SIM_SESSION",
        "macro_regimes": ["SIM_MACRO"],
        "feature_vector": {"feature_a": 1.0},
        "effective_weights": {"feature_a": 1.0},
        "policy_components": {
            "base_weights": {"feature_a": 1.0},
            "trust": {"feature_a": 1.0},
            "regime_multipliers": {},
        },
        "evaluation_score": 0.0,
        "action": "HOLD",
        "position_size": 0.0,
        "outcome": None,
        "provenance": {"policy_version": "active", "engine_version": "sim"},
    }
    decision_log.write_text(json.dumps(decision_entry) + "\n", encoding="utf-8")

    audit_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "timestamp_utc": decision_entry["timestamp_utc"],
                "issues": [],
                "summary": {},
            }
        ),
        encoding="utf-8",
    )

    stats_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "timestamp_utc": decision_entry["timestamp_utc"],
                "feature_importance": {"feature_a": 1.0},
                "feature_stability": {"feature_a": 1.0},
            }
        ),
        encoding="utf-8",
    )

    # Update storage for this run (append-only Parquet)
    storage_jobs.update_storage_for_run(run_id)

    summary = {}
    if compute_research_summary:
        df = research_api.load_decisions(research_api.DecisionsFilter())
        perf = research_api.compute_regime_performance(df)
        summary = {"regime_performance_rows": len(perf)}

    return {
        "mode": mode.value,
        "run_id": run_id,
        "decision_log": decision_log,
        "audit_path": audit_path,
        "stats_path": stats_path,
        "adapter_disabled": adapter.disabled,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily trading session")
    parser.add_argument("--mode", required=True, help="SIMULATION | PAPER | LIVE")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--confirm-live", action="store_true", help="Required for LIVE mode")
    parser.add_argument("--compute-summary", action="store_true", help="Compute lightweight research summary")
    args = parser.parse_args()

    run_daily(args.mode, args.run_id, confirm_live=args.confirm_live, compute_research_summary=args.compute_summary)


if __name__ == "__main__":
    import os

    main()
