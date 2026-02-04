from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

from engine import research_api
from engine.jobs import storage_jobs
from engine.modes import Mode, resolve_mode


def run_weekly_cycle(
    mode_str: str = "SIMULATION",
    days: int = 7,
    report_dir: Path = Path("reports"),
    allow_promote: bool = False,
    confirm_live: bool = False,
) -> Dict[str, object]:
    mode = resolve_mode(mode_str)
    if mode == Mode.LIVE and not confirm_live:
        raise ValueError("LIVE mode requires confirm_live=True")

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=max(0, days - 1))

    # Deterministic backfill for window
    try:
        storage_jobs.backfill_storage(start.isoformat(), end.isoformat())
    except Exception:
        pass

    # Research report
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"weekly_report_{end.isoformat()}.md"

    decisions = research_api.load_decisions(
        research_api.DecisionsFilter(start_date=start, end_date=end)
    )
    stats = research_api.load_stats(
        research_api.StatsFilter(start_date=start, end_date=end)
    )
    policies = research_api.load_policies(
        research_api.PolicyFilter(start_date=start, end_date=end)
    )

    regime_perf = research_api.compute_regime_performance(decisions)
    drift = research_api.compute_feature_drift_over_time(stats)
    policy_perf = research_api.compute_policy_version_performance(decisions, policies)

    report = [
        f"# Weekly Report {end.isoformat()}",
        f"Mode: {mode.value}",
        f"Window: {start.isoformat()} to {end.isoformat()}",
        "## Regime Performance",
        regime_perf.to_markdown(index=False) if not regime_perf.empty else "(no data)",
        "## Feature Drift",
        drift.head(20).to_markdown(index=False) if not drift.empty else "(no data)",
        "## Policy Version Performance",
        policy_perf.to_markdown(index=False) if not policy_perf.empty else "(no data)",
    ]
    report_path.write_text("\n\n".join(report), encoding="utf-8")

    promoted = False if not allow_promote else False

    return {
        "mode": mode.value,
        "report_path": report_path,
        "window": (start, end),
        "promoted": promoted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weekly research cycle")
    parser.add_argument(
        "--mode", default="SIMULATION", help="SIMULATION | PAPER | LIVE"
    )
    parser.add_argument("--days", type=int, default=7, help="Number of days to include")
    parser.add_argument(
        "--confirm-live", action="store_true", help="Required for LIVE mode"
    )
    args = parser.parse_args()
    run_weekly_cycle(args.mode, args.days, confirm_live=args.confirm_live)


if __name__ == "__main__":
    main()
