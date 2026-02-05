import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from analytics.replay_day import ReplayEngine
from sandbox_policy_override import SandboxPolicyOverride

SCENARIO_ROOT = Path("research/scenarios")
OUTPUT_DIR = Path("logs/aggressive_replay")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)  # deterministic selection


def _load_scenarios_by_month(root: Path) -> Dict[str, List[Dict]]:
    scenarios_by_month: Dict[str, List[Dict]] = defaultdict(list)

    # Support both nested month folders and flat files with date field.
    for path in sorted(root.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = path
        date_str = data.get("date") or ""
        month = "unknown"
        if len(date_str) >= 7:
            month = date_str[:7]
        elif path.parent != root:
            # Fallback to parent folder name if no date present
            month = path.parent.name
        scenarios_by_month[month].append(data)

    # Keep deterministic ordering
    for m in scenarios_by_month:
        scenarios_by_month[m] = sorted(
            scenarios_by_month[m], key=lambda s: s.get("date", "")
        )

    return scenarios_by_month


def _scenario_to_ohlcv(scn: Dict) -> pd.DataFrame:
    ticks = scn.get("ticks", [])
    if not ticks:
        raise ValueError(f"Scenario {scn.get('_path')} has no ticks")
    df = pd.DataFrame(ticks)
    if "price" not in df.columns:
        raise ValueError(f"Scenario {scn.get('_path')} missing price column")
    df = df.sort_values("timestamp")
    df["open"] = df["price"]
    df["high"] = df["price"]
    df["low"] = df["price"]
    df["close"] = df["price"]
    if "volume" not in df.columns:
        df["volume"] = 0
    df["volume"] = df["volume"].fillna(0)
    return df[["open", "high", "low", "close", "volume"]]


def _compute_trade_stats(snapshots) -> Dict:
    num_trades = sum(1 for s in snapshots if (s.filled_size or 0) != 0)
    wins = 0
    losses = 0
    prev_realized = 0.0
    for s in snapshots:
        delta = s.realized_pnl - prev_realized
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        prev_realized = s.realized_pnl
    win_rate = (wins / max(num_trades, 1)) if num_trades else 0.0
    return {
        "num_trades": num_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
    }


def _wrap_policy(engine: ReplayEngine):
    # Monkeypatch _apply_policy with a sandbox override while keeping the original intact.
    original_apply = engine._apply_policy
    override = SandboxPolicyOverride(original_apply)

    def patched(eval_result, snapshot):
        return override.evaluate(eval_result, snapshot)

    engine._apply_policy = patched  # type: ignore[attr-defined]


def run_month(month: str) -> Path:
    scenarios_by_month = _load_scenarios_by_month(SCENARIO_ROOT)
    if month == "auto":
        months = sorted(m for m in scenarios_by_month.keys() if m != "unknown")
        if not months:
            raise SystemExit("No dated scenarios found")
        month = months[0]

    if month not in scenarios_by_month:
        raise SystemExit(f"No scenarios for month {month}")

    scenarios = scenarios_by_month[month]
    summary = {
        "month": month,
        "scenarios": [],
        "totals": {
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "num_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
        },
        "events": {
            "guardrail": [],
            "anomaly": [],
            "health": [],
            "safe_mode": [],
        },
    }

    for scn in scenarios:
        symbol = scn.get("instrument", "UNKNOWN")
        df = _scenario_to_ohlcv(scn)
        engine = ReplayEngine(symbol=symbol, data=df, verbose=False)
        _wrap_policy(engine)

        snapshots = engine.run_full()
        engine.session.compute_stats()

        stats = engine.session.stats or {}
        trade_stats = _compute_trade_stats(snapshots)

        scenario_entry = {
            "scenario": str(scn.get("_path")),
            "date": scn.get("date"),
            "symbol": symbol,
            "total_pnl": stats.get("final_pnl", 0.0),
            "max_drawdown": stats.get("max_drawdown", 0.0),
            "daily_pnl": [s.daily_pnl for s in snapshots] if snapshots else [],
            "num_trades": trade_stats["num_trades"],
            "wins": trade_stats["wins"],
            "losses": trade_stats["losses"],
            "win_rate": trade_stats["win_rate"],
            "guardrail_events": [],
            "anomaly_events": [],
            "health_events": [],
            "safe_mode_events": [],
        }

        summary["scenarios"].append(scenario_entry)
        summary["totals"]["total_pnl"] += scenario_entry["total_pnl"]
        summary["totals"]["max_drawdown"] = min(
            summary["totals"]["max_drawdown"], scenario_entry["max_drawdown"]
        )
        summary["totals"]["num_trades"] += scenario_entry["num_trades"]
        summary["totals"]["wins"] += scenario_entry["wins"]
        summary["totals"]["losses"] += scenario_entry["losses"]

    total_trades = summary["totals"]["num_trades"]
    if total_trades:
        summary["totals"]["win_rate"] = summary["totals"]["wins"] / total_trades

    output_path = OUTPUT_DIR / f"replay_{month}.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Aggressive replay summary written to {output_path}")
    return output_path


def main():
    month = "auto"
    if len(sys.argv) > 1:
        month = sys.argv[1]
    run_month(month)


if __name__ == "__main__":
    main()
