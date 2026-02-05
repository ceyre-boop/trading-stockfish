import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from analytics.replay_day import ReplayEngine

SCENARIO_DIR = Path("research/scenarios")
OUTPUT_DIR = Path("logs/replay_summary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)  # deterministic selection


def _load_scenarios() -> List[Dict]:
    scenarios = []
    for path in sorted(SCENARIO_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = path
        scenarios.append(data)
    return scenarios


def _scenario_month(scn: Dict) -> str:
    date_str = scn.get("date")
    if not date_str:
        return "unknown"
    return date_str[:7]  # YYYY-MM


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


def main() -> None:
    scenarios = _load_scenarios()
    if not scenarios:
        raise SystemExit("No scenarios found under research/scenarios")

    scenarios_by_month: Dict[str, List[Dict]] = defaultdict(list)
    for scn in scenarios:
        scenarios_by_month[_scenario_month(scn)].append(scn)

    months = sorted(m for m in scenarios_by_month.keys() if m != "unknown")
    if not months:
        raise SystemExit("No dated scenarios found")

    month = random.choice(months)
    month_scenarios = sorted(scenarios_by_month[month], key=lambda s: s.get("date"))

    month_summary = {
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

    for scn in month_scenarios:
        symbol = scn.get("instrument", "UNKNOWN")
        df = _scenario_to_ohlcv(scn)
        engine = ReplayEngine(symbol=symbol, data=df, verbose=False)
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

        month_summary["scenarios"].append(scenario_entry)
        month_summary["totals"]["total_pnl"] += scenario_entry["total_pnl"]
        month_summary["totals"]["max_drawdown"] = min(
            month_summary["totals"]["max_drawdown"], scenario_entry["max_drawdown"]
        )
        month_summary["totals"]["num_trades"] += scenario_entry["num_trades"]
        month_summary["totals"]["wins"] += scenario_entry["wins"]
        month_summary["totals"]["losses"] += scenario_entry["losses"]

    total_trades = month_summary["totals"]["num_trades"]
    if total_trades:
        month_summary["totals"]["win_rate"] = (
            month_summary["totals"]["wins"] / total_trades
        )

    output_path = OUTPUT_DIR / f"replay_{month}.json"
    output_path.write_text(json.dumps(month_summary, indent=2), encoding="utf-8")
    print(f"Replay summary written to {output_path}")


if __name__ == "__main__":
    main()
