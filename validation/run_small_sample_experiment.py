"""
Small-sample experiment using mock dependencies is disabled. Real data only.
"""

raise RuntimeError(
    "run_small_sample_experiment is disabled: synthetic/mock pipelines are forbidden."
)

import json
import os
from typing import Any, Dict, List, Tuple

from realtime.engine_loop import create_engine_loop
from research.reporting import generate_experiment_report
from validation.run_full_replay import load_engine_dependencies

SCENARIO_PATHS = [
    os.path.join("research", "scenarios", "today_replay_experiment.json"),
    os.path.join("research", "scenarios", "down_move_test.json"),
]
RUNS_PER_SCENARIO = 5
MIN_CONFIDENCE = 0.55
MIN_SIZE = 0.01


def load_scenario(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scenario not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)


def adjust_confidence(diag: Dict[str, Any]) -> Dict[str, Any]:
    ev = diag.get("evaluator_output", {}) or {}
    if isinstance(ev, dict):
        ev_conf = ev.get("confidence", 0)
        if ev_conf < MIN_CONFIDENCE:
            ev["confidence"] = MIN_CONFIDENCE
        diag["evaluator_output"] = ev
    return diag


def enforce_min_size(policy_decision: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(policy_decision, dict):
        return {"action": "FLAT", "size": 0}
    action = policy_decision.get("action")
    if action and action not in ["FLAT", "HOLD", "EXIT", "CLOSE", "sell", "SELL"]:
        size = policy_decision.get("size", 0)
        if size <= 0:
            policy_decision["size"] = MIN_SIZE
    return policy_decision


def run_single_pass(run_id: int, scenario: Dict[str, Any]) -> Dict[str, Any]:
    experiment_id = f"small_sample_run_{run_id}"
    log_path = os.path.join("logs", f"{experiment_id}.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    deps = load_engine_dependencies()
    engine = create_engine_loop(
        {**deps, "experiment_one_shot": False, "experiment_id": experiment_id}
    )

    # Trade tracking
    trade_open = False
    trade_id = 0
    trades: List[Dict[str, Any]] = []
    current_entry: Dict[str, Any] = {}

    with open(log_path, "a", encoding="utf-8") as f:
        ticks = scenario.get("ticks", [])
        books = scenario.get("order_books", [])
        for idx, tick in enumerate(ticks):
            book = books[idx] if idx < len(books) else {}
            tick.setdefault("volume", 1)
            engine.run_step(
                {
                    "tick": tick,
                    "book": book,
                    "market": scenario.get("instrument", "unknown"),
                }
            )
            diag = engine.get_last_diagnostics() or {}
            diag = adjust_confidence(diag)
            policy_decision = enforce_min_size(diag.get("policy_decision", {}))
            diag["policy_decision"] = policy_decision

            action = policy_decision.get("action")
            exec_res = diag.get("execution_result", {}) or {}

            # Exit detection via execution_result
            if trade_open and "exit_price" in exec_res:
                trade_open = False
                record = {**diag, "phase": "exit", "trade_id": trade_id}
                f.write(json.dumps(record) + "\n")
                trades[-1]["exit"] = diag
                continue

            # Entry detection via action
            if not trade_open and action in [
                "ENTER_LONG",
                "ENTER_SHORT",
                "buy",
                "BUY",
                "SELL",
            ]:
                trade_open = True
                trade_id += 1
                current_entry = {
                    "trade_id": trade_id,
                    "entry_diag": diag,
                    "regime": diag.get("market_state", {}).get("regime_state"),
                }
                record = {**diag, "phase": "entry", "trade_id": trade_id}
                f.write(json.dumps(record) + "\n")
                trades.append({"trade_id": trade_id, "entry": diag})
                continue

            # Exit detection via explicit action
            if trade_open and action in ["EXIT", "CLOSE", "FLAT", "sell", "SELL"]:
                trade_open = False
                record = {**diag, "phase": "exit", "trade_id": trade_id}
                f.write(json.dumps(record) + "\n")
                trades[-1]["exit"] = diag
                continue

            # During trade logging
            phase = "during" if trade_open else "idle"
            record = {
                **diag,
                "phase": phase,
                "trade_id": trade_id if trade_open else None,
            }
            f.write(json.dumps(record) + "\n")

    # Generate per-run report
    report = generate_experiment_report(experiment_id)
    with open(
        os.path.join("reports", f"{experiment_id}.json"), "w", encoding="utf-8"
    ) as jf:
        json.dump(report, jf, indent=2)
    with open(
        os.path.join("reports", f"{experiment_id}.md"), "w", encoding="utf-8"
    ) as mf:
        mf.write(report.get("markdown", ""))

    return {"experiment_id": experiment_id, "trades": trades, "report": report}


def aggregate_summary(run_results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    total_trades = 0
    pnls: List[float] = []
    confidences: List[float] = []
    regimes: Dict[str, int] = {}
    factor_counts: Dict[str, int] = {}
    governance_vetoes = 0

    for res in run_results:
        # Count trades by presence of exit in run report
        report = res.get("report", {})
        if report.get("trade_opened"):
            total_trades += 1
        pnl = report.get("pnl")
        if pnl is not None:
            pnls.append(pnl)
        entry = report.get("entry") or {}
        ev = entry.get("evaluator_output") or {}
        conf = ev.get("confidence")
        if conf is not None:
            confidences.append(conf)
        regime_state = None
        if entry:
            ms = entry.get("market_state", {})
            regime_state = ms.get("regime_state") or ms.get("regime")
        if regime_state:
            key = (
                json.dumps(regime_state, sort_keys=True)
                if isinstance(regime_state, dict)
                else str(regime_state)
            )
            regimes[key] = regimes.get(key, 0) + 1
        factors = ev.get("factors") or {}
        for k in factors.keys():
            factor_counts[k] = factor_counts.get(k, 0) + 1
        gov = report.get("governance_at_entry") or {}
        if isinstance(gov, dict) and not gov.get("approved", True):
            governance_vetoes += 1

    avg_pnl = sum(pnls) / len(pnls) if pnls else None
    avg_conf = sum(confidences) / len(confidences) if confidences else None

    summary = {
        "runs": len(run_results),
        "total_trades": total_trades,
        "wins": len([p for p in pnls if p is not None and p > 0]),
        "losses": len([p for p in pnls if p is not None and p < 0]),
        "average_pnl": avg_pnl,
        "average_confidence": avg_conf,
        "regime_distribution": regimes,
        "factor_contributions": factor_counts,
        "governance_veto_frequency": governance_vetoes,
    }

    md = [
        "# Small Sample Summary",
        "",
        f"Runs: {len(run_results)}",
        f"Total trades: {total_trades}",
    ]
    md.append(f"Wins: {summary['wins']} | Losses: {summary['losses']}")
    md.append(f"Average PnL: {avg_pnl}")
    md.append(f"Average Confidence: {avg_conf}")
    md.append(f"Governance vetoes: {governance_vetoes}")
    md.append("Regime distribution:")
    for k, v in regimes.items():
        md.append(f"- {k}: {v}")
    md.append("Factor contributions:")
    for k, v in factor_counts.items():
        md.append(f"- {k}: {v}")
    markdown = "\n".join(md)

    return summary, markdown


def main():
    ensure_dirs()
    run_results: List[Dict[str, Any]] = []
    run_counter = 0
    for scenario_path in SCENARIO_PATHS:
        scenario = load_scenario(scenario_path)
        for i in range(1, RUNS_PER_SCENARIO + 1):
            run_counter += 1
            result = run_single_pass(run_counter, scenario)
            run_results.append(result)

    summary, markdown = aggregate_summary(run_results)
    with open(
        os.path.join("reports", "small_sample_summary.json"), "w", encoding="utf-8"
    ) as jf:
        json.dump(summary, jf, indent=2)
    with open(
        os.path.join("reports", "small_sample_summary.md"), "w", encoding="utf-8"
    ) as mf:
        mf.write(markdown)
    print("Small-sample experiment completed.")
    print(
        f"Total trades: {summary['total_trades']}, Wins: {summary['wins']}, Losses: {summary['losses']}"
    )


if __name__ == "__main__":
    main()
