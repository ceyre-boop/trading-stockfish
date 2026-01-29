"""
Reporting utilities for Trading Stockfish v4.0‑F
Deterministic, human-readable, JSON-exportable reports.
"""

import json
import os
from typing import Any, Dict


def generate_factor_contribution_report(
    match_results: Dict[str, Any],
) -> Dict[str, Any]:
    # Aggregate factor contributions from evaluation traces
    report = {}
    for match, data in match_results.items():
        factors = data.get("evaluation_trace", [])
        for f in factors:
            for k, v in f.get("factors", {}).items():
                report.setdefault(k, []).append(v)
    # Compute averages
    for k in report:
        vals = report[k]
        report[k] = sum(vals) / len(vals) if vals else 0.0
    return report


def generate_regime_performance_report(match_results: Dict[str, Any]) -> Dict[str, Any]:
    # Aggregate performance by regime
    report = {}
    for match, data in match_results.items():
        regimes = data.get("regime_transitions", [])
        pnl = data.get("pnl_curve", [])
        for r in regimes:
            tag = r.get("regime", "UNKNOWN")
            report.setdefault(tag, []).append(pnl[-1] if pnl else 0.0)
    for tag in report:
        vals = report[tag]
        report[tag] = sum(vals) / len(vals) if vals else 0.0
    return report


def generate_execution_quality_report(match_results: Dict[str, Any]) -> Dict[str, Any]:
    # Compare execution results to baseline
    report = {}
    for match, data in match_results.items():
        execs = data.get("execution_results", [])
        for e in execs:
            instr = e.get("instrument", "UNKNOWN")
            slippage = e.get("slippage", 0.0)
            report.setdefault(instr, []).append(slippage)
    for instr in report:
        vals = report[instr]
        report[instr] = sum(vals) / len(vals) if vals else 0.0
    return report


def generate_elo_summary(elo_ratings: Dict[str, float]) -> str:
    return json.dumps(elo_ratings, indent=2, sort_keys=True)


def generate_experiment_report(experiment_id: str) -> Dict[str, Any]:
    """
    Loads all logs for experiment_id, reconstructs the timeline, and summarizes the experiment.
    Returns a JSON-serializable dict and a human-readable markdown summary.
    """
    logs_dir = os.path.join("logs")
    log_file = os.path.join(logs_dir, f"{experiment_id}.jsonl")
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Experiment log not found: {log_file}")

    # Load all log lines
    events = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                evt = json.loads(line)
                events.append(evt)
            except Exception:
                continue

    # Reconstruct timeline
    entry, exit, during = None, None, []
    for evt in events:
        if evt.get("phase") == "entry":
            entry = evt
        elif evt.get("phase") == "exit":
            exit = evt
        elif evt.get("phase") == "during":
            during.append(evt)

    regimes = []
    confidences = []
    sizes = []
    factor_totals = {}
    factor_counts = {}
    for evt in events:
        ev = evt.get("evaluator_output", {}) or {}
        pd = evt.get("policy_decision", {}) or {}
        ms = evt.get("market_state", {}) or {}
        reg = ms.get("regime") or ms.get("regime_state")
        if reg:
            regimes.append(reg)
        if "confidence" in ev:
            confidences.append(ev.get("confidence"))
        if "size" in pd:
            sizes.append(pd.get("size"))
        for k, v in (ev.get("factors") or {}).items():
            factor_totals[k] = factor_totals.get(k, 0.0) + v
            factor_counts[k] = factor_counts.get(k, 0) + 1

    regime_distribution = {}
    for reg in regimes:
        key = json.dumps(reg, sort_keys=True) if isinstance(reg, dict) else str(reg)
        regime_distribution[key] = regime_distribution.get(key, 0) + 1

    factor_contributions = {}
    for k, total in factor_totals.items():
        cnt = factor_counts.get(k, 0)
        factor_contributions[k] = total / cnt if cnt else 0.0

    confidence_stats = {
        "min": min(confidences) if confidences else None,
        "max": max(confidences) if confidences else None,
        "mean": (sum(confidences) / len(confidences)) if confidences else None,
        "count": len(confidences),
    }

    size_stats = {
        "min": min(sizes) if sizes else None,
        "max": max(sizes) if sizes else None,
        "mean": (sum(sizes) / len(sizes)) if sizes else None,
        "count": len(sizes),
    }

    # Summarize
    exec_res = exit.get("execution_result", {}) if exit else {}
    pnl_value = exec_res.get("realized_pnl") if exec_res else None
    if pnl_value is None:
        pnl_value = exec_res.get("pnl") if exec_res else None

    summary = {
        "experiment_id": experiment_id,
        "entry": entry,
        "exit": exit,
        "during": during,
        "trade_opened": entry is not None,
        "trade_closed": exit is not None,
        "pnl": pnl_value,
        "max_adverse_excursion": (
            exit.get("execution_result", {}).get("max_adverse_excursion")
            if exit
            else None
        ),
        "max_favorable_excursion": (
            exit.get("execution_result", {}).get("max_favorable_excursion")
            if exit
            else None
        ),
        "regime_evolution": [
            d.get("regime_state") or d.get("market_state", {}).get("regime")
            for d in during
            if "regime_state" in d or d.get("market_state", {}).get("regime")
        ],
        "reason_for_entry": (
            entry.get("evaluator_output", {}).get("reason") if entry else None
        ),
        "reason_for_exit": (
            exit.get("policy_decision", {}).get("reason") if exit else None
        ),
        "governance_at_entry": entry.get("governance_decision") if entry else None,
        "governance_at_exit": exit.get("governance_decision") if exit else None,
        "regime_distribution": regime_distribution,
        "factor_contributions": factor_contributions,
        "confidence_stats": confidence_stats,
        "size_stats": size_stats,
    }

    # Markdown/text summary
    md = f"""# Experiment Report: {experiment_id}\n\n## Trade Opened: {summary['trade_opened']}\n## Trade Closed: {summary['trade_closed']}\n## PnL: {summary['pnl']}\n\n### Entry Reason:\n{summary['reason_for_entry']}\n\n### Exit Reason:\n{summary['reason_for_exit']}\n\n### Regime Evolution:\n{summary['regime_evolution']}\n\n### Regime Distribution:\n{summary['regime_distribution']}\n\n### Factor Contributions (avg):\n{summary['factor_contributions']}\n\n### Confidence Stats:\n{summary['confidence_stats']}\n\n### Size Stats:\n{summary['size_stats']}\n\n### Governance at Entry:\n{summary['governance_at_entry']}\n\n### Governance at Exit:\n{summary['governance_at_exit']}\n\n"""

    summary["markdown"] = md
    return summary
