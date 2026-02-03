"""Feedback loop driver for stats \u2192 policy updates.

This module orchestrates the offline nightly/weekly loop without altering
runtime evaluators. It loads audits and decision logs, runs drift + stats,
proposes a policy, gates it, applies safety rules, and writes a feedback
summary.
"""

from __future__ import annotations

import datetime
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from analytics.feature_drift_detector import detect_drifts, load_audits
from analytics.policy_builder import build_policy_from_stats
from engine.policy_gating import (
    BacktestResult,
    GatingConfig,
    GatingDecision,
    evaluate_candidate_policy,
)
from engine.policy_loader import load_policy
from engine.regime_multipliers import StatsResult
from engine.safety_mode import (
    SafetyConfig,
    SafetyDecision,
    apply_safety_decision,
    check_and_update_safety,
)


@dataclass
class FeedbackConfig:
    audit_dir: Path = Path("logs/feature_audits")
    decision_log_path: Path = Path("logs/decision_log.jsonl")
    stats_path: Path = Path("logs/feature_stats.json")
    policy_path: Path = Path("logs/policy_config.json")
    archive_dir: Path = Path("logs/policy_archive")
    feedback_dir: Path = Path("logs/feedback")
    window_days: Optional[int] = None
    drift_window: int = 20
    drift_spike_factor: float = 2.0
    drift_abs_threshold: int = 2
    drift_missing_frac_threshold: float = 0.8
    gate_config: GatingConfig = field(default_factory=GatingConfig)
    safety_config: SafetyConfig = field(default_factory=SafetyConfig)
    backtest_runner: Optional[Callable[[Dict[str, Any]], BacktestResult]] = None
    stats_loader: Optional[Callable[[Path], StatsResult]] = None
    gating_history: List[str] = field(default_factory=list)


@dataclass
class FeedbackResult:
    promoted: bool
    gate_decision: GatingDecision
    safety_decision: SafetyDecision
    feedback_summary_path: Path
    active_policy_path: Path
    archived_policy_path: Optional[Path]


def _parse_utc(ts: str) -> Optional[datetime]:
    try:
        if not ts:
            return None
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts).astimezone(timezone.utc)
    except Exception:
        return None


def _within_window(ts_str: Any, window_days: Optional[int]) -> bool:
    if window_days is None:
        return True
    ts = _parse_utc(str(ts_str)) if ts_str is not None else None
    if ts is None:
        return False
    now = datetime.now(timezone.utc)
    return ts >= now - timedelta(days=window_days)


def _load_decision_logs(path: Path, window_days: Optional[int]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
            if _within_window(obj.get("timestamp_utc"), window_days):
                entries.append(obj)
        except Exception:
            continue
    return entries


def _decision_counts(logs: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entry in logs:
        action = str(entry.get("action", "UNKNOWN")).upper()
        counts[action] = counts.get(action, 0) + 1
    return counts


def _load_stats(path: Path) -> StatsResult:
    if not path.exists():
        return StatsResult()
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return StatsResult()

    feature_importance = obj.get("feature_importance") or {}
    feature_stability = obj.get("feature_stability") or {}

    # If not present, derive stability from zscore_stability fields
    features_block = (
        obj.get("features", {}) if isinstance(obj.get("features"), dict) else {}
    )
    if not feature_importance and features_block:
        feature_importance = {
            k: float(v.get("mean", 1.0) or 1.0)
            for k, v in features_block.items()
            if isinstance(v, dict)
        }
    if not feature_stability and features_block:
        feature_stability = {
            k: float(v.get("zscore_stability", 1.0) or 1.0)
            for k, v in features_block.items()
            if isinstance(v, dict)
        }

    feature_perf_regime = obj.get("feature_performance_by_regime") or {}

    metadata = {
        "run_id": obj.get("run_id"),
        "experiment_id": obj.get("experiment_id"),
        "engine_version": obj.get("engine_version"),
    }

    return StatsResult(
        feature_importance=feature_importance,
        feature_stability=feature_stability,
        feature_performance_by_regime=feature_perf_regime,
        metadata=metadata,
    )


def _default_backtest(decision_logs: List[Dict[str, Any]]) -> BacktestResult:
    counts = _decision_counts(decision_logs)
    total = sum(counts.values()) or 1
    hit_rate = counts.get("LONG", 0) / total
    sharpe = 1.0 if hit_rate >= 0.5 else 0.0
    max_dd = 0.1 if sharpe > 0 else 0.3
    pnl = counts.get("LONG", 0) - counts.get("SHORT", 0) * 0.5
    return BacktestResult(
        pnl=float(pnl),
        sharpe=float(sharpe),
        hit_rate=float(hit_rate),
        max_drawdown=float(max_dd),
        per_regime={},
    )


def _propose_policy_from_stats(
    stats_result: StatsResult, run_id: str
) -> Dict[str, Any]:
    return build_policy_from_stats(
        stats_result,
        config={
            "run_id": run_id,
            "experiment_id": stats_result.metadata.get("experiment_id"),
            "engine_version": stats_result.metadata.get("engine_version"),
        },
    )


def _archive_policy(
    current_path: Path, archive_dir: Path, suffix: str
) -> Optional[Path]:
    if not current_path.exists():
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / f"{current_path.stem}_{suffix}{current_path.suffix}"
    shutil.copy2(current_path, target)
    return target


def _write_policy(path: Path, policy: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")


def run_feedback_loop(run_id: str, config: FeedbackConfig) -> FeedbackResult:
    audit_dir = Path(config.audit_dir)
    decision_log_path = Path(config.decision_log_path)
    stats_path = Path(config.stats_path)
    policy_path = Path(config.policy_path)
    archive_dir = Path(config.archive_dir)
    feedback_dir = Path(config.feedback_dir)

    audits = [
        a
        for a in load_audits(audit_dir)
        if _within_window(a.timestamp_utc, config.window_days)
    ]
    drift_findings = detect_drifts(
        records=audits,
        window=max(1, config.drift_window),
        spike_factor=max(1.0, config.drift_spike_factor),
        abs_threshold=max(1, config.drift_abs_threshold),
        missing_frac_threshold=max(0.0, min(1.0, config.drift_missing_frac_threshold)),
    )
    drift_report = {
        "findings": drift_findings,
        "aggregates": {
            "features_flagged": len(drift_findings),
            "messages": sum(len(v) for v in drift_findings.values()),
        },
    }

    stats_loader = config.stats_loader or _load_stats
    try:
        stats_result = stats_loader(stats_path, audits, None)  # type: ignore[arg-type]
    except TypeError:
        stats_result = stats_loader(stats_path)

    decision_logs = _load_decision_logs(decision_log_path, config.window_days)
    decision_counts = _decision_counts(decision_logs)

    candidate_policy = _propose_policy_from_stats(stats_result, run_id)

    backtest_runner = config.backtest_runner
    if backtest_runner:
        backtest_result = backtest_runner(candidate_policy)
    else:
        backtest_result = _default_backtest(decision_logs)

    gate_decision = evaluate_candidate_policy(
        candidate_policy, backtest_result, config.gate_config
    )

    gating_history = list(config.gating_history)
    gating_history.append(gate_decision.decision)

    prev_safety_state = getattr(config, "safety_state", None)
    safety_decision = check_and_update_safety(
        drift_report, gating_history, config.safety_config, prev_safety_state
    )

    promoted = False
    archived_policy_path: Optional[Path] = None

    last_good_policy = load_policy(policy_path)
    last_good_version = None
    if last_good_policy is not None:
        last_good_version = (last_good_policy.data or {}).get("version")

    safety_mode_triggered = safety_decision.new_state == "SAFE_MODE"
    # Provide fallback to last-good when entering SAFE_MODE
    if (
        safety_mode_triggered
        and safety_decision.fallback_policy is None
        and last_good_policy is not None
    ):
        safety_decision.fallback_policy = last_good_policy.data

    if gate_decision.decision == "PASS" and not safety_mode_triggered:
        suffix_version = last_good_version or "unknown"
        archived_policy_path = _archive_policy(
            policy_path, archive_dir, f"{run_id}_v{suffix_version}_prev"
        )
        _write_policy(policy_path, candidate_policy)
        promoted = True
    else:
        # Do not promote if gating fails or safety overrides
        rejection_suffix = "rejected_safe" if safety_mode_triggered else "rejected"
        _write_policy(
            archive_dir / f"candidate_{run_id}_{rejection_suffix}.json",
            candidate_policy,
        )

    base_policy = (
        candidate_policy
        if promoted
        else (last_good_policy.data if last_good_policy else candidate_policy)
    )
    safe_policy = apply_safety_decision(base_policy, safety_decision)

    if safe_policy is not base_policy:
        _write_policy(policy_path, safe_policy)
        active_policy = safe_policy
    else:
        active_policy = base_policy

    safety_applied = "none"
    if safety_mode_triggered:
        safety_applied = (
            "reverted" if safety_decision.fallback_policy is not None else "dampened"
        )

    gating_policy_version = (
        candidate_policy.get("version")
        if promoted
        else (last_good_version or candidate_policy.get("version"))
    )
    gating_status = "promoted" if promoted else "retained"

    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "drift_summary": drift_report,
        "stats_summary": {
            "feature_importance": stats_result.feature_importance,
            "feature_stability": stats_result.feature_stability,
        },
        "decision_counts": decision_counts,
        "backtest_metrics": {
            "pnl": getattr(backtest_result, "pnl", None),
            "sharpe": getattr(backtest_result, "sharpe", None),
            "hit_rate": getattr(backtest_result, "hit_rate", None),
            "max_drawdown": getattr(backtest_result, "max_drawdown", None),
            "per_regime": getattr(backtest_result, "per_regime", {}),
        },
        "gating": {
            "decision": gate_decision.decision,
            "reasons": gate_decision.reasons,
            "metrics": gate_decision.metrics,
            "policy_version": gating_policy_version,
            "policy_status": gating_status,
            "policy_path": str(policy_path),
        },
        "safety": {
            "previous_state": getattr(prev_safety_state, "current_state", "NORMAL"),
            "new_state": safety_decision.new_state,
            "reason": safety_decision.reason,
            "timestamp_utc": safety_decision.timestamp_utc,
            "action": safety_decision.action,
            "multiplier_scale": safety_decision.multiplier_scale,
            "applied": safety_applied,
            "metadata": safe_policy.get("safety_mode", {}),
        },
        "gate_decision": gate_decision.to_dict(),
        "safety_decision": safety_decision.to_dict(),
        "promoted": promoted,
        "candidate_policy_metadata": {
            "version": candidate_policy.get("version"),
            "run_id": candidate_policy.get("run_id"),
            "experiment_id": candidate_policy.get("experiment_id"),
        },
        "active_policy_path": str(policy_path),
        "archived_policy_path": (
            str(archived_policy_path) if archived_policy_path else None
        ),
    }

    feedback_dir.mkdir(parents=True, exist_ok=True)
    summary_path = feedback_dir / f"feedback_run_{run_id}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    return FeedbackResult(
        promoted=promoted,
        gate_decision=gate_decision,
        safety_decision=safety_decision,
        feedback_summary_path=summary_path,
        active_policy_path=policy_path,
        archived_policy_path=archived_policy_path,
    )
