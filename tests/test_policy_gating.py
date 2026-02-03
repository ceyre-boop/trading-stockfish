import json
from pathlib import Path

import pytest

from engine.feedback_loop import FeedbackConfig, run_feedback_loop
from engine.policy_gating import (
    BacktestResult,
    GatingConfig,
    GatingDecision,
    evaluate_candidate_policy,
)
from engine.regime_multipliers import StatsResult


def test_gate_passes_strong_candidate():
    candidate = {"id": "candidate"}
    backtest = BacktestResult(
        pnl=10.0,
        sharpe=1.5,
        hit_rate=0.7,
        max_drawdown=0.05,
        per_regime={"HIGH_VOL": 1.0, "LOW_VOL": 0.8},
    )
    cfg = GatingConfig(
        min_sharpe=0.8, min_hit_rate=0.55, max_drawdown=0.2, min_per_regime=0.0
    )
    decision = evaluate_candidate_policy(candidate, backtest, cfg)
    assert decision.decision == "PASS"
    assert decision.reasons == []
    assert decision.metrics["pnl"] == pytest.approx(10.0)
    assert decision.metrics["per_regime"]["HIGH_VOL"] == pytest.approx(1.0)


def test_gate_rejects_weak_candidate():
    candidate = {"id": "candidate"}
    backtest = BacktestResult(
        pnl=-2.0,
        sharpe=0.4,
        hit_rate=0.48,
        max_drawdown=0.25,
        per_regime={"HIGH_VOL": -1.5},
    )
    cfg = GatingConfig(
        min_sharpe=0.8, min_hit_rate=0.55, max_drawdown=0.2, min_per_regime=-0.5
    )
    decision = evaluate_candidate_policy(candidate, backtest, cfg)
    assert decision.decision == "FAIL"
    assert any("sharpe_below_threshold" in r for r in decision.reasons)
    assert any("hit_rate_below_threshold" in r for r in decision.reasons)
    assert any("max_drawdown_exceeded" in r for r in decision.reasons)
    assert any("regime_floor_breach" in r for r in decision.reasons)


def test_feedback_loop_respects_gate_pass(tmp_path: Path, monkeypatch):
    audits_dir = tmp_path / "audits"
    decision_log = tmp_path / "decision_log.jsonl"
    stats_path = tmp_path / "feature_stats.json"
    policy_path = tmp_path / "policy.json"
    archive_dir = tmp_path / "archive"
    feedback_dir = tmp_path / "feedback"

    policy_path.write_text(
        json.dumps({"version": "0.1", "base_weights": {"old": 1}}), encoding="utf-8"
    )

    stats_result = StatsResult(
        feature_importance={"f1": 1.0}, feature_stability={"f1": 1.0}
    )
    monkeypatch.setattr(
        "engine.feedback_loop._load_stats", lambda *args, **kwargs: stats_result
    )
    monkeypatch.setattr(
        "engine.feedback_loop._propose_policy_from_stats",
        lambda s, rid: {"version": "0.2", "base_weights": {"f1": 1.0}},
    )

    strong_bt = BacktestResult(pnl=5.0, sharpe=1.2, hit_rate=0.6, max_drawdown=0.1)
    cfg = FeedbackConfig(
        audit_dir=audits_dir,
        decision_log_path=decision_log,
        stats_path=stats_path,
        policy_path=policy_path,
        archive_dir=archive_dir,
        feedback_dir=feedback_dir,
        backtest_runner=lambda policy: strong_bt,
    )

    monkeypatch.setattr(
        "engine.feedback_loop.evaluate_candidate_policy",
        lambda candidate, bt, gate_cfg: GatingDecision("PASS", [], {"pnl": bt.pnl}),
    )

    result = run_feedback_loop("pass_run", cfg)

    assert result.promoted is True
    assert policy_path.read_text(encoding="utf-8")
    archived = list(archive_dir.glob("*prev*.json"))
    assert archived, "Expected archived previous policy"

    summary_path = feedback_dir / "feedback_run_pass_run.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["gating"]["decision"] == "PASS"


def test_feedback_loop_respects_gate_fail(tmp_path: Path, monkeypatch):
    audits_dir = tmp_path / "audits"
    decision_log = tmp_path / "decision_log.jsonl"
    stats_path = tmp_path / "feature_stats.json"
    policy_path = tmp_path / "policy.json"
    archive_dir = tmp_path / "archive"
    feedback_dir = tmp_path / "feedback"

    original_policy = {"version": "0.1", "base_weights": {"old": 1}}
    policy_path.write_text(json.dumps(original_policy), encoding="utf-8")

    stats_result = StatsResult(
        feature_importance={"f1": 1.0}, feature_stability={"f1": 1.0}
    )
    monkeypatch.setattr(
        "engine.feedback_loop._load_stats", lambda *args, **kwargs: stats_result
    )
    monkeypatch.setattr(
        "engine.feedback_loop._propose_policy_from_stats",
        lambda s, rid: {"version": "0.2", "base_weights": {"f1": 1.0}},
    )

    weak_bt = BacktestResult(pnl=-1.0, sharpe=0.2, hit_rate=0.4, max_drawdown=0.3)
    cfg = FeedbackConfig(
        audit_dir=audits_dir,
        decision_log_path=decision_log,
        stats_path=stats_path,
        policy_path=policy_path,
        archive_dir=archive_dir,
        feedback_dir=feedback_dir,
        backtest_runner=lambda policy: weak_bt,
    )

    monkeypatch.setattr(
        "engine.feedback_loop.evaluate_candidate_policy",
        lambda candidate, bt, gate_cfg: GatingDecision(
            "FAIL", ["sharpe_below_threshold"], {"pnl": bt.pnl}
        ),
    )

    result = run_feedback_loop("fail_run", cfg)

    assert result.promoted is False
    assert json.loads(policy_path.read_text(encoding="utf-8")) == original_policy
    rejected = list(archive_dir.glob("candidate_fail_run_rejected.json"))
    assert rejected, "Expected rejected candidate to be archived"

    summary_path = feedback_dir / "feedback_run_fail_run.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["gating"]["decision"] == "FAIL"
    assert "sharpe_below_threshold" in summary["gating"]["reasons"]
