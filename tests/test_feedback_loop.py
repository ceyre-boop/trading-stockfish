import json
from pathlib import Path

from engine.feedback_loop import FeedbackConfig, run_feedback_loop
from engine.policy_gating import GatingDecision


def _write_audit(dir_path: Path, name: str = "audit.json") -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / name).write_text(
        json.dumps(
            {
                "run_id": "r1",
                "experiment_id": "exp1",
                "timestamp_utc": "2026-02-02T00:00:00Z",
                "summary": {},
                "issues": [],
            }
        ),
        encoding="utf-8",
    )


def _write_stats(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": "stats_run",
                "feature_importance": {"feature_a": 1.0},
                "feature_stability": {"feature_a": 0.9},
                "feature_performance_by_regime": {"feature_a": {"HIGH_VOL": 1.0}},
            }
        ),
        encoding="utf-8",
    )


def _write_decision_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "action": "LONG",
        "evaluation_score": 0.5,
    }
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")


def test_feedback_loop_promotes_on_pass(tmp_path: Path, monkeypatch):
    audits_dir = tmp_path / "audits"
    decision_log = tmp_path / "decision_log.jsonl"
    stats_path = tmp_path / "feature_stats.json"
    policy_path = tmp_path / "policy.json"
    archive_dir = tmp_path / "archive"
    feedback_dir = tmp_path / "feedback"

    _write_audit(audits_dir)
    _write_stats(stats_path)
    _write_decision_log(decision_log)

    # existing policy to archive
    policy_path.write_text(json.dumps({"base_weights": {"old": 1}}), encoding="utf-8")

    cfg = FeedbackConfig(
        audit_dir=audits_dir,
        decision_log_path=decision_log,
        stats_path=stats_path,
        policy_path=policy_path,
        archive_dir=archive_dir,
        feedback_dir=feedback_dir,
    )

    monkeypatch.setattr(
        "engine.feedback_loop.evaluate_candidate_policy",
        lambda candidate, bt, gate_cfg: GatingDecision("PASS", [], {"pnl": 1}),
    )

    result = run_feedback_loop("pass_run", cfg)
    assert result.promoted is True
    assert policy_path.exists()
    # old policy archived
    archived = list(archive_dir.glob("*prev.json"))
    assert archived, "Expected archived previous policy"


def test_feedback_loop_rolls_back_on_fail(tmp_path: Path, monkeypatch):
    audits_dir = tmp_path / "audits"
    decision_log = tmp_path / "decision_log.jsonl"
    stats_path = tmp_path / "feature_stats.json"
    policy_path = tmp_path / "policy.json"
    archive_dir = tmp_path / "archive"
    feedback_dir = tmp_path / "feedback"

    _write_audit(audits_dir)
    _write_stats(stats_path)
    _write_decision_log(decision_log)

    old_policy = {"base_weights": {"old": 1}}
    policy_path.write_text(json.dumps(old_policy), encoding="utf-8")

    cfg = FeedbackConfig(
        audit_dir=audits_dir,
        decision_log_path=decision_log,
        stats_path=stats_path,
        policy_path=policy_path,
        archive_dir=archive_dir,
        feedback_dir=feedback_dir,
    )

    monkeypatch.setattr(
        "engine.feedback_loop.evaluate_candidate_policy",
        lambda candidate, bt, gate_cfg: GatingDecision("FAIL", ["bad"], {}),
    )

    result = run_feedback_loop("fail_run", cfg)
    assert result.promoted is False
    assert json.loads(policy_path.read_text(encoding="utf-8")) == old_policy


def test_feedback_loop_writes_feedback_summary(tmp_path: Path, monkeypatch):
    audits_dir = tmp_path / "audits"
    decision_log = tmp_path / "decision_log.jsonl"
    stats_path = tmp_path / "feature_stats.json"
    policy_path = tmp_path / "policy.json"
    archive_dir = tmp_path / "archive"
    feedback_dir = tmp_path / "feedback"

    _write_audit(audits_dir)
    _write_stats(stats_path)
    _write_decision_log(decision_log)

    cfg = FeedbackConfig(
        audit_dir=audits_dir,
        decision_log_path=decision_log,
        stats_path=stats_path,
        policy_path=policy_path,
        archive_dir=archive_dir,
        feedback_dir=feedback_dir,
    )

    monkeypatch.setattr(
        "engine.feedback_loop.evaluate_candidate_policy",
        lambda candidate, bt, gate_cfg: GatingDecision("PASS", [], {"pnl": 1}),
    )

    result = run_feedback_loop("summary_run", cfg)
    summary_path = feedback_dir / "feedback_run_summary_run.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary.get("gate_decision", {}).get("decision") in ("PASS", "FAIL")
    assert "drift_summary" in summary
    assert "stats_summary" in summary
    assert "decision_counts" in summary
    assert result.feedback_summary_path == summary_path
