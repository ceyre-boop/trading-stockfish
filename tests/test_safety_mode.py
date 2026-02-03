import json
from pathlib import Path

import pytest

from engine.feedback_loop import FeedbackConfig, run_feedback_loop
from engine.policy_gating import GatingDecision
from engine.regime_multipliers import StatsResult
from engine.safety_mode import (
    SafetyConfig,
    SafetyDecision,
    SafetyState,
    apply_safety_decision,
    check_and_update_safety,
)


def test_enters_safe_mode_on_drift():
    drift_result = {"aggregates": {"features_flagged": 3}}
    decision = check_and_update_safety(
        drift_result,
        gating_history=["PASS"],
        config=SafetyConfig(drift_feature_threshold=1),
    )
    assert decision.new_state == "SAFE_MODE"
    assert "drift" in decision.reason


def test_reverts_to_last_good_policy():
    policy = {"base_weights": {"f": 1.0}, "regime_multipliers": {}}
    fallback = {"base_weights": {"f": 0.5}, "regime_multipliers": {}}
    decision = SafetyDecision(
        new_state="SAFE_MODE", action="REVERT_LAST_GOOD", fallback_policy=fallback
    )
    safe_policy = apply_safety_decision(policy, decision)
    assert safe_policy["base_weights"] == fallback["base_weights"]
    assert safe_policy.get("safety_mode", {}).get("state") == "SAFE_MODE"


def test_downweights_multipliers_in_safe_mode():
    policy = {"regime_multipliers": {"HIGH_VOL": {"f": 2.0}}}
    decision = SafetyDecision(
        new_state="SAFE_MODE", action="DAMPEN_MULTIPLIERS", multiplier_scale=0.25
    )
    safe_policy = apply_safety_decision(policy, decision)
    assert safe_policy["regime_multipliers"]["HIGH_VOL"]["f"] == pytest.approx(0.5)
    meta = safe_policy.get("safety_mode", {})
    assert meta.get("state") == "SAFE_MODE"
    assert meta.get("multiplier_scale") == pytest.approx(0.25)


def test_exits_safe_mode_when_stable():
    drift_result = {"aggregates": {"features_flagged": 0}}
    decision = check_and_update_safety(
        drift_result,
        gating_history=["PASS", "PASS"],
        config=SafetyConfig(drift_feature_threshold=1, gate_fail_threshold=2),
        state=SafetyState(mode="SAFE_MODE"),
    )
    assert decision.new_state == "NORMAL"
    assert "stable" in decision.reason


def test_feedback_loop_respects_safety_mode(tmp_path: Path, monkeypatch):
    audits_dir = tmp_path / "audits"
    decision_log = tmp_path / "decision_log.jsonl"
    stats_path = tmp_path / "feature_stats.json"
    policy_path = tmp_path / "policy.json"
    archive_dir = tmp_path / "archive"
    feedback_dir = tmp_path / "feedback"

    last_good = {"version": "0.1", "base_weights": {"old": 1}}
    policy_path.write_text(json.dumps(last_good), encoding="utf-8")

    # Drift detection returns one flagged feature -> triggers SAFE_MODE
    monkeypatch.setattr(
        "engine.feedback_loop.detect_drifts", lambda **kwargs: {"feat": ["drift"]}
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

    strong_bt = lambda policy: None  # unused because gating is mocked
    cfg = FeedbackConfig(
        audit_dir=audits_dir,
        decision_log_path=decision_log,
        stats_path=stats_path,
        policy_path=policy_path,
        archive_dir=archive_dir,
        feedback_dir=feedback_dir,
        backtest_runner=strong_bt,
    )

    # Gating passes, but safety should override
    monkeypatch.setattr(
        "engine.feedback_loop.evaluate_candidate_policy",
        lambda candidate, bt, gate_cfg: GatingDecision("PASS", [], {"pnl": 1.0}),
    )

    result = run_feedback_loop("safe_override", cfg)

    # Candidate not promoted; last-good remains
    assert result.promoted is False
    final_policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert final_policy["base_weights"] == last_good["base_weights"]
    assert final_policy.get("safety_mode", {}).get("state") == "SAFE_MODE"

    summary = json.loads(
        (feedback_dir / "feedback_run_safe_override.json").read_text(encoding="utf-8")
    )
    assert summary["gating"]["decision"] == "PASS"
    assert summary["safety"]["new_state"] == "SAFE_MODE"
    assert summary["safety"]["applied"] in {"reverted", "dampened"}
