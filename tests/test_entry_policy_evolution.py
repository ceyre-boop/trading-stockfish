import pandas as pd

from engine.entry_policy_diff import diff_entry_policies
from engine.entry_policy_evolution import propose_entry_policy_updates
from engine.entry_policy_report import build_entry_policy_report

_DEFAULT_THRESHOLDS = {
    "winrate_min": 0.4,
    "expected_R_promote": 0.5,
    "winrate_promote": 0.6,
    "min_samples": 3,
    "min_samples_promote": 4,
    "min_samples_stable": 2,
}


def _stress_row(
    entry_id: str,
    winrate: float,
    expected_r: float,
    regret: float,
    sample: int,
    mp: str = "ACCUMULATION",
):
    return {
        "entry_model_id": entry_id,
        "market_profile_state": mp,
        "session_profile": "PROFILE_1A",
        "liquidity_bias_side": "UP",
        "winrate": winrate,
        "mean_expected_R": expected_r,
        "mean_regret": regret,
        "sample_size": sample,
    }


def test_downgrade_on_poor_performance():
    stress_df = pd.DataFrame(
        [
            _stress_row(
                "ENTRY_FVG_RESPECT_CONTINUATION",
                winrate=0.2,
                expected_r=-0.3,
                regret=1.0,
                sample=5,
            )
        ]
    )
    drift_df = pd.DataFrame(
        [
            {
                "entry_model_id": "ENTRY_FVG_RESPECT_CONTINUATION",
                "drift_flags": {"regret": True},
            }
        ]
    )
    current_policy = pd.DataFrame(
        [
            {
                "entry_model_id": "ENTRY_FVG_RESPECT_CONTINUATION",
                "market_profile_state": "ACCUMULATION",
                "session_profile": "PROFILE_1A",
                "liquidity_bias_side": "UP",
                "label": "PREFERRED",
            }
        ]
    )

    proposed = propose_entry_policy_updates(
        stress_df, drift_df, None, current_policy, _DEFAULT_THRESHOLDS
    )
    assert proposed.iloc[0]["proposed_label"] == "ALLOWED"
    assert proposed.iloc[0]["reason"] == "drift_detected"


def test_upgrade_on_strong_performance():
    stress_df = pd.DataFrame(
        [
            _stress_row(
                "ENTRY_OB_CONTINUATION",
                winrate=0.8,
                expected_r=0.8,
                regret=0.1,
                sample=6,
            )
        ]
    )
    drift_df = pd.DataFrame(
        [
            {
                "entry_model_id": "ENTRY_OB_CONTINUATION",
                "drift_flags": {"regret": False, "winrate": False},
            }
        ]
    )
    current_policy = pd.DataFrame(
        [
            {
                "entry_model_id": "ENTRY_OB_CONTINUATION",
                "market_profile_state": "ACCUMULATION",
                "session_profile": "PROFILE_1A",
                "liquidity_bias_side": "UP",
                "label": "ALLOWED",
            }
        ]
    )

    proposed = propose_entry_policy_updates(
        stress_df, drift_df, None, current_policy, _DEFAULT_THRESHOLDS
    )
    assert proposed.iloc[0]["proposed_label"] == "PREFERRED"
    assert proposed.iloc[0]["reason"] == "high_expected_R"


def test_no_change_on_low_sample():
    stress_df = pd.DataFrame(
        [
            _stress_row(
                "ENTRY_OB_CONTINUATION",
                winrate=0.1,
                expected_r=-0.2,
                regret=1.0,
                sample=1,
            )
        ]
    )
    drift_df = pd.DataFrame([])
    current_policy = pd.DataFrame(
        [
            {
                "entry_model_id": "ENTRY_OB_CONTINUATION",
                "market_profile_state": "ACCUMULATION",
                "session_profile": "PROFILE_1A",
                "liquidity_bias_side": "UP",
                "label": "DISCOURAGED",
            }
        ]
    )

    proposed = propose_entry_policy_updates(
        stress_df, drift_df, None, current_policy, _DEFAULT_THRESHOLDS
    )
    assert proposed.iloc[0]["proposed_label"] == "DISCOURAGED"
    assert proposed.iloc[0]["reason"] == "low_sample"


def test_policy_diff_detects_changes():
    current = pd.DataFrame(
        [
            {
                "entry_model_id": "ENTRY_OB_CONTINUATION",
                "market_profile_state": "ACCUMULATION",
                "session_profile": "PROFILE_1A",
                "liquidity_bias_side": "UP",
                "label": "ALLOWED",
            }
        ]
    )
    proposed = pd.DataFrame(
        [
            {
                "entry_model_id": "ENTRY_OB_CONTINUATION",
                "market_profile_state": "ACCUMULATION",
                "session_profile": "PROFILE_1A",
                "liquidity_bias_side": "UP",
                "proposed_label": "PREFERRED",
                "reason": "high_expected_R",
            }
        ]
    )

    diff = diff_entry_policies(current, proposed)
    assert len(diff) == 1
    row = diff.iloc[0]
    assert row["from_label"] == "ALLOWED"
    assert row["to_label"] == "PREFERRED"
    assert row["reason"] == "high_expected_R"


def test_policy_report_summarizes_changes(tmp_path, monkeypatch):
    # Ensure report writes to temp dir
    monkeypatch.setattr("engine.entry_policy_report._ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(
        "engine.entry_policy_report._REPORT_PATH",
        tmp_path / "brain_policy_entries.report.json",
    )

    diff_df = pd.DataFrame(
        [
            {
                "entry_model_id": "ENTRY_OB_CONTINUATION",
                "market_profile_state": "ACCUMULATION",
                "session_profile": "PROFILE_1A",
                "liquidity_bias_side": "UP",
                "from_label": "ALLOWED",
                "to_label": "PREFERRED",
                "reason": "high_expected_R",
                "winrate": 0.8,
                "mean_expected_R": 0.7,
                "mean_regret": 0.1,
                "sample_size": 10,
            }
        ]
    )
    perf_df = pd.DataFrame([])

    report = build_entry_policy_report(diff_df, perf_df)
    assert report["summary"]["num_upgrades"] == 1
    assert report["by_entry_model"][0]["entry_model_id"] == "ENTRY_OB_CONTINUATION"
    change = report["by_entry_model"][0]["changes"][0]
    assert change["reason"] == "high_expected_R"
    assert change["metrics"]["winrate"] == 0.8


def test_determinism_of_proposals():
    stress_df = pd.DataFrame(
        [
            _stress_row(
                "ENTRY_FVG_RESPECT_CONTINUATION",
                winrate=0.2,
                expected_r=-0.3,
                regret=1.0,
                sample=5,
            ),
            _stress_row(
                "ENTRY_OB_CONTINUATION",
                winrate=0.8,
                expected_r=0.8,
                regret=0.1,
                sample=6,
            ),
        ]
    )
    drift_df = pd.DataFrame(
        [
            {
                "entry_model_id": "ENTRY_FVG_RESPECT_CONTINUATION",
                "drift_flags": {"regret": True},
            },
            {
                "entry_model_id": "ENTRY_OB_CONTINUATION",
                "drift_flags": {"regret": False},
            },
        ]
    )
    current_policy = pd.DataFrame(
        [
            {
                "entry_model_id": "ENTRY_FVG_RESPECT_CONTINUATION",
                "market_profile_state": "ACCUMULATION",
                "session_profile": "PROFILE_1A",
                "liquidity_bias_side": "UP",
                "label": "PREFERRED",
            },
            {
                "entry_model_id": "ENTRY_OB_CONTINUATION",
                "market_profile_state": "ACCUMULATION",
                "session_profile": "PROFILE_1A",
                "liquidity_bias_side": "UP",
                "label": "ALLOWED",
            },
        ]
    )

    proposed_a = propose_entry_policy_updates(
        stress_df, drift_df, None, current_policy, _DEFAULT_THRESHOLDS
    )
    proposed_b = propose_entry_policy_updates(
        stress_df, drift_df, None, current_policy, _DEFAULT_THRESHOLDS
    )

    pd.testing.assert_frame_equal(proposed_a, proposed_b)
