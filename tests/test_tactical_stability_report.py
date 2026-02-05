import json
from pathlib import Path

from engine.tactical_stability_report import (
    build_tactical_stability_report,
    write_tactical_stability_report,
)

ENTRY_A = "ENTRY_SWEEP_DISPLACEMENT_REVERSAL"
ENTRY_B = "ENTRY_FVG_RESPECT_CONTINUATION"

TIPS = {
    ENTRY_A: {
        "entry_model_id": ENTRY_A,
        "avg_prob_select": 0.6,
        "avg_expected_R": 1.8,
        "winrate": 0.55,
        "sample_size": 10,
        "best_market_profile_state": "MANIPULATION",
        "worst_market_profile_state": "ACCUMULATION",
        "risk_expected_R": 2.5,
        "risk_aggressiveness": "AGGRESSIVE",
    },
    ENTRY_B: {
        "entry_model_id": ENTRY_B,
        "avg_prob_select": 0.4,
        "avg_expected_R": 1.2,
        "winrate": 0.45,
        "sample_size": 8,
        "best_market_profile_state": "DISTRIBUTION",
        "worst_market_profile_state": "PROFILE_1C",
        "risk_expected_R": 1.5,
        "risk_aggressiveness": "NEUTRAL",
    },
}

PRIORS = {
    ENTRY_A: {"risk_expected_R": 2.4, "risk_aggressiveness": "AGGRESSIVE"},
    ENTRY_B: {"risk_expected_R": 1.0, "risk_aggressiveness": "NEUTRAL"},
}

BELIEF_SHIFT = {
    ENTRY_A: {
        "entry_model_id": ENTRY_A,
        "delta_prob_select": 0.1,
        "delta_expected_R": -0.2,
        "delta_winrate": 0.05,
        "structural_shift": True,
    },
    ENTRY_B: {
        "entry_model_id": ENTRY_B,
        "delta_prob_select": -0.05,
        "delta_expected_R": 0.1,
        "delta_winrate": -0.02,
        "structural_shift": False,
    },
}

DRIFT = {
    ENTRY_A: {
        "entry_model_id": ENTRY_A,
        "regret_mean": 0.8,
        "score_drift_mean": 0.2,
        "eligibility_drift_rate": 0.3,
        "stability": 0.6,
    },
    ENTRY_B: {
        "entry_model_id": ENTRY_B,
        "regret_mean": 0.3,
        "score_drift_mean": 0.1,
        "eligibility_drift_rate": 0.1,
        "stability": 0.8,
    },
}


def test_tsr_combines_all_sources():
    report = build_tactical_stability_report(TIPS, PRIORS, BELIEF_SHIFT, DRIFT)
    r = report[ENTRY_A]
    assert r["avg_prob_select"] == TIPS[ENTRY_A]["avg_prob_select"]
    assert r["delta_expected_R"] == BELIEF_SHIFT[ENTRY_A]["delta_expected_R"]
    assert r["stability"] == DRIFT[ENTRY_A]["stability"]
    assert r["risk_expected_R"] == TIPS[ENTRY_A]["risk_expected_R"]


def test_prior_alignment_correct():
    report = build_tactical_stability_report(TIPS, PRIORS, BELIEF_SHIFT, DRIFT)
    r = report[ENTRY_A]
    expected_alignment = abs(
        TIPS[ENTRY_A]["avg_expected_R"] - PRIORS[ENTRY_A]["risk_expected_R"]
    )
    assert r["prior_alignment"] == expected_alignment


def test_commentary_deterministic():
    r1 = build_tactical_stability_report(TIPS, PRIORS, BELIEF_SHIFT, DRIFT)
    r2 = build_tactical_stability_report(TIPS, PRIORS, BELIEF_SHIFT, DRIFT)
    assert r1[ENTRY_A]["commentary"] == r2[ENTRY_A]["commentary"]


def test_artifact_sorted(tmp_path):
    report = build_tactical_stability_report(TIPS, PRIORS, BELIEF_SHIFT, DRIFT)
    target = tmp_path / "tactical_stability_report.json"
    write_tactical_stability_report(report, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    ids = [p["entry_model_id"] for p in payload]
    assert ids == sorted(ids)
