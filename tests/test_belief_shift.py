import json
from pathlib import Path

from engine.belief_shift import compute_belief_shift, write_belief_shift

ENTRY_A = "ENTRY_SWEEP_DISPLACEMENT_REVERSAL"
ENTRY_B = "ENTRY_FVG_RESPECT_CONTINUATION"


OLD_MAP = [
    {
        "entry_model_id": ENTRY_A,
        "avg_prob_select": 0.4,
        "avg_expected_R": 1.0,
        "empirical_success_rate": 0.45,
        "count": 10,
        "best_market_profile_state": "DISTRIBUTION",
    },
    {
        "entry_model_id": ENTRY_B,
        "avg_prob_select": 0.5,
        "avg_expected_R": 1.5,
        "empirical_success_rate": 0.55,
        "count": 12,
        "best_market_profile_state": "ACCUMULATION",
    },
]

NEW_MAP = [
    {
        "entry_model_id": ENTRY_A,
        "avg_prob_select": 0.6,
        "avg_expected_R": 1.4,
        "empirical_success_rate": 0.5,
        "count": 20,
        "best_market_profile_state": "MANIPULATION",
    },
    {
        "entry_model_id": ENTRY_B,
        "avg_prob_select": 0.45,
        "avg_expected_R": 1.2,
        "empirical_success_rate": 0.5,
        "count": 8,
        "best_market_profile_state": "ACCUMULATION",
    },
]

PRIORS = {
    ENTRY_A: {"risk_expected_R": 1.3},
    ENTRY_B: {"risk_expected_R": 1.0},
}


def test_belief_shift_computation():
    shifts = compute_belief_shift(OLD_MAP, NEW_MAP)
    shift_a = shifts[ENTRY_A]
    assert shift_a["delta_prob_select"] == 0.2
    assert shift_a["delta_expected_R"] == 0.4
    assert shift_a["delta_winrate"] == 0.05


def test_structural_shift_detection():
    shifts = compute_belief_shift(OLD_MAP, NEW_MAP)
    assert shifts[ENTRY_A]["structural_shift"] is True
    assert shifts[ENTRY_A]["old_best_state"] == "DISTRIBUTION"
    assert shifts[ENTRY_A]["new_best_state"] == "MANIPULATION"


def test_prior_alignment():
    shifts = compute_belief_shift(OLD_MAP, NEW_MAP, PRIORS)
    assert shifts[ENTRY_A]["prior_expected_R"] == PRIORS[ENTRY_A]["risk_expected_R"]
    assert shifts[ENTRY_A]["prior_alignment"] == abs(NEW_MAP[0]["avg_expected_R"] - PRIORS[ENTRY_A]["risk_expected_R"])


def test_commentary_deterministic():
    s1 = compute_belief_shift(OLD_MAP, NEW_MAP, PRIORS)
    s2 = compute_belief_shift(OLD_MAP, NEW_MAP, PRIORS)
    assert s1[ENTRY_A]["commentary"] == s2[ENTRY_A]["commentary"]


def test_artifact_sorted(tmp_path):
    shifts = compute_belief_shift(OLD_MAP, NEW_MAP, PRIORS)
    target = tmp_path / "belief_shift.json"
    write_belief_shift(shifts, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    ids = [p["entry_model_id"] for p in payload]
    assert ids == sorted(ids)
