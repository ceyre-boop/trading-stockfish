import json
from pathlib import Path

import pandas as pd

from engine.tactical_drift import compute_tactical_drift, write_tactical_drift

ENTRY_A = "ENTRY_SWEEP_DISPLACEMENT_REVERSAL"
ENTRY_B = "ENTRY_FVG_RESPECT_CONTINUATION"

BELIEF_SHIFT = {
    ENTRY_A: {
        "entry_model_id": ENTRY_A,
        "delta_prob_select": 0.1,
        "delta_expected_R": -0.2,
        "delta_winrate": 0.05,
        "delta_confidence": -0.1,
        "structural_shift": True,
        "old_best_state": "DISTRIBUTION",
        "new_best_state": "ACCUMULATION",
    },
    ENTRY_B: {
        "entry_model_id": ENTRY_B,
        "delta_prob_select": -0.05,
        "delta_expected_R": 0.3,
        "delta_winrate": -0.02,
        "delta_confidence": 0.0,
        "structural_shift": False,
        "old_best_state": "MANIPULATION",
        "new_best_state": "MANIPULATION",
    },
}

REPLAY_ROWS = [
    {
        "entry_model_id": ENTRY_A,
        "regret": 1.0,
        "score_drift": 0.2,
        "eligibility_drift": True,
    },
    {
        "entry_model_id": ENTRY_A,
        "regret": 0.5,
        "score_drift": 0.1,
        "eligibility_drift": False,
    },
    {
        "entry_model_id": ENTRY_B,
        "regret": 0.2,
        "score_drift": 0.05,
        "eligibility_drift": False,
    },
]


def test_drift_computation_basic():
    replay_df = pd.DataFrame(REPLAY_ROWS)
    diags = compute_tactical_drift(BELIEF_SHIFT, replay_df)
    diag_a = diags[ENTRY_A]
    assert diag_a["prob_shift"] == 0.1
    assert diag_a["expected_R_shift"] == -0.2
    assert diag_a["winrate_shift"] == 0.05
    assert abs(diag_a["regret_mean"] - 0.75) < 1e-6
    assert abs(diag_a["score_drift_mean"] - 0.15) < 1e-6
    assert abs(diag_a["eligibility_drift_rate"] - 0.5) < 1e-6


def test_structural_shift_propagation():
    replay_df = pd.DataFrame(REPLAY_ROWS)
    diags = compute_tactical_drift(BELIEF_SHIFT, replay_df)
    assert diags[ENTRY_A]["structural_shift"] is True
    assert diags[ENTRY_A]["old_best_state"] == "DISTRIBUTION"
    assert diags[ENTRY_A]["new_best_state"] == "ACCUMULATION"
    assert diags[ENTRY_B]["structural_shift"] is False


def test_stability_score_bounds():
    replay_df = pd.DataFrame(REPLAY_ROWS)
    diags = compute_tactical_drift(BELIEF_SHIFT, replay_df)
    for d in diags.values():
        assert 0.0 <= d["stability"] <= 1.0


def test_commentary_deterministic():
    replay_df = pd.DataFrame(REPLAY_ROWS)
    d1 = compute_tactical_drift(BELIEF_SHIFT, replay_df)
    d2 = compute_tactical_drift(BELIEF_SHIFT, replay_df)
    assert d1[ENTRY_A]["commentary"] == d2[ENTRY_A]["commentary"]


def test_artifact_sorted(tmp_path):
    replay_df = pd.DataFrame(REPLAY_ROWS)
    diags = compute_tactical_drift(BELIEF_SHIFT, replay_df)
    target = tmp_path / "tactical_drift.json"
    write_tactical_drift(diags, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    ids = [p["entry_model_id"] for p in payload]
    assert ids == sorted(ids)
