import json
from pathlib import Path

from engine.archetype_evolution import (
    compute_archetype_evolution,
    write_archetype_evolution,
)

ARCH_A = "ARCHETYPE_0"
ARCH_B = "ARCHETYPE_1"
ENTRY_A = "ENTRY_SWEEP_DISPLACEMENT_REVERSAL"
ENTRY_B = "ENTRY_FVG_RESPECT_CONTINUATION"

OLD_PERF = {
    ARCH_A: {
        "archetype_id": ARCH_A,
        "mean_expected_R": 1.0,
        "winrate": 0.5,
        "sample_size": 10,
        "by_market_profile_state": {
            "DISTRIBUTION": {"expected_R": 1.0},
            "ACCUMULATION": {"expected_R": 0.5},
        },
        "avg_stability": 0.6,
        "members": [ENTRY_A, ENTRY_B],
    },
    ARCH_B: {
        "archetype_id": ARCH_B,
        "mean_expected_R": 0.5,
        "winrate": 0.4,
        "sample_size": 5,
        "by_market_profile_state": {"DISTRIBUTION": {"expected_R": 0.4}},
        "avg_stability": 0.5,
        "members": [ENTRY_B],
    },
}

NEW_PERF = {
    ARCH_A: {
        "archetype_id": ARCH_A,
        "mean_expected_R": 1.3,
        "winrate": 0.55,
        "sample_size": 12,
        "by_market_profile_state": {
            "ACCUMULATION": {"expected_R": 1.2},
            "DISTRIBUTION": {"expected_R": 1.1},
        },
        "avg_stability": 0.7,
        "members": [ENTRY_A, ENTRY_B],
    },
    ARCH_B: {
        "archetype_id": ARCH_B,
        "mean_expected_R": 0.4,
        "winrate": 0.35,
        "sample_size": 4,
        "by_market_profile_state": {"DISTRIBUTION": {"expected_R": 0.3}},
        "avg_stability": 0.45,
        "members": [ENTRY_B],
    },
}

STABILITY = {
    ENTRY_A: {"stability": 0.8},
    ENTRY_B: {"stability": 0.6},
}

BELIEF_SHIFT = {
    ENTRY_A: {"delta_expected_R": 0.2},
    ENTRY_B: {"delta_expected_R": -0.1},
}


def test_evolution_deltas():
    signals = compute_archetype_evolution(OLD_PERF, NEW_PERF)
    a = signals[ARCH_A]
    assert abs(a["delta_expected_R"] - 0.3) < 1e-9
    assert abs(a["delta_winrate"] - 0.05) < 1e-9
    assert a["delta_sample_size"] == 2


def test_structural_shift_detection():
    signals = compute_archetype_evolution(OLD_PERF, NEW_PERF)
    assert signals[ARCH_A]["structural_shift"] is True
    assert signals[ARCH_A]["old_best_state"] == "DISTRIBUTION"
    assert signals[ARCH_A]["new_best_state"] == "ACCUMULATION"


def test_stability_evolution():
    signals = compute_archetype_evolution(OLD_PERF, NEW_PERF)
    a = signals[ARCH_A]
    assert abs(a["delta_stability"] - 0.1) < 1e-9


def test_member_drift_aggregation():
    signals = compute_archetype_evolution(OLD_PERF, NEW_PERF, STABILITY, BELIEF_SHIFT)
    a = signals[ARCH_A]
    expected_avg = (
        STABILITY[ENTRY_A]["stability"] + STABILITY[ENTRY_B]["stability"]
    ) / 2
    assert abs(a["avg_member_drift"] - expected_avg) < 1e-9


def test_commentary_deterministic():
    s1 = compute_archetype_evolution(OLD_PERF, NEW_PERF, STABILITY, BELIEF_SHIFT)
    s2 = compute_archetype_evolution(OLD_PERF, NEW_PERF, STABILITY, BELIEF_SHIFT)
    assert s1[ARCH_A]["commentary"] == s2[ARCH_A]["commentary"]


def test_artifact_sorted(tmp_path):
    signals = compute_archetype_evolution(OLD_PERF, NEW_PERF, STABILITY, BELIEF_SHIFT)
    target = tmp_path / "archetype_evolution.json"
    write_archetype_evolution(signals, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    ids = [p["archetype_id"] for p in payload]
    assert ids == sorted(ids)
