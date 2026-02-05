import json
from pathlib import Path

import pandas as pd

from engine.archetype_performance import compute_archetype_performance, write_archetype_performance

ARCH_A = "ARCHETYPE_0"
ARCH_B = "ARCHETYPE_1"
ENTRY_A = "ENTRY_SWEEP_DISPLACEMENT_REVERSAL"
ENTRY_B = "ENTRY_FVG_RESPECT_CONTINUATION"
ENTRY_C = "ENTRY_OB_CONTINUATION"

ARCHETYPES = {
    ARCH_A: {
        "archetype_id": ARCH_A,
        "members": [ENTRY_A, ENTRY_B],
        "label": "Stable High-Reward",
    },
    ARCH_B: {
        "archetype_id": ARCH_B,
        "members": [ENTRY_C],
        "label": "Unstable Low-Reward",
    },
}

REPLAY_ROWS = [
    {
        "entry_model_id": ENTRY_A,
        "expected_R": 1.5,
        "entry_success": 1,
        "market_profile_state": "DISTRIBUTION",
        "session_profile": "PROFILE_1A",
        "vol_regime": "HIGH",
        "trend_regime": "UP",
    },
    {
        "entry_model_id": ENTRY_B,
        "expected_R": 0.5,
        "entry_success": 0,
        "market_profile_state": "ACCUMULATION",
        "session_profile": "PROFILE_1B",
        "vol_regime": "NORMAL",
        "trend_regime": "FLAT",
    },
    {
        "entry_model_id": ENTRY_A,
        "expected_R": 2.0,
        "entry_success": 1,
        "market_profile_state": "DISTRIBUTION",
        "session_profile": "PROFILE_1A",
        "vol_regime": "HIGH",
        "trend_regime": "UP",
    },
]

STABILITY = {
    ENTRY_A: {"stability": 0.8},
    ENTRY_B: {"stability": 0.6},
    ENTRY_C: {"stability": 0.4},
}


def test_archetype_performance_basic():
    df = pd.DataFrame(REPLAY_ROWS)
    perf = compute_archetype_performance(ARCHETYPES, df)
    a = perf[ARCH_A]
    assert abs(a["mean_expected_R"] - 1.3333) < 1e-3
    assert a["sample_size"] == 3
    assert a["winrate"] == (2 / 3)


def test_structural_grouping():
    df = pd.DataFrame(REPLAY_ROWS)
    perf = compute_archetype_performance(ARCHETYPES, df)
    dist = perf[ARCH_A]["by_market_profile_state"].get("DISTRIBUTION")
    assert dist["samples"] == 2
    assert abs(dist["expected_R"] - 1.75) < 1e-6


def test_stability_integration():
    df = pd.DataFrame(REPLAY_ROWS)
    perf = compute_archetype_performance(ARCHETYPES, df, STABILITY)
    a = perf[ARCH_A]
    expected_avg = (STABILITY[ENTRY_A]["stability"] + STABILITY[ENTRY_B]["stability"]) / 2
    assert abs(a["avg_stability"] - expected_avg) < 1e-6


def test_commentary_deterministic():
    df = pd.DataFrame(REPLAY_ROWS)
    p1 = compute_archetype_performance(ARCHETYPES, df, STABILITY)
    p2 = compute_archetype_performance(ARCHETYPES, df, STABILITY)
    assert p1[ARCH_A]["commentary"] == p2[ARCH_A]["commentary"]


def test_artifact_sorted(tmp_path):
    df = pd.DataFrame(REPLAY_ROWS)
    perf = compute_archetype_performance(ARCHETYPES, df, STABILITY)
    target = tmp_path / "archetype_performance.json"
    write_archetype_performance(perf, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    ids = [p["archetype_id"] for p in payload]
    assert ids == sorted(ids)
