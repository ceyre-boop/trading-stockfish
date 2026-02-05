import json
from pathlib import Path

import pandas as pd

from engine.tactical_archetypes import (
    build_archetype_features,
    build_tactical_archetypes,
    cluster_tactics,
    write_tactical_archetypes,
)

ENTRY_A = "ENTRY_SWEEP_DISPLACEMENT_REVERSAL"
ENTRY_B = "ENTRY_FVG_RESPECT_CONTINUATION"
ENTRY_C = "ENTRY_OB_CONTINUATION"

TIPS = {
    ENTRY_A: {
        "entry_model_id": ENTRY_A,
        "avg_prob_select": 0.6,
        "avg_expected_R": 1.8,
        "winrate": 0.55,
        "sample_size": 10,
        "risk_expected_R": 2.5,
        "risk_aggressiveness": "AGGRESSIVE",
    },
    ENTRY_B: {
        "entry_model_id": ENTRY_B,
        "avg_prob_select": 0.4,
        "avg_expected_R": 1.0,
        "winrate": 0.45,
        "sample_size": 8,
        "risk_expected_R": 1.2,
        "risk_aggressiveness": "NEUTRAL",
    },
    ENTRY_C: {
        "entry_model_id": ENTRY_C,
        "avg_prob_select": 0.5,
        "avg_expected_R": 0.6,
        "winrate": 0.5,
        "sample_size": 6,
        "risk_expected_R": 0.9,
        "risk_aggressiveness": "CONSERVATIVE",
    },
}

PRIORS = {
    ENTRY_A: {
        "base_prob_prior": 0.55,
        "base_expected_R_prior": 2.0,
        "confidence": 0.7,
        "risk_expected_R": 2.5,
        "risk_aggressiveness": "AGGRESSIVE",
    },
    ENTRY_B: {
        "base_prob_prior": 0.35,
        "base_expected_R_prior": 1.1,
        "confidence": 0.5,
        "risk_expected_R": 1.1,
        "risk_aggressiveness": "NEUTRAL",
    },
    ENTRY_C: {
        "base_prob_prior": 0.45,
        "base_expected_R_prior": 0.8,
        "confidence": 0.4,
        "risk_expected_R": 0.8,
        "risk_aggressiveness": "CONSERVATIVE",
    },
}

BELIEF_SHIFT = {
    ENTRY_A: {
        "delta_prob_select": 0.1,
        "delta_expected_R": -0.1,
        "delta_winrate": 0.02,
        "delta_confidence": 0.05,
        "structural_shift": True,
    },
    ENTRY_B: {
        "delta_prob_select": -0.05,
        "delta_expected_R": 0.1,
        "delta_winrate": -0.01,
        "delta_confidence": 0.0,
        "structural_shift": False,
    },
    ENTRY_C: {
        "delta_prob_select": 0.0,
        "delta_expected_R": 0.05,
        "delta_winrate": 0.0,
        "delta_confidence": -0.02,
        "structural_shift": False,
    },
}

DRIFT = {
    ENTRY_A: {
        "regret_mean": 0.8,
        "score_drift_mean": 0.2,
        "eligibility_drift_rate": 0.3,
        "stability": 0.55,
    },
    ENTRY_B: {
        "regret_mean": 0.3,
        "score_drift_mean": 0.1,
        "eligibility_drift_rate": 0.1,
        "stability": 0.75,
    },
    ENTRY_C: {
        "regret_mean": 0.5,
        "score_drift_mean": 0.15,
        "eligibility_drift_rate": 0.2,
        "stability": 0.65,
    },
}


EXPECTED_COLUMNS = [
    "entry_model_id",
    "avg_prob_select",
    "avg_expected_R",
    "winrate",
    "sample_size",
    "risk_expected_R",
    "risk_aggressiveness",
    "base_prob_prior",
    "base_expected_R_prior",
    "confidence",
    "delta_prob_select",
    "delta_expected_R",
    "delta_winrate",
    "delta_confidence",
    "structural_shift",
    "regret_mean",
    "score_drift_mean",
    "eligibility_drift_rate",
    "stability",
]


def test_feature_vector_construction():
    df = build_archetype_features(TIPS, PRIORS, BELIEF_SHIFT, DRIFT)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert df.shape[0] == 3


def test_clustering_deterministic():
    df = build_archetype_features(TIPS, PRIORS, BELIEF_SHIFT, DRIFT)
    map1 = cluster_tactics(df, k=2)
    map2 = cluster_tactics(df, k=2)
    assert map1 == map2


def test_archetype_labeling():
    df = build_archetype_features(TIPS, PRIORS, BELIEF_SHIFT, DRIFT)
    cluster_map = {
        ENTRY_A: "ARCHETYPE_0",
        ENTRY_B: "ARCHETYPE_1",
        ENTRY_C: "ARCHETYPE_1",
    }
    archetypes = build_tactical_archetypes(df, cluster_map)
    assert "ARCHETYPE_0" in archetypes
    assert "label" in archetypes["ARCHETYPE_0"]
    assert archetypes["ARCHETYPE_0"]["label"]


def test_archetype_artifact_sorted(tmp_path):
    df = build_archetype_features(TIPS, PRIORS, BELIEF_SHIFT, DRIFT)
    cluster_map = {
        ENTRY_A: "ARCHETYPE_0",
        ENTRY_B: "ARCHETYPE_1",
        ENTRY_C: "ARCHETYPE_1",
    }
    archetypes = build_tactical_archetypes(df, cluster_map)
    target = tmp_path / "tactical_archetypes.json"
    write_tactical_archetypes(archetypes, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    ids = [p["archetype_id"] for p in payload]
    assert ids == sorted(ids)
