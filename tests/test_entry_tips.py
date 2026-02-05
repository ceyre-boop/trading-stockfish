import json
from pathlib import Path

import pandas as pd

from engine.entry_models import ENTRY_MODELS
from engine.entry_tips import build_tactical_intuition_profiles, write_entry_tips

ENTRY_A = "ENTRY_SWEEP_DISPLACEMENT_REVERSAL"
ENTRY_B = "ENTRY_FVG_RESPECT_CONTINUATION"


BELIEF_MAP = [
    {
        "entry_model_id": ENTRY_A,
        "avg_prob_select": 0.6,
        "avg_expected_R": 1.8,
        "empirical_success_rate": 0.55,
        "count": 10,
    },
    {
        "entry_model_id": ENTRY_B,
        "avg_prob_select": 0.4,
        "avg_expected_R": 1.2,
        "empirical_success_rate": 0.45,
        "count": 8,
    },
]


REPLAY_ROWS = [
    {
        "entry_model_id": ENTRY_A,
        "expected_R": 2.0,
        "entry_success": 1,
        "market_profile_state": "MANIPULATION",
        "session_profile": "PROFILE_1A",
    },
    {
        "entry_model_id": ENTRY_A,
        "expected_R": -0.5,
        "entry_success": 0,
        "market_profile_state": "ACCUMULATION",
        "session_profile": "PROFILE_1B",
    },
    {
        "entry_model_id": ENTRY_B,
        "expected_R": 1.5,
        "entry_success": 1,
        "market_profile_state": "DISTRIBUTION",
        "session_profile": "PROFILE_1A",
    },
    {
        "entry_model_id": ENTRY_B,
        "expected_R": -1.0,
        "entry_success": 0,
        "market_profile_state": "ACCUMULATION",
        "session_profile": "PROFILE_1C",
    },
]


def test_tip_builder_produces_profiles():
    replay_df = pd.DataFrame(REPLAY_ROWS)
    tips = build_tactical_intuition_profiles(BELIEF_MAP, replay_df, ENTRY_MODELS)
    assert ENTRY_A in tips
    assert ENTRY_B in tips
    for tip in tips.values():
        for key in [
            "avg_prob_select",
            "avg_expected_R",
            "winrate",
            "sample_size",
            "best_market_profile_state",
            "worst_market_profile_state",
            "risk_expected_R",
            "confidence_score",
            "commentary",
        ]:
            assert key in tip


def test_tip_structural_preferences_correct():
    replay_df = pd.DataFrame(REPLAY_ROWS)
    tips = build_tactical_intuition_profiles(BELIEF_MAP, replay_df, ENTRY_MODELS)
    tip_a = tips[ENTRY_A]
    assert tip_a["best_market_profile_state"] == "MANIPULATION"
    assert tip_a["worst_market_profile_state"] == "ACCUMULATION"
    assert tip_a["best_session_profile"] == "PROFILE_1A"
    assert tip_a["worst_session_profile"] == "PROFILE_1B"


def test_tip_risk_profile_included():
    replay_df = pd.DataFrame(REPLAY_ROWS)
    tips = build_tactical_intuition_profiles(BELIEF_MAP, replay_df, ENTRY_MODELS)
    risk = ENTRY_MODELS[ENTRY_A].risk_profile
    tip = tips[ENTRY_A]
    assert tip["risk_expected_R"] == risk.get("expected_R")
    assert tip["risk_MAE_bucket"] == risk.get("mae_bucket")
    assert tip["risk_MFE_bucket"] == risk.get("mfe_bucket")
    assert tip["risk_time_horizon"] == risk.get("time_horizon")
    assert tip["risk_aggressiveness"] == risk.get("aggressiveness")


def test_tip_commentary_deterministic():
    replay_df = pd.DataFrame(REPLAY_ROWS)
    tips1 = build_tactical_intuition_profiles(BELIEF_MAP, replay_df, ENTRY_MODELS)
    tips2 = build_tactical_intuition_profiles(BELIEF_MAP, replay_df, ENTRY_MODELS)
    assert tips1[ENTRY_A]["commentary"] == tips2[ENTRY_A]["commentary"]


def test_tip_artifact_written_sorted(tmp_path):
    replay_df = pd.DataFrame(REPLAY_ROWS)
    tips = build_tactical_intuition_profiles(BELIEF_MAP, replay_df, ENTRY_MODELS)
    target = tmp_path / "entry_tips.json"
    write_entry_tips(tips, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    ids = [item["entry_model_id"] for item in payload]
    assert ids == sorted(ids)
