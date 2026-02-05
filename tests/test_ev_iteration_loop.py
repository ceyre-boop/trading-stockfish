import json
from pathlib import Path

import pandas as pd

from engine.ev_iteration_loop import run_ev_iteration, validate_policy_stability


def _replay_config():
    decision_logs = pd.DataFrame(
        [
            {
                "decision_id": "d1",
                "timestamp_utc": "2026-02-04T12:00:00Z",
                "entry_model_id": "ENTRY_SWEEP_DISPLACEMENT_REVERSAL",
                "action": "LONG",
                "bar_index": 1,
                "decision_frame": {
                    "market_profile_state": "ACCUMULATION",
                    "vol_regime": "NORMAL",
                    "trend_regime": "UP",
                    "session_profile": "PROFILE_1A",
                    "session_profile_confidence": 0.5,
                    "market_profile_confidence": 0.6,
                    "liquidity_frame": {"bias": "UP"},
                    "condition_vector": {"session": "RTH"},
                },
                "entry_outcome": 0.3,
                "max_adverse_excursion": -0.1,
                "max_favorable_excursion": 0.8,
                "time_to_outcome": 3,
            },
            {
                "decision_id": "d2",
                "timestamp_utc": "2026-02-04T12:05:00Z",
                "entry_model_id": "ENTRY_SWEEP_DISPLACEMENT_REVERSAL",
                "action": "LONG",
                "bar_index": 2,
                "decision_frame": {
                    "market_profile_state": "ACCUMULATION",
                    "vol_regime": "NORMAL",
                    "trend_regime": "UP",
                    "session_profile": "PROFILE_1A",
                    "session_profile_confidence": 0.5,
                    "market_profile_confidence": 0.6,
                    "liquidity_frame": {"bias": "UP"},
                    "condition_vector": {"session": "RTH"},
                },
                "entry_outcome": 0.2,
                "max_adverse_excursion": -0.2,
                "max_favorable_excursion": 0.9,
                "time_to_outcome": 4,
            },
        ]
    )
    return {
        "decision_logs": decision_logs,
        "entry_selector_artifacts": None,
        "brain_policy_entries": None,
    }


def test_validate_policy_stability_rules():
    old = {"A": "ALLOWED", "B": "DISABLED"}
    new_stable = {"A": "ALLOWED", "B": "DISCOURAGED"}
    report = validate_policy_stability(old, new_stable)
    assert report["stable"] is True

    new_unstable = {"A": "PREFERRED", "B": "PREFERRED"}
    report2 = validate_policy_stability(old, new_unstable)
    assert report2["stable"] is False
    assert report2["severe_flip"] is True


def test_run_ev_iteration_writes_artifacts(tmp_path, monkeypatch):
    cfg = _replay_config()
    out_dir = tmp_path / "iter1"

    monkeypatch.setattr(
        "engine.adversarial_replay.score_entry_selector",
        lambda frame, eligible_models, artifacts: {
            eid: {"expected_R": 0.1} for eid in eligible_models
        },
    )

    summary = run_ev_iteration(cfg, min_samples=1, output_dir=out_dir)

    expected_files = [
        "dataset.parquet",
        "ev_brain_v1.pkl",
        "brain_policy_entries.learned.json",
        "validation.json",
    ]
    for fname in expected_files:
        assert (out_dir / fname).exists()

    with (out_dir / "validation.json").open("r", encoding="utf-8") as handle:
        validation = json.load(handle)
    assert "stable" in validation

    # Promotion only if stable; depending on thresholds it may promote. Accept either but path should exist if promoted.
    if summary.get("promoted"):
        assert (out_dir / "promoted_policy.json").exists()
