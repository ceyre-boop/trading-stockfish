import json
from pathlib import Path

import pandas as pd

from engine.adversarial_replay import adversarial_replay


class DummyArtifacts:
    pass


def test_candidate_and_outcome_logging(tmp_path, monkeypatch):
    # Synthetic decision log with one decision
    decision_logs = pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-02-04T12:00:00Z",
                "entry_model_id": "ENTRY_FVG_RESPECT_CONTINUATION",
                "decision_frame": {
                    "market_profile_state": "DISTRIBUTION",
                    "session_profile": "PROFILE_1C",
                    "liquidity_frame": {
                        "bias": "UP",
                        "sweep_state": "POST_SWEEP",
                        "distance_bucket": "NEAR",
                    },
                    "entry_signals_present": {
                        "fvg": True,
                        "sweep": True,
                        "displacement": True,
                    },
                    "condition_vector": {"vol": "NORMAL", "trend": "UP"},
                },
                "entry_outcome": 1.2,
                "max_adverse_excursion": -0.4,
                "max_favorable_excursion": 2.5,
                "time_to_outcome": 5,
                "fill_price": 100.0,
                "exit_price": 101.2,
                "bar_index": 10,
            }
        ]
    )

    # Policy entries enable only a subset
    brain_policy_entries = pd.DataFrame(
        [
            {"entry_model_id": "ENTRY_FVG_RESPECT_CONTINUATION", "label": "ALLOWED"},
            {
                "entry_model_id": "ENTRY_SWEEP_DISPLACEMENT_REVERSAL",
                "label": "PREFERRED",
            },
            {"entry_model_id": "ENTRY_OB_CONTINUATION", "label": "ALLOWED"},
        ]
    )

    def fake_get_eligible(frame):
        return ["ENTRY_FVG_RESPECT_CONTINUATION", "ENTRY_SWEEP_DISPLACEMENT_REVERSAL"]

    def fake_score_selector(frame, eligible_ids, artifacts):
        return {
            eid: {"expected_R": 1.0 if eid.endswith("CONTINUATION") else 2.0}
            for eid in eligible_ids
        }

    monkeypatch.setattr(
        "engine.adversarial_replay.get_eligible_entry_models",
        lambda frame: fake_get_eligible(frame),
    )
    monkeypatch.setattr(
        "engine.adversarial_replay.score_entry_selector", fake_score_selector
    )

    candidate_path = tmp_path / "entry_candidates.parquet"
    outcome_path = tmp_path / "entry_outcomes.parquet"

    df = adversarial_replay(
        decision_logs,
        DummyArtifacts(),
        brain_policy_entries,
        candidate_path=candidate_path,
        outcome_path=outcome_path,
    )

    assert not df.empty
    assert candidate_path.exists()
    assert outcome_path.exists()

    candidates = pd.read_parquet(candidate_path)
    outcomes = pd.read_parquet(outcome_path)

    # Candidate rows contain required fields
    required_candidate = [
        "entry_model_id",
        "eligible",
        "raw_score",
        "adjusted_score",
        "policy_label",
        "risk_profile",
        "timestamp_utc",
    ]
    for col in required_candidate:
        assert col in candidates.columns

    # Outcome rows contain required fields
    required_outcome = [
        "entry_model_id",
        "realized_R",
        "max_adverse_excursion",
        "max_favorable_excursion",
        "time_to_outcome",
        "fill_price",
        "exit_price",
    ]
    for col in required_outcome:
        assert col in outcomes.columns

    # Deterministic ordering
    assert list(
        candidates.sort_values(["timestamp_utc", "entry_model_id"]).entry_model_id
    ) == list(candidates.entry_model_id)
    assert list(
        outcomes.sort_values(["timestamp_utc", "entry_model_id"]).entry_model_id
    ) == list(outcomes.entry_model_id)

    # Disabled entries have zero adjusted score
    disabled_rows = candidates[candidates["policy_label"] == "DISABLED"]
    if not disabled_rows.empty:
        assert (disabled_rows["adjusted_score"] == 0.0).all()
