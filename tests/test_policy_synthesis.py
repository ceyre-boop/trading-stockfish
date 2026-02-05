import json
from pathlib import Path

import pandas as pd

from engine.policy_synthesis import synthesize_entry_policy


def _dataset():
    rows = []
    # ENTRY_A: strong positive, enough samples
    for i in range(60):
        rows.append(
            {
                "entry_model_id": "ENTRY_A",
                "action_type": "OPEN_LONG",
                "label_realized_R": 0.3,
            }
        )
    # ENTRY_B: mild positive, enough samples
    for i in range(55):
        rows.append(
            {
                "entry_model_id": "ENTRY_B",
                "action_type": "OPEN_SHORT",
                "label_realized_R": 0.1,
            }
        )
    # ENTRY_C: negative
    for i in range(70):
        rows.append(
            {
                "entry_model_id": "ENTRY_C",
                "action_type": "OPEN_LONG",
                "label_realized_R": -0.2,
            }
        )
    # ENTRY_D: insufficient samples
    for i in range(10):
        rows.append(
            {
                "entry_model_id": "ENTRY_D",
                "action_type": "OPEN_LONG",
                "label_realized_R": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_policy_synthesis_labels(tmp_path):
    df = _dataset()
    out_path = tmp_path / "brain_policy_entries.learned.json"
    labels = synthesize_entry_policy(df, min_samples=50, output_path=out_path)

    assert labels["ENTRY_A"] == "PREFERRED"
    assert labels["ENTRY_B"] == "ALLOWED"
    assert labels["ENTRY_C"] == "DISCOURAGED"
    assert labels["ENTRY_D"] == "DISABLED"

    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    keys = [rec["entry_model_id"] for rec in artifact.get("policy", [])]
    assert keys == sorted(keys)


def test_policy_synthesis_filters_non_open_actions():
    df = pd.DataFrame(
        [
            {
                "entry_model_id": "ENTRY_Z",
                "action_type": "NO_TRADE",
                "label_realized_R": 0.5,
            },
        ]
    )
    labels = synthesize_entry_policy(df, min_samples=1)
    assert labels == {}
