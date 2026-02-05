import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from engine.tactical_cycle import run_tactical_cycle

ENTRY_A = "ENTRY_SWEEP_DISPLACEMENT_REVERSAL"
ENTRY_B = "ENTRY_FVG_RESPECT_CONTINUATION"


def _policy_df():
    return pd.DataFrame(
        [
            {
                "entry_model_id": ENTRY_A,
                "market_profile_state": "DISTRIBUTION",
                "session_profile": "PROFILE_1A",
                "liquidity_bias_side": "UP",
                "label": "PREFERRED",
            },
            {
                "entry_model_id": ENTRY_B,
                "market_profile_state": "DISTRIBUTION",
                "session_profile": "PROFILE_1A",
                "liquidity_bias_side": "UP",
                "label": "BASELINE",
            },
        ]
    )


def _write_logs(logs_dir: Path, records):
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "decision_log.jsonl"
    with log_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
    return log_path


def _record(ts: str, entry_id: str, outcome: float) -> dict:
    return {
        "timestamp_utc": ts,
        "entry_model_id": entry_id,
        "entry_outcome_R": outcome,
        "eligible_entry_models": [
            entry_id,
            ENTRY_B if entry_id == ENTRY_A else ENTRY_A,
        ],
        "decision_frame": {
            "timestamp_utc": ts,
            "market_profile_state": "DISTRIBUTION",
            "session_profile": "PROFILE_1A",
            "liquidity_frame": {"bias": "UP"},
            "vol_regime": "HIGH",
            "trend_regime": "UP",
            "eligible_entry_models": [
                entry_id,
                ENTRY_B if entry_id == ENTRY_A else ENTRY_A,
            ],
        },
    }


def test_cycle_runs_end_to_end_with_synthetic_data(tmp_path):
    logs_dir = tmp_path / "logs"
    _write_logs(
        logs_dir,
        [
            _record("2024-01-01T00:00:00Z", ENTRY_A, 2.0),
            _record("2024-01-01T01:00:00Z", ENTRY_B, -1.0),
        ],
    )

    out_dir = tmp_path / "out"
    summary = run_tactical_cycle(
        start_date="2024-01-01",
        end_date="2024-01-02",
        logs_dir=str(logs_dir),
        selector_artifact_path=str(out_dir / "selector.joblib"),
        belief_map_path=str(out_dir / "belief.json"),
        policy_path="ignored.json",
        brain_policy_entries=_policy_df(),
    )

    assert summary["rows_replayed"] == 2
    assert summary["dataset_rows"] >= summary["rows_replayed"]
    assert Path(summary["artifact_path"]).exists()
    assert Path(summary["belief_map_path"]).exists()

    belief_map = json.loads(
        Path(summary["belief_map_path"]).read_text(encoding="utf-8")
    )
    assert belief_map, "belief map should not be empty"


def test_cycle_outputs_are_deterministic(tmp_path):
    logs_dir = tmp_path / "logs"
    _write_logs(
        logs_dir,
        [
            _record("2024-02-01T00:00:00Z", ENTRY_A, 1.0),
            _record("2024-02-01T02:00:00Z", ENTRY_B, 0.5),
        ],
    )

    policy_df = _policy_df()

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    summary1 = run_tactical_cycle(
        "2024-02-01",
        "2024-02-02",
        str(logs_dir),
        str(out1 / "selector.joblib"),
        str(out1 / "belief.json"),
        "ignored.json",
        policy_df,
    )
    summary2 = run_tactical_cycle(
        "2024-02-01",
        "2024-02-02",
        str(logs_dir),
        str(out2 / "selector.joblib"),
        str(out2 / "belief.json"),
        "ignored.json",
        policy_df,
    )

    belief1 = Path(summary1["belief_map_path"]).read_bytes()
    belief2 = Path(summary2["belief_map_path"]).read_bytes()
    assert belief1 == belief2

    art1 = joblib.load(summary1["artifact_path"])
    art2 = joblib.load(summary2["artifact_path"])

    def _unpack(obj):
        if isinstance(obj, dict):
            return obj.get("classifier"), obj.get("regressor")
        return obj.classifier, obj.regressor

    clf1, reg1 = _unpack(art1)
    clf2, reg2 = _unpack(art2)

    np.testing.assert_allclose(clf1.coef_, clf2.coef_)
    np.testing.assert_allclose(clf1.intercept_, clf2.intercept_)
    np.testing.assert_allclose(reg1.feature_importances_, reg2.feature_importances_)


def test_cycle_respects_date_window(tmp_path):
    logs_dir = tmp_path / "logs"
    _write_logs(
        logs_dir,
        [
            _record("2024-03-01T00:00:00Z", ENTRY_A, 1.0),
            _record("2024-03-05T00:00:00Z", ENTRY_B, 1.0),
        ],
    )

    out_dir = tmp_path / "out"
    summary = run_tactical_cycle(
        "2024-03-01",
        "2024-03-02",
        str(logs_dir),
        str(out_dir / "selector.joblib"),
        str(out_dir / "belief.json"),
        "ignored.json",
        _policy_df(),
    )

    assert summary["rows_replayed"] == 1
    belief_map = json.loads(
        Path(summary["belief_map_path"]).read_text(encoding="utf-8")
    )
    assert len(belief_map) == 1


def test_cycle_summary_fields_present(tmp_path):
    logs_dir = tmp_path / "logs"
    _write_logs(logs_dir, [_record("2024-04-01T00:00:00Z", ENTRY_A, 0.25)])
    out_dir = tmp_path / "out"
    summary = run_tactical_cycle(
        "2024-04-01",
        "2024-04-02",
        str(logs_dir),
        str(out_dir / "selector.joblib"),
        str(out_dir / "belief.json"),
        "ignored.json",
        _policy_df(),
    )

    for key in ["rows_replayed", "dataset_rows", "artifact_path", "belief_map_path"]:
        assert key in summary
    assert summary["artifact_path"].endswith("selector.joblib")
    assert summary["belief_map_path"].endswith("belief.json")
