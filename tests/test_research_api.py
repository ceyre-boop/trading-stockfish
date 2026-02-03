import json
from datetime import date
from pathlib import Path

import pandas as pd

from engine import research_api


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_load_decisions_filters_by_date_and_symbol(tmp_path, monkeypatch):
    # Arrange synthetic storage
    monkeypatch.chdir(tmp_path)
    storage_dir = tmp_path / "storage" / "decisions"
    storage_dir.mkdir(parents=True)

    schema = json.loads(
        (_project_root() / "schemas/decisions_storage.schema.json").read_text(
            encoding="utf-8"
        )
    )
    cols = list(schema["properties"].keys())

    rows = [
        {
            "run_id": "r1",
            "decision_id": "d1",
            "timestamp_utc": "2026-02-01T00:00:00Z",
            "symbol": "ES",
            "timeframe": "M5",
            "session_regime": "LONDON",
            "macro_regimes": ["RISK_ON"],
            "key_features": {"f": 1},
            "evaluation_score": 0.1,
            "action": "LONG",
            "outcome_pnl": 1.0,
            "outcome_hit": True,
            "outcome_max_drawdown": 0.1,
            "outcome_holding_period_bars": 5,
            "policy_version": "p1",
            "engine_version": "e1",
            "provenance": {"engine_version": "e1", "policy_version": "p1"},
        },
        {
            "run_id": "r2",
            "decision_id": "d2",
            "timestamp_utc": "2026-02-02T00:00:00Z",
            "symbol": "NQ",
            "timeframe": "M5",
            "session_regime": "NY",
            "macro_regimes": ["RISK_OFF"],
            "key_features": {"f": 2},
            "evaluation_score": 0.2,
            "action": "SHORT",
            "outcome_pnl": -1.0,
            "outcome_hit": False,
            "outcome_max_drawdown": 0.2,
            "outcome_holding_period_bars": 3,
            "policy_version": "p2",
            "engine_version": "e1",
            "provenance": {"engine_version": "e1", "policy_version": "p2"},
        },
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_parquet(storage_dir / "decisions_20260201.parquet", index=False)
    df.iloc[[1]].to_parquet(storage_dir / "decisions_20260202.parquet", index=False)

    filt = research_api.DecisionsFilter(
        start_date=date(2026, 2, 1),
        end_date=date(2026, 2, 1),
        symbols=["ES"],
    )

    # Act
    out = research_api.load_decisions(filt)

    # Assert
    assert set(out.columns) == set(cols)
    assert len(out) == 1
    assert out.iloc[0]["symbol"] == "ES"
    assert out.iloc[0]["timestamp_utc"].startswith("2026-02-01")


def test_compute_regime_performance_returns_expected_metrics():
    data = pd.DataFrame(
        [
            {"session_regime": "A", "outcome_pnl": 1.0, "outcome_hit": True},
            {"session_regime": "A", "outcome_pnl": -1.0, "outcome_hit": False},
            {"session_regime": "B", "outcome_pnl": 2.0, "outcome_hit": True},
        ]
    )

    out = research_api.compute_regime_performance(data)

    a_row = out[out["session_regime"] == "A"].iloc[0]
    b_row = out[out["session_regime"] == "B"].iloc[0]

    # Regime A: pnl mean = 0.0, variance = 1.0, sharpe = 0.0, hit rate = 0.5
    assert a_row["pnl_sum"] == 0.0
    assert a_row["pnl_variance"] == 1.0
    assert a_row["sharpe"] == 0.0
    assert a_row["hit_rate"] == 0.5

    # Regime B: single row, sharpe = 0.0 by definition (sigma=0)
    assert b_row["pnl_sum"] == 2.0
    assert b_row["sharpe"] == 0.0
    assert b_row["hit_rate"] == 1.0

    # Deterministic ordering by regime
    assert list(out["session_regime"]) == sorted(out["session_regime"].tolist())


def test_policy_version_performance_distinguishes_versions():
    decisions = pd.DataFrame(
        [
            {"policy_version": "v1", "outcome_pnl": 1.0, "outcome_hit": True},
            {"policy_version": "v1", "outcome_pnl": -1.0, "outcome_hit": False},
            {"policy_version": "v2", "outcome_pnl": 2.0, "outcome_hit": True},
        ]
    )

    policies = pd.DataFrame(
        [
            {"policy_version": "v1", "timestamp_utc": "2026-02-01T00:00:00Z"},
            {"policy_version": "v2", "timestamp_utc": "2026-02-02T00:00:00Z"},
        ]
    )

    out = research_api.compute_policy_version_performance(decisions, policies)

    assert set(out["policy_version"]) == {"v1", "v2"}

    v1 = out[out["policy_version"] == "v1"].iloc[0]
    v2 = out[out["policy_version"] == "v2"].iloc[0]

    # v1 pnl sum = 0, hit rate = 0.5
    assert v1["pnl_sum"] == 0.0
    assert v1["hit_rate"] == 0.5

    # v2 pnl sum = 2, hit rate = 1
    assert v2["pnl_sum"] == 2.0
    assert v2["hit_rate"] == 1.0

    # Deterministic join keeps policy timestamps per version
    assert v1["policy_timestamp_utc_min"].startswith("2026-02-01")
    assert v2["policy_timestamp_utc_min"].startswith("2026-02-02")
