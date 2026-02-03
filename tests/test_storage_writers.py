import json
from pathlib import Path

import pandas as pd

from engine import storage_writers


def _decision_entry(decision_id: str, ts: str) -> dict:
    return {
        "run_id": "run_1",
        "decision_id": decision_id,
        "timestamp_utc": ts,
        "symbol": "ES",
        "timeframe": "M5",
        "session_regime": "LONDON_NY_OVERLAP",
        "macro_regimes": ["RISK_ON", "LONDON_NY_OVERLAP"],
        "feature_vector": {"session_range": 1.0},
        "effective_weights": {"session_range": 1.0},
        "policy_components": {
            "base_weights": {"session_range": 1.0},
            "trust": {"session_range": 1.0},
            "regime_multipliers": {},
        },
        "evaluation_score": 0.5,
        "action": "LONG",
        "position_size": 1.0,
        "outcome": {
            "pnl": 0.1,
            "max_drawdown": 0.02,
            "holding_period_bars": 5,
        },
        "provenance": {
            "policy_version": "test_policy",
            "engine_version": "engine_v1",
        },
    }


def _write_decision_log(path: Path, entries: list[dict]) -> None:
    lines = [json.dumps(e) for e in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_write_decisions_to_storage_creates_parquet(tmp_path: Path) -> None:
    log_path = tmp_path / "decision_log.jsonl"
    entries = [
        _decision_entry("d1", "2026-02-03T00:00:00Z"),
        _decision_entry("d2", "2026-02-03T00:05:00Z"),
    ]
    _write_decision_log(log_path, entries)

    cfg = storage_writers.StorageConfig(base_dir=tmp_path / "storage")
    storage_writers.write_decisions_to_storage(log_path, cfg)

    target = cfg.base_dir / "decisions" / "decisions_20260203.parquet"
    assert target.exists()

    df = pd.read_parquet(target)
    schema_cols = set(
        json.loads(
            Path("schemas/decisions_storage.schema.json").read_text(encoding="utf-8")
        )["properties"].keys()
    )
    assert set(df.columns) == schema_cols
    assert len(df) == len(entries)


def test_storage_writers_are_idempotent(tmp_path: Path) -> None:
    log_path = tmp_path / "decision_log.jsonl"
    entries = [
        _decision_entry("d1", "2026-02-03T00:00:00Z"),
        _decision_entry("d2", "2026-02-03T00:05:00Z"),
    ]
    _write_decision_log(log_path, entries)

    cfg = storage_writers.StorageConfig(base_dir=tmp_path / "storage")
    storage_writers.write_decisions_to_storage(log_path, cfg)
    storage_writers.write_decisions_to_storage(log_path, cfg)

    target = cfg.base_dir / "decisions" / "decisions_20260203.parquet"
    df = pd.read_parquet(target)
    assert len(df) == len(entries)


def test_policy_storage_tracks_versions(tmp_path: Path) -> None:
    cfg = storage_writers.StorageConfig(base_dir=tmp_path / "storage")

    policy_v1 = tmp_path / "policy_v1.json"
    policy_v1.write_text(
        json.dumps(
            {
                "version": "v1",
                "timestamp_utc": "2026-02-03T00:00:00Z",
                "base_weights": {"feature_a": 1.0},
                "trust": {"feature_a": 1.0},
                "regime_multipliers": {"RISK_ON": {"feature_a": 1.0}},
                "engine_version": "engine_v1",
                "source": "test",
            }
        ),
        encoding="utf-8",
    )

    policy_v2 = tmp_path / "policy_v2.json"
    policy_v2.write_text(
        json.dumps(
            {
                "version": "v2",
                "timestamp_utc": "2026-02-03T01:00:00Z",
                "base_weights": {"feature_a": 2.0},
                "trust": {"feature_a": 1.0},
                "regime_multipliers": {"RISK_ON": {"feature_a": 2.0}},
                "engine_version": "engine_v1",
                "source": "test",
            }
        ),
        encoding="utf-8",
    )

    storage_writers.write_policy_to_storage(policy_v1, cfg)
    storage_writers.write_policy_to_storage(policy_v2, cfg)

    target = cfg.base_dir / "policies" / "policies.parquet"
    df = pd.read_parquet(target)
    assert set(df["policy_version"]) == {"v1", "v2"}
    assert len(df) == 2
