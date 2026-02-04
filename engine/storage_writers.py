"""Storage writers for long-term Parquet archives.

Deterministic, append-only consolidation of existing log artifacts into
columnar storage. This module does **not** change logging; it only reads
existing artifacts, validates them against log schemas, transforms them
into the storage schemas, and writes Parquet partitions with de-duplication
on stable keys.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None


@dataclass
class StorageConfig:
    base_dir: Path = Path("storage")
    format: str = "parquet"  # only parquet implemented
    parquet_engine: str = "pyarrow"  # preferred engine


# -------------------------- helpers --------------------------


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_date(ts: Optional[str]) -> Optional[str]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y%m%d")
    except Exception:
        return None


def _load_json_schema(path: Path) -> Optional[Dict[str, Any]]:
    if not jsonschema or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover
        return None


def _validate(entry: Dict[str, Any], schema: Optional[Dict[str, Any]]) -> None:
    if schema is None or not jsonschema:
        return
    try:
        jsonschema.validate(instance=entry, schema=schema)
    except jsonschema.ValidationError as exc:  # pragma: no cover
        raise ValueError(f"Validation failed: {exc.message}")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return records


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_parquet(df: pd.DataFrame, path: Path, engine: str) -> None:
    _ensure_dir(path)
    df.to_parquet(path, index=False, engine=engine)


def _load_parquet(path: Path, engine: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_parquet(path, engine=engine)
    except Exception:
        logging.getLogger(__name__).warning(
            "skipping invalid parquet file", extra={"path": str(path)}
        )
        return pd.DataFrame()


# -------------------------- transformations --------------------------


def _decision_rows_from_log(entry: Dict[str, Any]) -> Dict[str, Any]:
    prov = (
        entry.get("provenance", {}) if isinstance(entry.get("provenance"), dict) else {}
    )
    outcome = entry.get("outcome", {}) if isinstance(entry.get("outcome"), dict) else {}
    return {
        "run_id": entry.get("run_id"),
        "decision_id": entry.get("decision_id"),
        "timestamp_utc": entry.get("timestamp_utc"),
        "symbol": entry.get("symbol"),
        "timeframe": entry.get("timeframe"),
        "session_regime": entry.get("session_regime"),
        "macro_regimes": entry.get("macro_regimes", []),
        "key_features": entry.get("feature_vector") or {},
        "evaluation_score": entry.get("evaluation_score"),
        "action": entry.get("action"),
        "outcome_pnl": outcome.get("pnl"),
        "outcome_hit": outcome.get("hit"),
        "outcome_max_drawdown": outcome.get("max_drawdown"),
        "outcome_holding_period_bars": outcome.get("holding_period_bars"),
        "policy_version": prov.get("policy_version"),
        "engine_version": prov.get("engine_version"),
        "provenance": prov,
    }


def _audit_rows_from_obj(
    obj: Dict[str, Any], audit_id_fallback: str
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metadata = obj.get("metadata") or None
    issues = obj.get("issues") if isinstance(obj.get("issues"), list) else [None]
    for issue in issues:
        issue_dict = issue if isinstance(issue, dict) else {}
        rows.append(
            {
                "run_id": obj.get("run_id"),
                "audit_id": issue_dict.get("id")
                or obj.get("audit_id")
                or audit_id_fallback,
                "timestamp_utc": obj.get("timestamp_utc"),
                "feature_name": issue_dict.get("feature_name")
                or issue_dict.get("feature"),
                "raw_value": issue_dict.get("raw_value") or issue_dict.get("raw"),
                "normalized_value": issue_dict.get("normalized_value")
                or issue_dict.get("normalized"),
                "session_regime": issue_dict.get("session_regime")
                or obj.get("session_regime"),
                "macro_regimes": issue_dict.get("macro_regimes")
                or obj.get("macro_regimes")
                or [],
                "issues": (
                    issue_dict.get("issues")
                    if isinstance(issue_dict.get("issues"), list)
                    else []
                ),
                "policy_version": obj.get("policy_version")
                or obj.get("provenance", {}).get("policy_version"),
                "engine_version": obj.get("engine_version")
                or obj.get("provenance", {}).get("engine_version"),
                "metadata": metadata,
            }
        )
    # Emit a placeholder row when no issues are present so clean runs are still recorded.
    if not rows:
        rows.append(
            {
                "run_id": obj.get("run_id"),
                "audit_id": obj.get("audit_id") or audit_id_fallback,
                "timestamp_utc": obj.get("timestamp_utc"),
                "feature_name": None,
                "raw_value": None,
                "normalized_value": None,
                "session_regime": obj.get("session_regime"),
                "macro_regimes": obj.get("macro_regimes") or [],
                "issues": [],
                "policy_version": obj.get("policy_version")
                or obj.get("provenance", {}).get("policy_version"),
                "engine_version": obj.get("engine_version")
                or obj.get("provenance", {}).get("engine_version"),
                "metadata": metadata,
            }
        )
    return rows


def _stats_rows_from_obj(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    run_id = obj.get("run_id")
    ts = obj.get("timestamp_utc") or obj.get("timestamp")
    perf = obj.get("feature_performance_by_regime") or {}
    metadata = obj.get("metadata") or None
    # Flatten per feature, per regime metrics if present
    for feature, regimes in perf.items():
        if not isinstance(regimes, dict):
            continue
        for regime, metrics in regimes.items():
            metrics_dict = metrics if isinstance(metrics, dict) else {}
            rows.append(
                {
                    "run_id": run_id,
                    "timestamp_utc": ts,
                    "regime": regime,
                    "feature_name": feature,
                    "sharpe": metrics_dict.get("sharpe"),
                    "hit_rate": metrics_dict.get("hit_rate"),
                    "variance": metrics_dict.get("variance"),
                    "stability": metrics_dict.get("stability"),
                    "window_start": metrics_dict.get("window_start"),
                    "window_end": metrics_dict.get("window_end"),
                    "metadata": metadata,
                }
            )
    # Fallback: if no per-regime block, emit per-feature aggregates if available
    if not rows:
        fi = obj.get("feature_importance") or {}
        fs = obj.get("feature_stability") or {}
        for feature in set(list(fi.keys()) + list(fs.keys())):
            rows.append(
                {
                    "run_id": run_id,
                    "timestamp_utc": ts,
                    "regime": None,
                    "feature_name": feature,
                    "sharpe": None,
                    "hit_rate": None,
                    "variance": None,
                    "stability": fs.get(feature),
                    "window_start": None,
                    "window_end": None,
                    "metadata": metadata,
                }
            )
    return rows


def _policy_row_from_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "policy_version": obj.get("version") or obj.get("policy_version"),
        "timestamp_utc": obj.get("timestamp_utc"),
        "base_weights": obj.get("base_weights") or {},
        "trust": obj.get("trust") or {},
        "regime_multipliers": obj.get("regime_multipliers") or {},
        "metadata": {
            "source": obj.get("source"),
            "notes": obj.get("notes"),
            "engine_version": obj.get("engine_version")
            or obj.get("provenance", {}).get("engine_version"),
        },
    }


# -------------------------- core writers --------------------------


def write_decisions_to_storage(
    decision_log_path: Path, storage_config: Optional[StorageConfig] = None
) -> Path:
    cfg = storage_config or StorageConfig()
    log_schema = _load_json_schema(Path("schemas/decision_log.schema.json"))
    storage_path = cfg.base_dir / "decisions"

    records = _read_jsonl(Path(decision_log_path))
    rows = []
    for entry in records:
        _validate(entry, log_schema)
        row = _decision_rows_from_log(entry)
        rows.append(row)

    if not rows:
        return storage_path

    # partition by date
    for row in rows:
        date_str = _parse_date(row.get("timestamp_utc")) or "undated"
        target = storage_path / f"decisions_{date_str}.parquet"
        existing = _load_parquet(target, cfg.parquet_engine)
        df_new = pd.DataFrame([row])
        combined = pd.concat([existing, df_new], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["run_id", "decision_id"], keep="first"
        )
        _write_parquet(combined, target, cfg.parquet_engine)

    return storage_path


def write_audits_to_storage(
    audit_paths: Iterable[Path], storage_config: Optional[StorageConfig] = None
) -> Path:
    cfg = storage_config or StorageConfig()
    storage_path = cfg.base_dir / "audits"
    # No dedicated log schema yet; skip validation if absent.

    for audit_path in audit_paths:
        obj = _read_json(audit_path)
        rows = _audit_rows_from_obj(obj, audit_id_fallback=audit_path.stem)
        if not rows:
            continue
        date_str = _parse_date(rows[0].get("timestamp_utc")) or "undated"
        target = storage_path / f"audits_{date_str}.parquet"
        existing = _load_parquet(target, cfg.parquet_engine)
        df_new = pd.DataFrame(rows)
        combined = pd.concat([existing, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["run_id", "audit_id"], keep="first")
        _write_parquet(combined, target, cfg.parquet_engine)

    return storage_path


def write_stats_to_storage(
    stats_path: Path, storage_config: Optional[StorageConfig] = None
) -> Path:
    cfg = storage_config or StorageConfig()
    storage_path = cfg.base_dir / "stats"
    obj = _read_json(stats_path)
    rows = _stats_rows_from_obj(obj)
    if not rows:
        return storage_path

    date_str = _parse_date(rows[0].get("timestamp_utc")) or "undated"
    target = storage_path / f"stats_{date_str}.parquet"
    existing = _load_parquet(target, cfg.parquet_engine)
    df_new = pd.DataFrame(rows)
    combined = pd.concat([existing, df_new], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["run_id", "regime", "feature_name"], keep="first"
    )
    _write_parquet(combined, target, cfg.parquet_engine)
    return storage_path


def write_policy_to_storage(
    policy_config: Path | Dict[str, Any], storage_config: Optional[StorageConfig] = None
) -> Path:
    cfg = storage_config or StorageConfig()
    storage_path = cfg.base_dir / "policies"

    if isinstance(policy_config, (str, Path)):
        obj = _read_json(Path(policy_config))
    else:
        obj = policy_config

    row = _policy_row_from_obj(obj)
    if not row.get("policy_version"):
        raise ValueError("policy_version is required to store policy")

    target = storage_path / "policies.parquet"
    existing = _load_parquet(target, cfg.parquet_engine)
    df_new = pd.DataFrame([row])
    combined = pd.concat([existing, df_new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["policy_version"], keep="first")
    _write_parquet(combined, target, cfg.parquet_engine)
    return storage_path
