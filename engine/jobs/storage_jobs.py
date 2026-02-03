from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from engine import storage_writers


def _log(msg: str, **kwargs: object) -> None:
    entry: Dict[str, object] = {"level": "INFO", "msg": msg}
    entry.update(kwargs)
    print(json.dumps(entry, default=str))


def _date_range(from_date: str, to_date: str) -> List[str]:
    start = datetime.strptime(from_date, "%Y-%m-%d").date()
    end = datetime.strptime(to_date, "%Y-%m-%d").date()
    if start > end:
        raise ValueError("from_date must be on or before to_date")
    days = []
    cur = start
    while cur <= end:
        days.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return days


def _decision_logs(base: Path = Path("logs")) -> List[Path]:
    if not base.exists():
        return []
    paths = {p for p in base.glob("*.jsonl") if p.is_file()}
    return sorted(paths)


def _audit_logs(base: Path = Path("logs/feature_audits")) -> List[Path]:
    if not base.exists():
        return []
    return sorted(p for p in base.glob("*.json") if p.is_file())


def _stats_logs(base: Path = Path("logs")) -> List[Path]:
    if not base.exists():
        return []
    return sorted(p for p in base.glob("feature_stats*.json") if p.is_file())


def _policy_logs(base: Path = Path("logs")) -> List[Path]:
    paths: List[Path] = []
    active = base / "policy_config.json"
    if active.is_file():
        paths.append(active)
    archive = base / "policy_archive"
    if archive.exists():
        paths.extend(sorted(p for p in archive.glob("*.json") if p.is_file()))
    return paths


def _filter_decisions_by_date(
    date_str: str, sources: Sequence[Path]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sources:
        for entry in storage_writers._read_jsonl(path):
            ts = storage_writers._parse_date(entry.get("timestamp_utc"))
            if ts == date_str:
                rows.append(entry)
    return rows


def _filter_decisions_by_run(
    run_id: str, sources: Sequence[Path]
) -> Dict[str, List[Dict[str, object]]]:
    rows: Dict[str, List[Dict[str, object]]] = {}
    for path in sources:
        for entry in storage_writers._read_jsonl(path):
            if str(entry.get("run_id")) != run_id:
                continue
            ts = storage_writers._parse_date(entry.get("timestamp_utc")) or "undated"
            rows.setdefault(ts, []).append(entry)
    return rows


def _filter_audits_by_date(date_str: str, sources: Sequence[Path]) -> List[Path]:
    matched: List[Path] = []
    for path in sources:
        try:
            obj = storage_writers._read_json(path)
        except Exception:
            continue
        ts = storage_writers._parse_date(obj.get("timestamp_utc"))
        if ts == date_str:
            matched.append(path)
    return matched


def _filter_audits_by_run(run_id: str, sources: Sequence[Path]) -> List[Path]:
    matched: List[Path] = []
    for path in sources:
        try:
            obj = storage_writers._read_json(path)
        except Exception:
            continue
        if str(obj.get("run_id")) == run_id:
            matched.append(path)
    return matched


def _filter_stats_by_date(date_str: str, sources: Sequence[Path]) -> List[Path]:
    matched: List[Path] = []
    for path in sources:
        try:
            obj = storage_writers._read_json(path)
        except Exception:
            continue
        ts = storage_writers._parse_date(obj.get("timestamp_utc"))
        if ts == date_str:
            matched.append(path)
    return matched


def _filter_stats_by_run(run_id: str, sources: Sequence[Path]) -> List[Path]:
    matched: List[Path] = []
    for path in sources:
        try:
            obj = storage_writers._read_json(path)
        except Exception:
            continue
        if str(obj.get("run_id")) == run_id:
            matched.append(path)
    return matched


def _filter_policies_by_date(date_str: str, sources: Sequence[Path]) -> List[Path]:
    matched: List[Path] = []
    for path in sources:
        try:
            obj = storage_writers._read_json(path)
        except Exception:
            continue
        ts = storage_writers._parse_date(obj.get("timestamp_utc"))
        if ts == date_str:
            matched.append(path)
    return matched


def _filter_policies_by_run(run_id: str, sources: Sequence[Path]) -> List[Path]:
    matched: List[Path] = []
    for path in sources:
        try:
            obj = storage_writers._read_json(path)
        except Exception:
            continue
        if str(obj.get("run_id")) == run_id:
            matched.append(path)
    return matched


def _write_decisions(
    records: Sequence[Dict[str, object]], cfg: storage_writers.StorageConfig
) -> None:
    if not records:
        return
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as handle:
            tmp_path = Path(handle.name)
            for entry in records:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        storage_writers.write_decisions_to_storage(tmp_path, cfg)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def backfill_storage(
    from_date: str,
    to_date: str,
    storage_config: Optional[storage_writers.StorageConfig] = None,
) -> None:
    cfg = storage_config or storage_writers.StorageConfig()
    dates = _date_range(from_date, to_date)
    decision_sources = _decision_logs()
    audit_sources = _audit_logs()
    stats_sources = _stats_logs()
    policy_sources = _policy_logs()

    for date_str in dates:
        decisions = _filter_decisions_by_date(date_str, decision_sources)
        audits = _filter_audits_by_date(date_str, audit_sources)
        stats = _filter_stats_by_date(date_str, stats_sources)
        policies = _filter_policies_by_date(date_str, policy_sources)

        if not any([decisions, audits, stats, policies]):
            _log("no artifacts found", date=date_str)
            continue

        if decisions:
            _write_decisions(decisions, cfg)
            _log("decisions appended", date=date_str, count=len(decisions))

        if audits:
            storage_writers.write_audits_to_storage(audits, cfg)
            _log("audits appended", date=date_str, count=len(audits))

        for path in stats:
            storage_writers.write_stats_to_storage(path, cfg)
            _log("stats appended", date=date_str, path=str(path))

        for path in policies:
            storage_writers.write_policy_to_storage(path, cfg)
            _log("policies appended", date=date_str, path=str(path))


def update_storage_for_run(
    run_id: str, storage_config: Optional[storage_writers.StorageConfig] = None
) -> None:
    cfg = storage_config or storage_writers.StorageConfig()
    decision_sources = _decision_logs()
    audit_sources = _audit_logs()
    stats_sources = _stats_logs()
    policy_sources = _policy_logs()

    decisions_by_date = _filter_decisions_by_run(run_id, decision_sources)
    audits = _filter_audits_by_run(run_id, audit_sources)
    stats = _filter_stats_by_run(run_id, stats_sources)
    policies = _filter_policies_by_run(run_id, policy_sources)

    if not any([decisions_by_date, audits, stats, policies]):
        _log("no artifacts found", run_id=run_id)
        return

    for date_str, records in sorted(decisions_by_date.items()):
        _write_decisions(records, cfg)
        _log("decisions appended", run_id=run_id, date=date_str, count=len(records))

    if audits:
        storage_writers.write_audits_to_storage(audits, cfg)
        _log("audits appended", run_id=run_id, count=len(audits))

    for path in stats:
        storage_writers.write_stats_to_storage(path, cfg)
        _log("stats appended", run_id=run_id, path=str(path))

    for path in policies:
        storage_writers.write_policy_to_storage(path, cfg)
        _log("policies appended", run_id=run_id, path=str(path))


def _cli_backfill() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill Parquet storage from log artifacts"
    )
    parser.add_argument(
        "--from",
        dest="from_date",
        required=True,
        help="Start date inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--to", dest="to_date", required=True, help="End date inclusive (YYYY-MM-DD)"
    )
    args = parser.parse_args()
    backfill_storage(args.from_date, args.to_date)


def _cli_update_run() -> None:
    parser = argparse.ArgumentParser(
        description="Append storage rows for a single run_id"
    )
    parser.add_argument(
        "--run-id", dest="run_id", required=True, help="Run identifier to consolidate"
    )
    args = parser.parse_args()
    update_storage_for_run(args.run_id)


if __name__ == "__main__":
    # Default to backfill CLI when invoked directly.
    _cli_backfill()
