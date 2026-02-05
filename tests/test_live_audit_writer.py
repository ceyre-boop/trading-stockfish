from pathlib import Path

import pytest

from engine.live_audit_writer import LiveAuditWriter


def test_order_event_logged(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    writer = LiveAuditWriter(path)
    writer.write("order", {"id": "1", "side": "BUY"})
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "order" in content


def test_fill_event_logged(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    writer = LiveAuditWriter(path)
    writer.write("fill", {"id": "1", "filled": 1})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert any("fill" in line for line in lines)


def test_safety_event_logged(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    writer = LiveAuditWriter(path)
    writer.write("safety", {"state": "SAFE_MODE"})
    content = path.read_text(encoding="utf-8")
    assert "SAFE_MODE" in content


def test_anomaly_event_logged(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    writer = LiveAuditWriter(path)
    writer.write("anomaly", {"kind": "volatility_spike"})
    content = path.read_text(encoding="utf-8")
    assert "volatility_spike" in content


def test_invalid_payload_raises(tmp_path: Path):
    writer = LiveAuditWriter(tmp_path / "audit.jsonl")
    with pytest.raises(ValueError):
        writer.write("", {"id": 1})
