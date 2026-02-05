"""Live audit writer for Phase 11.

Writes JSONL records for live events with minimal schema validation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class AuditEvent:
    event_type: str
    payload: Dict[str, object]
    timestamp: float
    correlation_id: Optional[str] = None


class LiveAuditWriter:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.parent.mkdir(parents=True, exist_ok=True)

    def _validate(self, event: AuditEvent) -> None:
        if not event.event_type:
            raise ValueError("event_type required")
        if not isinstance(event.payload, dict):
            raise ValueError("payload must be dict")

    def write(
        self,
        event_type: str,
        payload: Dict[str, object],
        correlation_id: Optional[str] = None,
    ) -> Path:
        event = AuditEvent(
            event_type=event_type,
            payload=payload,
            timestamp=time.time(),
            correlation_id=correlation_id,
        )
        self._validate(event)
        line = json.dumps(asdict(event))
        (
            self.base_path.write_text("", encoding="utf-8", errors="ignore")
            if not self.base_path.exists()
            else None
        )
        with self.base_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return self.base_path
