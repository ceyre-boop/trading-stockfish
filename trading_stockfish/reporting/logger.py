"""
Structured experiment logger: writes JSON-lines experiment records.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentRecord:
    experiment_id: str
    timestamp: float
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    tags: List[str] = field(default_factory=list)


class ExperimentLogger:
    """Appends experiment records to a JSON-lines log file."""

    def __init__(self, log_path: str | Path = "experiment_logs.jsonl") -> None:
        self.log_path = Path(log_path)

    def log(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ) -> ExperimentRecord:
        record = ExperimentRecord(
            experiment_id=experiment_id,
            timestamp=time.time(),
            config=config,
            metrics=metrics,
            tags=tags or [],
        )
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")
        return record

    def load(self) -> List[ExperimentRecord]:
        """Load all records from the log file."""
        if not self.log_path.exists():
            return []
        records = []
        with self.log_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(ExperimentRecord(**json.loads(line)))
        return records
