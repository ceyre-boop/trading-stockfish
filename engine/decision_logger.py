import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None


class DecisionLogger:
    """Append-only JSONL decision logger with optional schema validation."""

    def __init__(self, log_path: Path | str, schema_path: Optional[Path | str] = None):
        self.log_path = Path(log_path)
        self.schema_path = Path(schema_path) if schema_path else None
        self._schema: Optional[Dict[str, Any]] = None
        if self.schema_path and jsonschema is None:
            # Schema exists but jsonschema not installed; proceed without validation.
            self._schema = None

    def _load_schema(self) -> None:
        if self._schema is not None or self.schema_path is None or jsonschema is None:
            return
        try:
            self._schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to load decision log schema: {exc}")

    def log_decision(self, entry: Dict[str, Any]) -> None:
        """Validate (optional) and append a single decision entry as JSONL."""

        self._load_schema()
        if self._schema is not None and jsonschema is not None:
            try:
                jsonschema.validate(instance=entry, schema=self._schema)
            except jsonschema.ValidationError as exc:  # pragma: no cover
                raise ValueError(f"Decision entry failed validation: {exc.message}")

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(entry, ensure_ascii=False)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception as exc:  # pragma: no cover
            raise IOError(f"Failed to write decision log entry: {exc}")
