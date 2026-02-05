import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .decision_actions import (
    ActionType,
    DecisionAction,
    DecisionOutcome,
    DecisionRecord,
)

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

    def _normalize(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(entry)
        cv = normalized.get("condition_vector")
        if cv is not None and is_dataclass(cv):
            normalized["condition_vector"] = asdict(cv)
        df = normalized.get("decision_frame")
        if df is not None:
            if is_dataclass(df):
                normalized["decision_frame"] = asdict(df)
            elif hasattr(df, "to_dict"):
                try:
                    normalized["decision_frame"] = df.to_dict()
                except Exception:
                    pass
        # Normalize optional brain influence fields for deterministic logging
        for key in [
            "brain_label",
            "brain_prob_good",
            "brain_expected_reward",
            "brain_sample_size",
            "brain_influence_applied",
            "brain_adjusted_score",
            "entry_brain_labels",
            "entry_brain_scores",
        ]:
            if key not in normalized:
                normalized[key] = None
        return normalized


def _coerce_action_type(raw: Any) -> ActionType:
    if isinstance(raw, ActionType):
        return raw
    value = str(raw).upper() if raw is not None else "NO_TRADE"
    if value in {"LONG", "OPEN_LONG"}:
        return ActionType.OPEN_LONG
    if value in {"SHORT", "OPEN_SHORT"}:
        return ActionType.OPEN_SHORT
    if value == ActionType.MANAGE_POSITION.value:
        return ActionType.MANAGE_POSITION
    return ActionType.NO_TRADE


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def build_decision_record(log_entry: Dict[str, Any]) -> DecisionRecord:
    """Construct a DecisionRecord from an existing decision log row without altering logging behavior."""

    if log_entry is None:
        raise ValueError("log_entry is required")

    action_type = _coerce_action_type(log_entry.get("action"))
    direction = log_entry.get("direction")
    if direction is None and action_type in (
        ActionType.OPEN_LONG,
        ActionType.OPEN_SHORT,
    ):
        direction = "LONG" if action_type == ActionType.OPEN_LONG else "SHORT"

    action = DecisionAction(
        action_type=action_type,
        entry_model_id=log_entry.get("entry_model_id"),
        direction=direction,
        size_bucket=log_entry.get("size_bucket"),
        stop_structure=log_entry.get("stop_structure"),
        tp_structure=log_entry.get("tp_structure"),
        manage_payload=log_entry.get("manage_payload"),
    )

    outcome_payload = log_entry.get("outcome") or {}
    realized_r = outcome_payload.get("pnl")
    if realized_r is None:
        realized_r = log_entry.get("entry_outcome") or log_entry.get("realized_R")
    mae = outcome_payload.get("max_drawdown")
    if mae is None:
        mae = log_entry.get("max_adverse_excursion")
    mfe = outcome_payload.get("max_favorable_excursion")
    if mfe is None:
        mfe = log_entry.get("max_favorable_excursion")
    time_in_trade = outcome_payload.get("holding_period_bars")
    if time_in_trade is None:
        time_in_trade = log_entry.get("time_to_outcome")
    drawdown_impact = outcome_payload.get("max_drawdown")

    outcome: Optional[DecisionOutcome] = None
    if realized_r is not None:
        outcome = DecisionOutcome(
            realized_R=_safe_float(realized_r, 0.0),
            max_adverse_excursion=_safe_float(mae, 0.0),
            max_favorable_excursion=_safe_float(mfe, 0.0),
            time_in_trade_bars=_safe_int(time_in_trade, 0),
            drawdown_impact=_safe_float(drawdown_impact, 0.0),
        )

    decision_id = str(log_entry.get("decision_id") or log_entry.get("id") or "")

    metadata_candidates = {
        "symbol": log_entry.get("symbol"),
        "timeframe": log_entry.get("timeframe"),
        "session_regime": log_entry.get("session_regime"),
        "macro_regimes": log_entry.get("macro_regimes"),
        "provenance": log_entry.get("provenance"),
    }
    metadata = {k: v for k, v in metadata_candidates.items() if v is not None}

    state_ref = (
        log_entry.get("state_ref")
        or log_entry.get("replay_id")
        or log_entry.get("run_id")
    )

    return DecisionRecord(
        decision_id=decision_id,
        timestamp_utc=log_entry.get("timestamp_utc"),
        bar_index=log_entry.get("bar_index"),
        state_ref=state_ref,
        action=action,
        outcome=outcome,
        metadata=metadata,
    )

    def log_decision(self, entry: Dict[str, Any]) -> None:
        """Validate (optional) and append a single decision entry as JSONL."""

        entry = self._normalize(entry)

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
