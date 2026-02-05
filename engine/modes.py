from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Type

if TYPE_CHECKING:
    from engine import guardrails


class Mode(enum.Enum):
    SIMULATION = "SIMULATION"
    PAPER = "PAPER"
    LIVE = "LIVE"


def resolve_mode(mode_str: str) -> Mode:
    try:
        return Mode(mode_str.upper())
    except Exception:
        raise ValueError(f"Unknown mode: {mode_str}")


@dataclass
class ExecutionAdapter:
    name: str
    live: bool = False
    disabled: bool = False
    placed_orders: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)

    def place_order(self, *_, **kwargs):
        # Guardrails: enforce only in LIVE; log-only in SIM/PAPER.
        state = kwargs.pop("guardrail_state", {}) or {}
        metrics = kwargs.pop("guardrail_metrics", {}) or {}

        # Lazy import to avoid circular dependency with guardrails module.
        from engine import guardrails

        decision = guardrails.check_runtime_limits(state, metrics)
        if self.live:
            result = guardrails.apply_guardrail_decision(decision, adapter=self)
            if decision.triggered:
                return {
                    "status": "blocked_by_guardrail",
                    "guardrail_type": decision.guardrail_type,
                    "reason": decision.reason,
                    "safe_mode_required": decision.safe_mode_required,
                    "result": result,
                }
        else:
            if decision.triggered:
                logging.getLogger(__name__).info(
                    "Guardrail would block order in %s: %s",
                    self.name,
                    decision.guardrail_type,
                )

        if self.disabled:
            raise RuntimeError("Adapter is disabled")

        self.placed_orders += 1
        return {"status": "simulated", "adapter": self.name}

    def disable_orders(self) -> None:
        self.disabled = True


class SimulationAdapter(ExecutionAdapter):
    def __init__(self) -> None:
        super().__init__(name="SIMULATION", live=False)


class PaperAdapter(ExecutionAdapter):
    def __init__(self) -> None:
        super().__init__(name="PAPER", live=False)


class LiveAdapter(ExecutionAdapter):
    def __init__(self) -> None:
        super().__init__(name="LIVE", live=True)


_ADAPTERS: Dict[Mode, Type[ExecutionAdapter]] = {
    Mode.SIMULATION: SimulationAdapter,
    Mode.PAPER: PaperAdapter,
    Mode.LIVE: LiveAdapter,
}


def get_adapter(mode: Mode) -> ExecutionAdapter:
    cls = _ADAPTERS.get(mode)
    if cls is None:
        raise ValueError(f"No adapter for mode {mode}")
    return cls()
