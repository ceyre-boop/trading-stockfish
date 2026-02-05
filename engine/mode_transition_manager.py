from __future__ import annotations

from .invariants import InvariantChecker
from .live_modes import LiveMode


class ModeTransitionManager:
    def __init__(self, invariant_checker: InvariantChecker):
        self.current_mode = LiveMode.SIM_REPLAY
        self.invariant_checker = invariant_checker

    def transition_to(self, new_mode: LiveMode) -> LiveMode:
        self.invariant_checker.assert_mode_transition_valid(self.current_mode, new_mode)
        self.current_mode = new_mode
        return self.current_mode
