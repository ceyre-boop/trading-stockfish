import pytest

from engine.invariants import InvariantChecker, InvariantViolation
from engine.live_modes import LiveMode
from engine.mode_transition_manager import ModeTransitionManager


def test_valid_transitions_succeed():
    checker = InvariantChecker()
    manager = ModeTransitionManager(checker)

    assert manager.current_mode == LiveMode.SIM_REPLAY
    assert manager.transition_to(LiveMode.SIM_LIVE_FEED) == LiveMode.SIM_LIVE_FEED
    assert manager.transition_to(LiveMode.PAPER_TRADING) == LiveMode.PAPER_TRADING


def test_invalid_transition_raises():
    checker = InvariantChecker()
    manager = ModeTransitionManager(checker)
    with pytest.raises(InvariantViolation):
        manager.transition_to(LiveMode.PAPER_TRADING)
