import pytest

from tournament_harness import (
    assert_cross_mode_determinism,
    run_mode,
    sample_market_states,
)


@pytest.mark.parametrize(
    "mode", ["OFFICIAL_MODE", "SANDBOX_MODE", "TEST_MODE", "REPLAY_MODE"]
)
def test_run_mode_returns_trace(mode):
    states = sample_market_states()
    trace = run_mode(mode, states)
    assert len(trace) == len(states)
    assert all(entry.mode == mode for entry in trace)


def test_cross_mode_determinism_holds():
    states = sample_market_states()
    assert_cross_mode_determinism(states)
