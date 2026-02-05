import types

import pytest

from engine.decision_frame import DecisionFrame
from engine.entry_eligibility import is_entry_eligible
from engine.entry_eligibility_overrides import (
    eligible_mean_reversion_range_extreme,
    eligible_sweep_displacement_reversal,
)
from engine.entry_models import ENTRY_MODELS


def test_override_called_for_sweep_displacement(monkeypatch):
    entry = ENTRY_MODELS["ENTRY_SWEEP_DISPLACEMENT_REVERSAL"]
    frame = DecisionFrame(entry_signals_present={"sweep": False})

    called = {}

    def fake_override(df):
        called["used"] = True
        return True

    original_fn = entry.eligibility_fn
    object.__setattr__(entry, "eligibility_fn", fake_override)
    try:
        assert is_entry_eligible(entry, frame) is True
        assert called.get("used") is True
    finally:
        object.__setattr__(entry, "eligibility_fn", original_fn)


def test_override_called_for_mean_reversion(monkeypatch):
    entry = ENTRY_MODELS["ENTRY_MEAN_REVERSION_RANGE_EXTREME"]
    frame = DecisionFrame(entry_signals_present={"sweep": True})

    called = {}

    def fake_override(df):
        called["used"] = True
        return False

    original_fn = entry.eligibility_fn
    object.__setattr__(entry, "eligibility_fn", fake_override)
    try:
        assert is_entry_eligible(entry, frame) is False
        assert called.get("used") is True
    finally:
        object.__setattr__(entry, "eligibility_fn", original_fn)


def test_generic_logic_used_when_no_override(monkeypatch):
    entry = ENTRY_MODELS["ENTRY_FVG_RESPECT_CONTINUATION"]
    frame = DecisionFrame(
        market_profile_state="DISTRIBUTION",
        session_profile="PROFILE_1C",
        liquidity_frame={
            "bias": "UP",
            "sweep_state": "ANY",
            "distance_bucket": "NEAR",
        },
        entry_signals_present={"fvg": True},
        condition_vector={"vol": "NORMAL", "trend": "UP"},
    )

    # Ensure no override exists
    assert entry.eligibility_fn is None
    assert is_entry_eligible(entry, frame) is True


@pytest.mark.parametrize(
    "func,signals,disp,expected",
    [
        (eligible_sweep_displacement_reversal, {"sweep": True}, 0.5, True),
        (eligible_sweep_displacement_reversal, {"sweep": False}, 0.5, False),
        (eligible_mean_reversion_range_extreme, {"sweep": False}, 0.2, True),
        (eligible_mean_reversion_range_extreme, {"sweep": True}, 0.2, False),
    ],
)
def test_override_behaviors_deterministic(func, signals, disp, expected):
    frame = DecisionFrame(
        entry_signals_present=signals,
        market_profile_evidence={"displacement_score": disp},
    )
    assert func(frame) is expected
