import pytest

from engine.market_state_builder import build_market_state


def test_market_state_builder_emits_amd_state():
    state = build_market_state(symbol="TEST", order_book_events=[])
    assert "amd_state" in state
    amd = state["amd_state"]
    assert amd["amd_tag"] == "NEUTRAL"
    assert "amd_confidence" in amd
    # Regime engine should surface amd_regime
    assert "regime_state" in state
    assert state["regime_state"].get("amd_regime") == amd.get("amd_tag")
