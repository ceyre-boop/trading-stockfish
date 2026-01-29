from datetime import datetime, timezone

import pandas as pd
import pytest

from analytics.data_loader import MarketStateBuilder


def create_sample_ohlcv():
    """Create a simple OHLCV dataframe for testing"""
    dates = pd.date_range(
        "2026-01-15 03:00:00", periods=100, freq="1min", tz=timezone.utc
    )
    df = pd.DataFrame(
        {
            "open": [5000.0 + i * 0.1 for i in range(100)],
            "high": [5001.0 + i * 0.1 for i in range(100)],
            "low": [4999.0 + i * 0.1 for i in range(100)],
            "close": [5000.5 + i * 0.1 for i in range(100)],
            "volume": [1000 + i * 10 for i in range(100)],
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


def test_market_state_builder_session_integration():
    """Test that MarketStateBuilder correctly injects session/flow context"""
    builder = MarketStateBuilder("ES", "1m", lookback=30)
    df = create_sample_ohlcv()

    # Build states
    states = builder.build_states(df, time_causal_check=False)

    # Verify session context fields exist
    for state in states:
        assert hasattr(state, "session_name")
        assert hasattr(state, "session_vol_scale")
        assert hasattr(state, "session_liq_scale")
        assert hasattr(state, "session_risk_scale")
        assert hasattr(state, "prior_high")
        assert hasattr(state, "prior_low")
        assert hasattr(state, "overnight_high")
        assert hasattr(state, "overnight_low")
        assert hasattr(state, "vwap")
        assert hasattr(state, "vwap_distance_pct")
        assert hasattr(state, "round_level_proximity")
        assert hasattr(state, "stop_run_detected")
        assert hasattr(state, "initiative_move_detected")
        assert hasattr(state, "level_reaction_score")

    # Verify session transitions appear
    sessions_seen = set()
    for state in states:
        if state.session_name:
            sessions_seen.add(state.session_name)
        assert hasattr(state, "amd_tag")
        assert hasattr(state, "amd_confidence")
        assert state.amd_tag in {
            "NEUTRAL",
            "ACCUMULATION",
            "DISTRIBUTION",
            "MANIPULATION",
        }

    # Should see session transitions over 100-minute range starting at 03:00
    assert len(sessions_seen) > 0, "No sessions detected"

    # Verify modifiers are scaling factors
    for state in states:
        assert (
            0.5 <= state.session_vol_scale <= 2.0
        ), f"Volatility scale out of range: {state.session_vol_scale}"
        assert (
            0.5 <= state.session_liq_scale <= 2.0
        ), f"Liquidity scale out of range: {state.session_liq_scale}"
        assert (
            0.5 <= state.session_risk_scale <= 2.0
        ), f"Risk scale out of range: {state.session_risk_scale}"


def test_market_state_serializable():
    """Test that MarketState with session/flow fields is serializable"""
    builder = MarketStateBuilder("NQ", "1m", lookback=30)
    df = create_sample_ohlcv()

    states = builder.build_states(df, time_causal_check=False)

    # Ensure we can convert to dict (serializable)
    for i, state in enumerate(states[:5]):  # Test first 5
        state_dict = vars(state)
        assert isinstance(state_dict, dict)
        assert "session_name" in state_dict
        assert "vwap" in state_dict
        assert "amd_tag" in state_dict
        assert "amd_confidence" in state_dict
