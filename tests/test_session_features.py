import datetime as _dt

from engine.market_state_builder import SessionFeatureTracker


def test_session_features_accumulate_within_session():
    tracker = SessionFeatureTracker()
    session = "ASIA"

    metrics = tracker.compute(
        prices=[100.0, 101.5, 99.5],
        sessions=[session, session, session],
        fallback_session=session,
    )

    assert metrics["session_high"] == 101.5
    assert metrics["session_low"] == 99.5
    assert metrics["session_range"] == 101.5 - 99.5


def test_session_features_reset_on_new_session():
    tracker = SessionFeatureTracker()
    prices = [100.0, 101.0, 110.0, 108.0]
    sessions = ["LONDON", "LONDON", "NEW_YORK", "NEW_YORK"]

    metrics = tracker.compute(
        prices=prices,
        sessions=sessions,
        fallback_session="LONDON",
    )

    assert tracker.last_session == "NEW_YORK"
    assert metrics["session_high"] == 110.0
    assert metrics["session_low"] == 108.0
    assert metrics["session_range"] == 2.0
