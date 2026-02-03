import datetime as _dt

from engine.market_state_builder import build_market_state
from engine.session_regimes import classify_session


def test_classify_session_boundaries():
    tz = _dt.timezone.utc
    cases = [
        ("ASIA", _dt.datetime(2024, 1, 2, 1, 30, tzinfo=tz)),
        ("PRE_SESSION", _dt.datetime(2024, 1, 2, 5, 30, tzinfo=tz)),
        ("LONDON", _dt.datetime(2024, 1, 2, 8, 30, tzinfo=tz)),
        ("LONDON_NY_OVERLAP", _dt.datetime(2024, 1, 2, 12, 30, tzinfo=tz)),
        ("NY", _dt.datetime(2024, 1, 2, 17, 15, tzinfo=tz)),
        ("POST_SESSION", _dt.datetime(2024, 1, 2, 22, 15, tzinfo=tz)),
    ]

    for expected, ts in cases:
        assert classify_session(ts) == expected


def test_weekend_no_session():
    tz = _dt.timezone.utc
    saturday = _dt.datetime(2024, 1, 6, 10, 0, tzinfo=tz)
    sunday = _dt.datetime(2024, 1, 7, 10, 0, tzinfo=tz)
    assert classify_session(saturday) == "NO_SESSION"
    assert classify_session(sunday) == "NO_SESSION"


def test_market_state_includes_session_regime():
    ts = _dt.datetime(2024, 1, 2, 9, 0, tzinfo=_dt.timezone.utc).timestamp()
    state = build_market_state(symbol="TEST", order_book_events=[], timestamp=ts)

    assert "session_regime" in state
    allowed = {
        "ASIA",
        "PRE_SESSION",
        "LONDON",
        "LONDON_NY_OVERLAP",
        "NY",
        "POST_SESSION",
        "NO_SESSION",
        # Legacy labels (backward compatibility)
        "NEW_YORK",
        "OFF_HOURS",
    }
    assert str(state["session_regime"]) in allowed
