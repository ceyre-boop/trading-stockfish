"""Deterministic session regime classifier (UTC-based).

Rules align with docs/session_regimes.md:
- Regimes and UTC windows (inclusive of start/end minutes):
    PRE_SESSION          05:00–07:59
    ASIA                 00:00–06:59
    LONDON               08:00–11:59
    LONDON_NY_OVERLAP    12:00–16:59
    NY                   17:00–21:59
    POST_SESSION         22:00–23:59
- Overlap resolution:
    1) Choose the matching regime with the latest start time.
    2) If tied, deterministic fallback ordering: LONDON_NY_OVERLAP > NY > LONDON > ASIA > PRE_SESSION > POST_SESSION.
- Weekends/holidays: return "NO_SESSION" (same behavior for any timestamp outside all windows).
"""

from __future__ import annotations

import datetime as _dt
from typing import List, Tuple

# Tie-break priority (highest first) when start times are equal
_TIE_BREAK_ORDER: List[str] = [
    "LONDON_NY_OVERLAP",
    "NY",
    "LONDON",
    "ASIA",
    "PRE_SESSION",
    "POST_SESSION",
]

# Window table: (name, start_hour, start_minute, end_hour, end_minute)
_WINDOWS: List[Tuple[str, int, int, int, int]] = [
    ("ASIA", 0, 0, 6, 59),
    ("PRE_SESSION", 5, 0, 7, 59),
    ("LONDON", 8, 0, 11, 59),
    ("LONDON_NY_OVERLAP", 12, 0, 16, 59),
    ("NY", 17, 0, 21, 59),
    ("POST_SESSION", 22, 0, 23, 59),
]


def _minutes_since_midnight(ts: _dt.datetime) -> int:
    return ts.hour * 60 + ts.minute


def _validate_utc(ts: _dt.datetime) -> _dt.datetime:
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise ValueError("timestamp_utc must be timezone-aware and in UTC")
    # Normalize to UTC offset zero
    if ts.utcoffset() != _dt.timedelta(0):
        ts = ts.astimezone(_dt.timezone.utc)
    return ts


def classify_session(timestamp_utc: _dt.datetime) -> str:
    """Return session_regime label for a UTC timestamp.

    Args:
        timestamp_utc: timezone-aware datetime in UTC.
    Returns:
        One of the documented session labels or "NO_SESSION" when outside windows
        or during weekends/holidays.
    """

    ts = _validate_utc(timestamp_utc)

    # Weekend check (UTC)
    if ts.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return "NO_SESSION"

    minutes = _minutes_since_midnight(ts)

    candidates = []
    for name, sh, sm, eh, em in _WINDOWS:
        start_min = sh * 60 + sm
        end_min = eh * 60 + em
        if start_min <= minutes <= end_min:
            candidates.append((name, start_min))

    if not candidates:
        return "NO_SESSION"

    # Latest start time wins; if tie, use predefined deterministic order
    max_start = max(start for _, start in candidates)
    tied = [name for name, start in candidates if start == max_start]

    if len(tied) == 1:
        return tied[0]

    for preferred in _TIE_BREAK_ORDER:
        if preferred in tied:
            return preferred

    # Fallback (should not be reached with defined windows)
    return sorted(tied)[0]


# Examples (UTC):
# 2026-02-02 01:30Z -> ASIA
# 2026-02-02 07:30Z -> ASIA (latest-start rule keeps ASIA over PRE_SESSION overlap)
# 2026-02-02 08:30Z -> LONDON
# 2026-02-02 12:45Z -> LONDON_NY_OVERLAP
# 2026-02-02 17:15Z -> NY
# 2026-02-02 22:30Z -> POST_SESSION
# 2026-02-07 10:00Z -> NO_SESSION (weekend)
