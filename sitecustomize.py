"""
Compatibility shims for environment-wide imports.

Current need: Pandas 3.0 made frequency aliases case-sensitive, but the
codebase/tests still pass uppercase strings like "1H". This hook normalizes
frequency strings so legacy aliases keep working without touching test logic.
"""

from __future__ import annotations

try:
    from pandas._libs import tslibs as _tslibs
except Exception:  # pandas not installed or unexpected layout
    _tslibs = None


def _normalize_freq(freq: object) -> object:
    """Lowercase string frequencies while leaving other types untouched."""
    if isinstance(freq, str):
        return freq.lower()
    return freq


if _tslibs is not None:
    try:
        _offsets = _tslibs.offsets
        _original_to_offset = _offsets.to_offset
    except Exception:
        _offsets = None
        _original_to_offset = None

    if _offsets is not None and _original_to_offset is not None:

        def _to_offset_compat(freq, *args, **kwargs):
            """Wrapper that retries with lowercase for backward compatibility."""
            try:
                return _original_to_offset(freq, *args, **kwargs)
            except Exception:
                normalized = _normalize_freq(freq)
                if normalized != freq:
                    return _original_to_offset(normalized, *args, **kwargs)
                raise

        _offsets.to_offset = _to_offset_compat

        # Keep python-level helper in sync
        try:
            from pandas.tseries import frequencies as _freq_mod
        except Exception:
            _freq_mod = None

        if _freq_mod is not None:
            _freq_mod.to_offset = _to_offset_compat

        # Expose helper for observability/debugging if needed
        TO_OFFSET_COMPAT_APPLIED = True
    else:
        TO_OFFSET_COMPAT_APPLIED = False
else:
    TO_OFFSET_COMPAT_APPLIED = False
