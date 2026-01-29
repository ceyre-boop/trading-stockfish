import os

import pytest

from engine.canonical_stack_validator import validate_canonical_stack


def test_canonical_validator_requires_volatility_shock_flag(monkeypatch):
    monkeypatch.setenv("OFFICIAL_MODE", "1")
    with pytest.raises(ValueError):
        validate_canonical_stack(
            context="test",
            use_causal=True,
            volatility_shock_present=False,
        )
    monkeypatch.delenv("OFFICIAL_MODE", raising=False)


def test_canonical_validator_requires_swing_structure(monkeypatch):
    monkeypatch.setenv("OFFICIAL_MODE", "1")
    with pytest.raises(ValueError):
        validate_canonical_stack(
            context="test",
            use_causal=True,
            swing_structure_present=False,
        )
    monkeypatch.delenv("OFFICIAL_MODE", raising=False)
