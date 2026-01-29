"""
Environment loader for Trading Stockfish.
Reads .env and exposes required credentials without logging secrets.
"""

from __future__ import annotations

import os
from typing import Dict

ENV_KEYS = [
    "POLYGON_API_KEY",
    "MT5_LOGIN",
    "MT5_PASSWORD",
    "MT5_SERVER",
]


def _parse_dotenv(path: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not os.path.exists(path):
        return values
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value
    return values


def load_env(dotenv_path: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from .env and os.environ.

    Returns a dict containing POLYGON_API_KEY, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    when present. Secrets are not logged or printed.
    """

    env_values = _parse_dotenv(dotenv_path)
    # Overlay with process environment to allow overrides
    for key in ENV_KEYS:
        if key in os.environ:
            env_values[key] = os.environ[key]
    return {key: env_values.get(key) for key in ENV_KEYS}


__all__ = ["load_env", "ENV_KEYS"]
