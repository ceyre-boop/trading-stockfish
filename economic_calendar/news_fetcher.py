"""
News fetcher: retrieves raw text for a released economic event.
"""

from __future__ import annotations

from typing import Dict

import requests


def fetch_event_text(event: Dict) -> str:
    url = event.get("detail_link") or ""
    if not url:
        return ""
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.text
