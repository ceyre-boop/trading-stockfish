"""
Local Ollama client used exclusively by ollama_gateway.
Engine code must never call Ollama directly.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

import requests

from ollama_gateway import config

LOGGER = logging.getLogger("ollama_client")
if not LOGGER.hasHandlers():
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(handler)

SYSTEM_PROMPT = """
You are a financial news parser. Read the ENTIRE article below, including all expanded sections.

Your ONLY output must be valid JSON with the following fields:

{
    "summary": "...",
    "keywords": ["...", "..."],
    "directional_bias": "UP | DOWN | NEUTRAL",
    "impact": "LOW | MEDIUM | HIGH",
    "numeric_extractions": {
            "inflation_rate": null or number,
            "interest_rate": null or number,
            "gdp_growth": null or number,
            "employment_change": null or number
    },
    "confidence": 0.0 to 1.0
}

Rules:
- JSON ONLY.
- No prose outside JSON.
- No markdown.
- No commentary.
- If a field is not present in the article, set it to null.
- directional_bias must reflect market impact for ES/NQ/SPX/DXY/BTC.
- summary must be concise and factual.
- keywords must be 3–7 core terms.
- confidence reflects certainty in the extraction.
"""


def _build_prompt(raw_text: str, metadata: Dict[str, Any]) -> str:
    meta_lines = [f"{k}: {v}" for k, v in metadata.items() if v is not None]
    meta_block = "\n".join(meta_lines)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Metadata:\n{meta_block}\n\n"
        f"ARTICLE:\n{raw_text}\n"
    )


def check_ollama_health() -> tuple[bool, str]:
    """Ping Ollama to prove connectivity before attempting any parsing."""
    try:
        resp = requests.get(
            config.OLLAMA_HEALTH_ENDPOINT, timeout=config.OLLAMA_HEALTH_TIMEOUT
        )
        resp.raise_for_status()
        return True, ""
    except Exception as exc:  # pragma: no cover - network boundary
        return False, str(exc)


def parse_with_ollama(raw_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _build_prompt(raw_text, metadata)
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/generate",
        json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=config.OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    body = payload.get("response") or payload.get("message", {}).get("content")
    if not body:
        raise ValueError("Ollama response missing content")
    text = str(body).strip()
    if not (text.startswith("{") or text.startswith("[")):
        raise ValueError("Ollama response is not JSON")
    return json.loads(text)
