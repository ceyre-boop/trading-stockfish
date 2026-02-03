"""
Local Ollama client used exclusively by ollama_gateway.
Engine code must never call Ollama directly.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

import requests

from ollama_gateway import config, prompt_builder

LOGGER = logging.getLogger("ollama_client")
if not LOGGER.hasHandlers():
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(handler)


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


def _post_chat(messages: list[Dict[str, str]]) -> str:
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": config.OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"num_predict": 512, "temperature": 0},
        },
        timeout=config.OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content") or ""


def _post_generate(prompt: str) -> str:
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/generate",
        json={
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"num_predict": 512, "temperature": 0},
        },
        timeout=config.OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("response") or payload.get("message", {}).get("content") or ""


def parse_with_ollama(raw_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    messages = prompt_builder.build_prompt(raw_text, metadata)
    # Prefer chat endpoint; fallback to generate if chat 404s.
    body = ""
    try:
        body = _post_chat(messages)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            # Concatenate into a single prompt for generate fallback.
            joined = "\n\n".join(m.get("content", "") for m in messages)
            body = _post_generate(joined)
        else:
            raise

    if not body:
        raise ValueError("Ollama response missing content")

    text = str(body).strip()

    def _clean_json(text_in: str) -> str:
        stripped = text_in.strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Ollama response is not JSON: {stripped[:400]}")
        before = stripped[:start].strip()
        after = stripped[end + 1 :].strip()
        if before or after:
            raise ValueError(
                f"Ollama response contains non-JSON content: {stripped[:200]}"
            )
        candidate = stripped[start : end + 1]
        # Remove trailing commas before closing braces/brackets.
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        candidate = candidate.strip()
        if not (candidate.startswith("{") and candidate.endswith("}")):
            raise ValueError(f"Ollama response is not a JSON object: {candidate[:400]}")
        return candidate

    cleaned = _clean_json(text)
    return json.loads(cleaned)
