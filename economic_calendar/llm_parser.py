"""
Deterministic-ish wrapper around a local LLM (Ollama/LM Studio/GPT4All).

System prompt is fixed; parsing expects JSON only. If the model call fails, a
fallback deterministic payload is returned to preserve pipeline continuity.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import requests

SYSTEM_PROMPT = "You are a deterministic financial news parser. Output ONLY valid JSON."
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))


def _build_prompt(raw_text: str, meta: Dict) -> str:
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Event Title: {meta.get('title','')}\n"
        f"Currency: {meta.get('currency','')}\n"
        f"Impact: {meta.get('impact','')}\n"
        f"Timestamp: {meta.get('timestamp','')}\n"
        f"Source URL: {meta.get('detail_link','')}\n"
        f"Raw Text:\n{raw_text}\n"
        "Return JSON with fields: event_type, impact_level, directional_bias, "
        "confidence, summary, keywords, numeric_extractions, timestamp, source_url."
    )


def _call_model(prompt: str) -> Dict:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    txt = resp.json().get("response", "{}")
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {}


def parse_event(raw_text: str, meta: Dict) -> Dict:
    try:
        prompt = _build_prompt(raw_text, meta)
        parsed = _call_model(prompt)
    except Exception:
        parsed = {}

    if not isinstance(parsed, dict) or not parsed:
        parsed = {
            "event_type": meta.get("title", ""),
            "impact_level": meta.get("impact", ""),
            "directional_bias": "neutral",
            "confidence": 0.0,
            "summary": "",
            "keywords": [],
            "numeric_extractions": {},
            "timestamp": meta.get("timestamp", ""),
            "source_url": meta.get("detail_link", ""),
        }
    return parsed
