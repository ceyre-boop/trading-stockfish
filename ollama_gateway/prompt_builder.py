"""
Prompt builder for Ollama JSON parsing.
Creates a deterministic system+user prompt enforcing a strict JSON schema.
"""

from __future__ import annotations

from typing import Any, Dict

SCHEMA_BLOCK = """
You are a deterministic JSON parser. You MUST output exactly one JSON object.
No prose. No markdown. No commentary. No text before or after the JSON.
Your output MUST match this schema exactly:
{
  "summary": "...",
  "keywords": ["..."],
  "directional_bias": "UP | DOWN | NEUTRAL",
  "impact": "LOW | MEDIUM | HIGH",
  "numeric_extractions": { "value": "...", "unit": "..." },
  "confidence": 0-100
}
""".strip()


def build_prompt(raw_text: str, metadata: Dict[str, Any]) -> list[Dict[str, str]]:
    """Return chat messages enforcing the JSON-only contract."""
    meta_lines = [f"{k}: {v}" for k, v in metadata.items() if v is not None]
    meta_block = "\n".join(meta_lines)
    user_block = f"Metadata:\n{meta_block}\n\nARTICLE:\n{raw_text}\n"
    return [
        {"role": "system", "content": SCHEMA_BLOCK},
        {"role": "user", "content": user_block},
    ]
