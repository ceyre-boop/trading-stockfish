"""
Ollama Gateway configuration.
Single choke-point between raw ForexFactory data and the engine.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTBOX_DIR = Path(os.getenv("OLLAMA_OUTBOX", BASE_DIR / "outbox"))
OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
OUTBOX_SNAPSHOT = OUTBOX_DIR / "news_snapshot.json"

RAW_EVENTS_DB = Path(
    os.getenv(
        "OLLAMA_RAW_EVENTS_DB", BASE_DIR.parent / "economic_calendar" / "raw_events.db"
    )
)

# Comma-separated traded assets, filtered before sending to Ollama
TRADED_ASSETS = [
    a.strip()
    for a in os.getenv("OLLAMA_ASSETS", "ES,NQ,SPX,DXY,BTC").split(",")
    if a.strip()
]

ENGINE_ENDPOINT = os.getenv("OLLAMA_ENGINE_ENDPOINT", "")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "news-parser")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "20"))
# Health check endpoint and timeout guard the connectivity proof step.
OLLAMA_HEALTH_ENDPOINT = os.getenv("OLLAMA_HEALTH_ENDPOINT", f"{OLLAMA_HOST}/api/tags")
OLLAMA_HEALTH_TIMEOUT = float(os.getenv("OLLAMA_HEALTH_TIMEOUT", "5"))

CONFIDENCE_FALLBACK = float(os.getenv("OLLAMA_CONFIDENCE_FALLBACK", "0.5"))

IMPACT_LEVELS = {"LOW", "MEDIUM", "HIGH"}
DIRECTIONAL_BIAS = {"UP", "DOWN", "NEUTRAL"}
