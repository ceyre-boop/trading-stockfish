"""
Unified, engine-safe schema for news parsed by the Ollama Gateway.
This is the only schema the engine ingests.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class NewsRecord:
    source: str  # forex
    origin_id: str
    timestamp: str
    asset_scope: List[str] = field(default_factory=list)
    event_type: str = "OTHER"
    impact_level: str = "MEDIUM"
    directional_bias: str = "NEUTRAL"
    confidence: float = 0.0
    sentiment_score: float = 0.0
    sentiment_volatility: float = 0.0
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    numeric_extractions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NewsSnapshot:
    generated_at: str
    future_events: List[Dict[str, Any]]
    parsed_events: List[NewsRecord]
    aggregated_scores: Dict[str, float] = field(default_factory=dict)
    ollama_unreachable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        parsed = [
            r.to_dict() if isinstance(r, NewsRecord) else r for r in self.parsed_events
        ]
        return {
            "generated_at": self.generated_at,
            "future_events": self.future_events,
            "parsed_events": parsed,
            "aggregated_scores": self.aggregated_scores,
            "record_count": len(parsed),
            "future_count": len(self.future_events),
            "ollama_unreachable": self.ollama_unreachable,
        }


ENGINE_SAFE_SCHEMA = {
    "source": "forex",
    "origin_id": "string",
    "timestamp": "ISO8601",
    "asset_scope": ["ES", "NQ", "EURUSD"],
    "event_type": "CPI|FOMC|NFP|PMI|GDP|EARNINGS|OTHER",
    "impact_level": "LOW|MEDIUM|HIGH",
    "directional_bias": "UP|DOWN|NEUTRAL",
    "confidence": "float 0.0-1.0",
    "sentiment_score": "float -1.0-1.0",
    "sentiment_volatility": "float 0.0-1.0",
    "summary": "short text",
    "keywords": ["string"],
    "numeric_extractions": {"key": "float"},
    "future_events": [
        {
            "origin_id": "string",
            "event_name": "string",
            "timestamp": "ISO8601",
            "time_delta_minutes": "float",
            "impact_level": "LOW|MEDIUM|HIGH",
            "event_type": "CPI|FOMC|NFP|PMI|GDP|OTHER",
            "asset_scope": ["string"],
            "risk_window": "bool",
            "macro_pressure_score": "float 0-1",
        }
    ],
    "parsed_events": ["NewsRecord"],
    "aggregated_scores": {
        "news_sentiment_score": "float",
        "news_sentiment_volatility": "float",
        "news_macro_impact": "float",
        "news_directional_bias": "float",
        "news_confidence": "float",
    },
    "ollama_unreachable": False,
}
