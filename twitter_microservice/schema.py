"""
JSON schema contract for Twitter microservice outputs.
"""

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class TwitterRecord:
    source: str
    account: str
    tweet_id: str
    timestamp: str
    text: str
    sentiment_score: float
    sentiment_volatility: float
    topic: str
    impact_level: str
    directional_bias: str
    confidence: float

    def to_dict(self) -> Dict:
        return asdict(self)


SCHEMA_EXAMPLE = {
    "source": "twitter",
    "account": "realDonaldTrump",
    "tweet_id": "1234567890",
    "timestamp": "2026-01-28T12:34:56Z",
    "text": "...",
    "sentiment_score": 0.12,
    "sentiment_volatility": 0.05,
    "topic": "rates",
    "impact_level": "HIGH",
    "directional_bias": "UP",
    "confidence": 0.85,
}
