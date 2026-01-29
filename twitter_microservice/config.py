"""
Configuration for the Twitter microservice.
Deterministic defaults; override via environment variables.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATABASE_FILE = os.getenv("TWITTER_DB", str(BASE_DIR / "twitter_news.db"))
ACCOUNTS = [
    acc.strip()
    for acc in os.getenv(
        "TWITTER_ACCOUNTS",
        "realDonaldTrump,POTUS,FederalReserve,CNBC,Reuters,TheTerminal",
    ).split(",")
    if acc.strip()
]
FETCH_LIMIT_PER_ACCOUNT = int(os.getenv("TWITTER_FETCH_LIMIT", "20"))
POLL_INTERVAL_SECONDS = int(os.getenv("TWITTER_POLL_INTERVAL", "120"))
ROUTER_OUTBOX = Path(os.getenv("TWITTER_OUTBOX", BASE_DIR / "outbox"))
ROUTER_OUTBOX.mkdir(parents=True, exist_ok=True)
ENGINE_ENDPOINT = os.getenv("STOCKFISH_NEWS_ENDPOINT", "")
USER_AGENT = os.getenv(
    "TWITTER_UA",
    "StockfishTrade-TwitterMicroservice/1.0 (+deterministic)",
)

# Topic classification keywords (lowercase)
TOPIC_KEYWORDS = {
    "rates": ["rate", "fomc", "fed", "yields", "hike", "cut"],
    "tariffs": ["tariff", "tariffs", "trade war", "duties", "sanction"],
    "jobs": ["jobs", "employment", "unemployment", "payroll"],
    "crypto": ["crypto", "bitcoin", "ethereum", "btc", "eth"],
    "china": ["china", "beijing", "cny"],
    "politics": ["election", "congress", "senate", "house", "white house", "president"],
}

IMPACT_BY_ACCOUNT = {
    "realdonaldtrump": "HIGH",
    "potus": "HIGH",
    "federalreserve": "HIGH",
    "ecb": "HIGH",
    "bankofengland": "HIGH",
    "treasurypress": "MEDIUM",
    "cnbc": "MEDIUM",
    "reuters": "MEDIUM",
    "theterminal": "MEDIUM",
}

# Directional bias mapping by topic + sentiment sign
TOPIC_BIAS = {
    "rates": {"positive": "DOWN", "negative": "UP"},  # dovish rates -> equities up
    "tariffs": {"positive": "UP", "negative": "DOWN"},
    "jobs": {"positive": "UP", "negative": "DOWN"},
    "crypto": {"positive": "UP", "negative": "DOWN"},
    "china": {"positive": "UP", "negative": "DOWN"},
    "politics": {"positive": "NEUTRAL", "negative": "NEUTRAL"},
}
