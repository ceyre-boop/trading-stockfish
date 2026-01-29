"""
Parser for Twitter microservice.
- Reads tweets with NULL sentiment_score
- Scores sentiment (VADER + fallback lexicon)
- Classifies topic, impact_level, directional_bias
- Updates SQLite and emits JSON snapshot
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

from twitter_microservice import config, schema

try:  # pragma: no cover - optional dependency
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover
    SentimentIntensityAnalyzer = None  # type: ignore

POS_LEX = {"strong", "growth", "up", "beat", "record", "optimistic", "win"}
NEG_LEX = {"weak", "down", "cut", "fear", "tariff", "sanction", "loss", "risk"}


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    return conn


def _fetch_pending(conn: sqlite3.Connection, limit: int) -> List[Dict]:
    rows = conn.execute(
        """
        SELECT id, account, tweet_id, timestamp, text
        FROM tweets
        WHERE sentiment_score IS NULL
        ORDER BY timestamp ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "account": r[1],
            "tweet_id": r[2],
            "timestamp": r[3],
            "text": r[4],
        }
        for r in rows
    ]


def _classify_topic(text_l: str) -> str:
    for topic, kws in config.TOPIC_KEYWORDS.items():
        if any(kw in text_l for kw in kws):
            return topic
    return "other"


def _classify_impact(account: str, topic: str) -> str:
    acct = account.lower()
    if acct in config.IMPACT_BY_ACCOUNT:
        return config.IMPACT_BY_ACCOUNT[acct]
    if topic in {"rates", "tariffs"}:
        return "HIGH"
    return "MEDIUM"


def _score_sentiment(text: str, analyzer) -> float:
    if analyzer:
        try:
            return float(analyzer.polarity_scores(text).get("compound", 0.0))
        except Exception:
            pass
    text_l = text.lower()
    bull = sum(1 for w in POS_LEX if w in text_l)
    bear = sum(1 for w in NEG_LEX if w in text_l)
    if bull == bear == 0:
        return 0.0
    return max(-1.0, min(1.0, (bull - bear) / 5.0))


def _directional_bias(topic: str, score: float) -> str:
    sentiment_dir = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    mapping = config.TOPIC_BIAS.get(topic, {})
    if sentiment_dir == "neutral":
        return "NEUTRAL"
    return mapping.get(sentiment_dir, "NEUTRAL")


def _update(conn: sqlite3.Connection, rows: List[Dict]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        UPDATE tweets
        SET sentiment_score = ?, sentiment_volatility = ?, topic = ?, impact_level = ?,
            directional_bias = ?, confidence = ?
        WHERE id = ?
        """,
        [
            (
                r["sentiment_score"],
                r["sentiment_volatility"],
                r["topic"],
                r["impact_level"],
                r["directional_bias"],
                r["confidence"],
                r["id"],
            )
            for r in rows
        ],
    )
    conn.commit()


def _latest_snapshot(conn: sqlite3.Connection, limit: int) -> Dict:
    rows = conn.execute(
        """
        SELECT account, tweet_id, timestamp, text, sentiment_score, sentiment_volatility,
               topic, impact_level, directional_bias, confidence
        FROM tweets
        WHERE sentiment_score IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    records = [
        schema.TwitterRecord(
            source="twitter",
            account=r[0],
            tweet_id=r[1],
            timestamp=r[2],
            text=r[3],
            sentiment_score=float(r[4]),
            sentiment_volatility=float(r[5]),
            topic=r[6],
            impact_level=r[7],
            directional_bias=r[8],
            confidence=float(r[9]),
        ).to_dict()
        for r in rows
    ]
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "source": "twitter",
        "count": len(records),
        "records": records,
    }


def process(db_path: str, limit: int, snapshot_limit: int) -> Dict:
    conn = _connect(db_path)
    analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
    pending = _fetch_pending(conn, limit)
    enriched: List[Dict] = []
    scores: List[float] = []
    for row in pending:
        text = row["text"]
        score = _score_sentiment(text, analyzer)
        topic = _classify_topic(text.lower())
        impact = _classify_impact(row["account"], topic)
        bias = _directional_bias(topic, score)
        scores.append(score)
        enriched.append(
            {
                **row,
                "sentiment_score": score,
                "sentiment_volatility": 0.0,
                "topic": topic,
                "impact_level": impact,
                "directional_bias": bias,
                "confidence": min(1.0, abs(score)),
            }
        )
    _update(conn, enriched)

    # simple run-level volatility proxy
    if scores:
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        stdev = math.sqrt(var)
        conn.execute(
            """
            UPDATE tweets
            SET sentiment_volatility = ?
            WHERE id IN (SELECT id FROM tweets WHERE sentiment_score IS NOT NULL ORDER BY timestamp DESC LIMIT ?)
            """,
            (float(stdev), snapshot_limit),
        )
        conn.commit()

    return _latest_snapshot(conn, snapshot_limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pending tweets")
    parser.add_argument("--db", default=config.DATABASE_FILE)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--snapshot", type=int, default=100)
    args = parser.parse_args()

    snapshot = process(args.db, args.limit, args.snapshot)
    print(json.dumps(snapshot, indent=2))
