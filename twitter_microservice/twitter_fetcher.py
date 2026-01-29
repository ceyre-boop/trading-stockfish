"""
Deterministic Twitter fetcher.
- Uses curated account list from config.
- Inserts tweets into SQLite with sentiment_score NULL for unprocessed rows.
- Provides one-shot or looped execution.

This is an offline-friendly stub: replace `_fetch_account_feed` with a real
client when available; it must return stable dicts to keep determinism.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from twitter_microservice import config

LOGGER = logging.getLogger("twitter_fetcher")
if not LOGGER.hasHandlers():
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(handler)


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account TEXT NOT NULL,
            tweet_id TEXT NOT NULL UNIQUE,
            timestamp TEXT NOT NULL,
            text TEXT NOT NULL,
            sentiment_score REAL,
            sentiment_volatility REAL,
            topic TEXT,
            impact_level TEXT,
            directional_bias TEXT,
            confidence REAL
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tweets_ts ON tweets(timestamp);")
    conn.commit()
    return conn


def _fetch_account_feed(account: str, limit: int) -> List[Dict]:
    """
    Deterministic stub feed.
    Replace with real API wrapper that returns list of dicts:
    {"tweet_id", "timestamp", "text", "account"}
    """

    now = datetime.now(tz=timezone.utc)
    base_ts = now.replace(microsecond=0).isoformat()
    sample = [
        {
            "tweet_id": f"{account}-rates-{base_ts}",
            "timestamp": base_ts,
            "text": f"{account} notes rate path looks steady; watching inflation closely.",
            "account": account,
        },
        {
            "tweet_id": f"{account}-jobs-{base_ts}",
            "timestamp": base_ts,
            "text": f"{account} highlights strong payroll momentum and wage gains.",
            "account": account,
        },
    ]
    return sample[:limit]


def _insert_tweets(conn: sqlite3.Connection, tweets: Iterable[Dict]) -> Dict[str, int]:
    sql = """
        INSERT OR IGNORE INTO tweets
        (account, tweet_id, timestamp, text, sentiment_score, sentiment_volatility,
         topic, impact_level, directional_bias, confidence)
        VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL);
    """
    inserted = 0
    skipped = 0
    for t in tweets:
        try:
            cur = conn.execute(
                sql,
                (
                    t.get("account", ""),
                    t.get("tweet_id", ""),
                    t.get("timestamp", ""),
                    t.get("text", ""),
                ),
            )
            inserted += cur.rowcount
        except sqlite3.IntegrityError:
            skipped += 1
    conn.commit()
    return {"inserted": inserted, "skipped": skipped}


def fetch_once(db_path: str, accounts: List[str], limit_per_account: int) -> Dict:
    conn = _connect(db_path)
    total_inserted = 0
    total_skipped = 0
    for acc in accounts:
        feed = _fetch_account_feed(acc, limit_per_account)
        stats = _insert_tweets(conn, feed)
        total_inserted += stats["inserted"]
        total_skipped += stats["skipped"]
    return {
        "db_path": db_path,
        "accounts": len(accounts),
        "inserted": total_inserted,
        "skipped": total_skipped,
    }


def fetch_loop(
    db_path: str, accounts: List[str], limit_per_account: int, interval: int
) -> None:
    while True:
        stats = fetch_once(db_path, accounts, limit_per_account)
        LOGGER.info(
            "Fetched accounts=%s inserted=%s skipped=%s",
            stats["accounts"],
            stats["inserted"],
            stats["skipped"],
        )
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic Twitter fetcher")
    parser.add_argument("--db", default=config.DATABASE_FILE)
    parser.add_argument("--accounts", nargs="*", default=config.ACCOUNTS)
    parser.add_argument("--limit", type=int, default=config.FETCH_LIMIT_PER_ACCOUNT)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=config.POLL_INTERVAL_SECONDS)
    args = parser.parse_args()

    if not args.accounts:
        LOGGER.error("No accounts configured; exiting")
        sys.exit(1)

    if args.loop:
        fetch_loop(args.db, args.accounts, args.limit, args.interval)
    else:
        result = fetch_once(args.db, args.accounts, args.limit)
        print(result)
