"""
Deterministic RSS fetcher for the sentiment pipeline.

collect_snapshot() stays stateless for replay/testing. ingest_once() persists
into SQLite with deduplication, and run_loop() mirrors the original polling
daemon (every N seconds) so a second terminal can read from the DB.
"""

import concurrent.futures
import logging
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import requests

try:
    import feedparser
except ImportError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "feedparser is required for rss_fetcher.py (pip install feedparser)"
    ) from exc


LOGGER = logging.getLogger("sentiment_rss_fetcher")
if not LOGGER.hasHandlers():
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(handler)


_ENV_FEEDS = {
    "WSJ RHS": os.getenv("FEED_URL_1", ""),
    "FT UK Economy": os.getenv("FEED_URL_2", ""),
    "Reuters Markets": os.getenv("FEED_URL_3", ""),
}

DEFAULT_FEEDS: Dict[str, str] = {
    **{k: v for k, v in _ENV_FEEDS.items() if v},
    "Bloomberg Markets": "https://feeds.bloomberg.com/markets/news.rss",
    "Bloomberg Politics": "https://feeds.bloomberg.com/politics/news.rss",
    "Investing.com Breaking": "https://www.investing.com/rss/news_285.rss",
    "CNBC Top News": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Yahoo Finance Top Stories": "https://finance.yahoo.com/news/rss",
}

DATABASE_FILE = os.getenv("DATABASE_FILE", "news_database.db")


@dataclass(frozen=True)
class Article:
    source: str
    headline: str
    link: str
    published_iso: str
    summary: Optional[str] = None


def _strip_html(text: str) -> str:
    # Light-weight tag removal to keep deterministic output without a heavy parser
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _connect_db(db_path: str = DATABASE_FILE) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            link TEXT NOT NULL UNIQUE,
            published_date TEXT NOT NULL,
            summary TEXT,
            sentiment_score REAL
        );
        """
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_articles_published
        ON articles(published_date);
        """
    )
    conn.commit()
    return conn


def _coerce_published(entry: dict) -> str:
    try:
        ts = entry.get("published_parsed")
        if ts:
            return datetime(*ts[:6], tzinfo=timezone.utc).isoformat()
    except Exception:
        pass
    return datetime.now(tz=timezone.utc).isoformat()


def _parse_entry_summary(entry: dict) -> Optional[str]:
    summary = None
    try:
        summary = entry.get("summary") or entry.get("description")
    except Exception:
        summary = None

    if not summary:
        try:
            content = entry.get("content")
            if content and isinstance(content, list) and len(content) > 0:
                summary = content[0].get("value")
        except Exception:
            summary = None

    return _strip_html(summary) if summary else None


def _persist_articles(
    conn: sqlite3.Connection, articles: List[Article]
) -> Tuple[int, int]:
    inserted = 0
    skipped = 0
    sql = """
        INSERT OR IGNORE INTO articles
        (title, source, link, published_date, summary)
        VALUES (?, ?, ?, ?, ?);
    """
    for art in articles:
        try:
            cur = conn.execute(
                sql,
                (
                    art.headline,
                    art.source,
                    art.link,
                    art.published_iso,
                    art.summary,
                ),
            )
            if cur.rowcount:
                inserted += 1
            else:
                skipped += 1
        except sqlite3.IntegrityError:
            skipped += 1
    conn.commit()
    return inserted, skipped


def fetch_feed(source_name: str, url: str, max_items: int = 50) -> List[Article]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    LOGGER.info("Fetching feed %s", source_name)
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)
    articles: List[Article] = []
    for entry in feed.entries[:max_items]:
        articles.append(
            Article(
                source=source_name,
                headline=entry.get("title", "N/A"),
                link=entry.get("link", "N/A"),
                published_iso=_coerce_published(entry),
                summary=_parse_entry_summary(entry),
            )
        )
    return articles


def fetch_feeds(
    feeds: Optional[Dict[str, str]] = None,
    max_items_per_feed: int = 50,
    max_workers: Optional[int] = None,
) -> List[Dict]:
    """
    Fetch multiple RSS feeds concurrently with bounded workers.

    Returns a list of article dictionaries sorted by published time (oldest first)
    to improve deterministic replay behavior.
    """

    selected_feeds = feeds or DEFAULT_FEEDS
    workers = max_workers or min(4, len(selected_feeds)) or 1

    results: List[Article] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(fetch_feed, name, url, max_items_per_feed): name
            for name, url in selected_feeds.items()
        }
        for future in concurrent.futures.as_completed(future_map):
            try:
                results.extend(future.result())
            except Exception as exc:
                LOGGER.warning("Feed %s failed: %s", future_map[future], exc)

    results.sort(key=lambda a: a.published_iso)
    return [asdict(article) for article in results]


def collect_snapshot(
    feeds: Optional[Dict[str, str]] = None,
    max_items_per_feed: int = 50,
    max_workers: Optional[int] = None,
) -> Dict:
    """
    Deterministic wrapper returning a fixed JSON schema.
    """

    articles = fetch_feeds(
        feeds=feeds, max_items_per_feed=max_items_per_feed, max_workers=max_workers
    )
    snapshot = {
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_count": len(feeds or DEFAULT_FEEDS),
        "article_count": len(articles),
        "articles": articles,
    }
    return snapshot


def ingest_once(
    db_path: str = DATABASE_FILE,
    feeds: Optional[Dict[str, str]] = None,
    max_items_per_feed: int = 50,
    max_workers: Optional[int] = None,
) -> Dict:
    """
    Fetch feeds, persist to SQLite with deduplication, and return stats.
    """

    snapshot = collect_snapshot(
        feeds=feeds, max_items_per_feed=max_items_per_feed, max_workers=max_workers
    )
    conn = _connect_db(db_path)
    inserted, skipped = _persist_articles(
        conn, [Article(**a) for a in snapshot["articles"]]
    )
    return {
        "db_path": db_path,
        "fetched": snapshot["article_count"],
        "inserted": inserted,
        "skipped": skipped,
    }


def run_loop(
    interval_seconds: int = 90,
    db_path: str = DATABASE_FILE,
    feeds: Optional[Dict[str, str]] = None,
    max_items_per_feed: int = 50,
    max_workers: Optional[int] = None,
) -> None:
    while True:
        stats = ingest_once(
            db_path=db_path,
            feeds=feeds,
            max_items_per_feed=max_items_per_feed,
            max_workers=max_workers,
        )
        LOGGER.info(
            "Fetched %s articles, inserted=%s, skipped=%s",
            stats["fetched"],
            stats["inserted"],
            stats["skipped"],
        )
        time.sleep(interval_seconds)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="RSS fetcher with SQLite persistence and optional loop",
    )
    parser.add_argument("--db", default=DATABASE_FILE, help="SQLite db path")
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--loop", action="store_true", help="Run forever with sleep")
    parser.add_argument(
        "--interval", type=int, default=90, help="Sleep seconds when looping"
    )
    args = parser.parse_args()

    if args.loop:
        run_loop(
            interval_seconds=args.interval,
            db_path=args.db,
            max_items_per_feed=args.max_items,
            max_workers=args.workers,
        )
    else:
        stats = ingest_once(
            db_path=args.db,
            max_items_per_feed=args.max_items,
            max_workers=args.workers,
        )
        print(json.dumps(stats, indent=2))
