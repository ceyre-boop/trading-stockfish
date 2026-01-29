"""
Simple append-only logger for calendar maintenance runs.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "calendar.log"

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def _ensure_logger(level: str = "INFO") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("calendar")
    logger.setLevel(_LEVELS.get(level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_run_start(level: str = "INFO") -> None:
    logger = _ensure_logger(level)
    logger.info("run_start")


def log_scrape_stats(
    fetched: int, inserted: int, deduped: int, level: str = "INFO"
) -> None:
    logger = _ensure_logger(level)
    logger.info("scrape fetched=%s inserted=%s deduped=%s", fetched, inserted, deduped)


def log_parse_stats(normalized: int, level: str = "INFO") -> None:
    logger = _ensure_logger(level)
    logger.info("parse normalized=%s", normalized)


def log_retention_stats(
    purged_past: int, purged_future: int, total_remaining: int, level: str = "INFO"
) -> None:
    logger = _ensure_logger(level)
    logger.info(
        "retention purged_past=%s purged_future=%s remaining=%s",
        purged_past,
        purged_future,
        total_remaining,
    )


def log_error(message: str, level: str = "ERROR") -> None:
    logger = _ensure_logger(level)
    logger.error(message)


def log_run_end(level: str = "INFO") -> None:
    logger = _ensure_logger(level)
    logger.info("run_end")
