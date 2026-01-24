import logging
import os
from datetime import datetime, timezone

BASE_DIR = os.path.join("logs", "session")
os.makedirs(BASE_DIR, exist_ok=True)


def _make_logger(name: str, filename_prefix: str, level=logging.INFO):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = f"{filename_prefix}_{ts}.log"
    path = os.path.join(BASE_DIR, fname)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fh = logging.FileHandler(path, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def get_session_logger():
    return _make_logger("session_context", "session_preview")


def get_capacity_logger():
    return _make_logger("capacity_events", "capacity_preview")


def get_flow_logger():
    return _make_logger("flow_context", "flow_preview")
