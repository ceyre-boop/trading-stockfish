"""Connector validation for SIM and PAPER modes.

Deterministically instantiates adapters for SIMULATION and PAPER, checks that
no LIVE endpoints are marked, initialization succeeds, and a heartbeat/status
call works without errors. Writes log to logs/system/connector_validation_YYYYMMDD.log
and exits non-zero on any failure.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Dict, List

from engine.modes import Mode, get_adapter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "system"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"connector_validation_{dt.datetime.utcnow():%Y%m%d}.log"


class Logger:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def log(self, msg: str) -> None:
        ts = dt.datetime.utcnow().isoformat() + "Z"
        line = f"{ts} {msg}"
        print(line)
        self.lines.append(line)

    def write(self) -> None:
        LOG_PATH.write_text("\n".join(self.lines), encoding="utf-8")


def _heartbeat(adapter) -> Dict[str, object]:
    try:
        return adapter.place_order(None)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"heartbeat failed: {exc}")


def _validate_mode(mode: Mode, logger: Logger) -> List[str]:
    errors: List[str] = []
    logger.log(f"Validating mode={mode.value}")
    try:
        adapter = get_adapter(mode)
        logger.log(f"Adapter created: name={adapter.name} live={adapter.live} disabled={adapter.disabled}")
    except Exception as exc:
        errors.append(f"adapter init failed for {mode.value}: {exc}")
        return errors

    if adapter.live:
        errors.append(f"adapter for {mode.value} unexpectedly marked live")

    # Heartbeat/status check
    try:
        resp = _heartbeat(adapter)
        logger.log(f"Heartbeat response: {resp}")
    except Exception as exc:
        errors.append(f"heartbeat failed for {mode.value}: {exc}")

    # SAFE_MODE should not block SIM/PAPER init; check disabled flag
    if adapter.disabled:
        errors.append(f"adapter for {mode.value} is disabled; SAFE_MODE should not block init")

    return errors


def main() -> None:
    logger = Logger()
    all_errors: List[str] = []

    for mode in (Mode.SIMULATION, Mode.PAPER):
        all_errors.extend(_validate_mode(mode, logger))

    if all_errors:
        logger.log("FAIL: connector validation failed")
        for err in all_errors:
            logger.log(f" - {err}")
        logger.write()
        sys.exit(1)

    logger.log("PASS: connectors initialize cleanly for SIM and PAPER")
    logger.write()
    sys.exit(0)


if __name__ == "__main__":
    main()
