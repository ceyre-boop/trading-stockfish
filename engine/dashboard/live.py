"""Minimal live dashboard stub for Phase 11.

CLI entry: python -m engine.dashboard.live
Displays placeholder health summary from log presence.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    safe_mode = "UNKNOWN"
    safe_file = PROJECT_ROOT / "logs" / "safe_mode_state.txt"
    if safe_file.exists():
        safe_mode = safe_file.read_text(encoding="utf-8").strip()

    weekly_review = PROJECT_ROOT / "logs" / "scheduled" / "weekly_cycle_review.log"
    weekly_status = "missing"
    if weekly_review.exists():
        try:
            data = json.loads(weekly_review.read_text(encoding="utf-8"))
            weekly_status = "errors" if data.get("errors") else "ok"
        except Exception:
            weekly_status = "unreadable"

    print("=== Live Dashboard (stub) ===")
    print(f"SAFE_MODE: {safe_mode}")
    print(f"Weekly review: {weekly_status}")
    print("Connector health: see logs/system/")
    print("Guardrails: see engine.guardrails runtime checks")


if __name__ == "__main__":
    main()
