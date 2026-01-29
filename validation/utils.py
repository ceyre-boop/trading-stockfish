"""
Validation utility functions for logging, diagnostics, and reporting.
"""

import json
import logging


def setup_validation_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )


from typing import Any, List


def save_diagnostics_report(diagnostics: List[Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(diagnostics, f, indent=2)


def compare_diagnostics(live: List[Any], replay: List[Any]) -> dict:
    # Simple comparison: count mismatches and summarize
    len_live = len(live)
    len_replay = len(replay)

    report = {
        "total": max(len_live, len_replay),
        "mismatches": 0,
        "details": [],
        "live_total": len_live,
        "replay_total": len_replay,
        "length_mismatch": len_live != len_replay,
    }

    # Compare items up to the length of the shorter list
    min_len = min(len_live, len_replay)
    for i in range(min_len):
        live_item = live[i]
        replay_item = replay[i]
        if live_item != replay_item:
            report["mismatches"] += 1
            report["details"].append(
                {"index": i, "live": live_item, "replay": replay_item}
            )

    # If lengths differ, record mismatches for extra items in the longer list
    if len_live > min_len:
        for i in range(min_len, len_live):
            live_item = live[i]
            report["mismatches"] += 1
            report["details"].append(
                {"index": i, "live": live_item, "replay": None}
            )
    elif len_replay > min_len:
        for i in range(min_len, len_replay):
            replay_item = replay[i]
            report["mismatches"] += 1
            report["details"].append(
                {"index": i, "live": None, "replay": replay_item}
            )
    return report


def print_comparison_report(report: dict) -> None:
    print(f"Total steps: {report['total']}")
    print(f"Mismatches: {report['mismatches']}")
    if report["mismatches"] > 0:
        print("First 5 mismatches:")
        for d in report["details"][:5]:
            print(d)
