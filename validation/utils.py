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
    report = {"total": len(live), "mismatches": 0, "details": []}
    for i, (l, r) in enumerate(zip(live, replay)):
        if l != r:
            report["mismatches"] += 1
            report["details"].append({"index": i, "live": l, "replay": r})
    return report


def print_comparison_report(report: dict) -> None:
    print(f"Total steps: {report['total']}")
    print(f"Mismatches: {report['mismatches']}")
    if report["mismatches"] > 0:
        print("First 5 mismatches:")
        for d in report["details"][:5]:
            print(d)
