"""
Compare diagnostics from live trading and replay runs.
Produces a report on consistency and discrepancies.
"""

import json

from validation.utils import compare_diagnostics, print_comparison_report


def main():
    with open("validation/live_diagnostics.json") as f:
        live = json.load(f)
    with open("validation/full_replay_diagnostics.json") as f:
        replay = json.load(f)
    report = compare_diagnostics(live, replay)
    print_comparison_report(report)
    with open("validation/live_vs_replay_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Live vs Replay comparison complete. Report saved.")


if __name__ == "__main__":
    main()
