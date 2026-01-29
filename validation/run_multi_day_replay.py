"""
Run a multi-day replay using the realtime engine loop.
Logs diagnostics and produces summary reports.
"""

from typing import Any, Dict, List

from realtime.engine_loop import create_engine_loop
from replay.replay_loader import load_replay_events
from validation.utils import save_diagnostics_report, setup_validation_logging


def main():
    setup_validation_logging()
    deps = load_engine_dependencies()
    engine = create_engine_loop(deps)
    diagnostics: List[Any] = []
    for day in ["day1.json", "day2.json", "day3.json"]:
        events = load_replay_events(f"data/{day}")
        for event in events:
            engine.run_step(event)
            diagnostics.append(engine.get_last_diagnostics())
    save_diagnostics_report(diagnostics, "validation/multi_day_replay_diagnostics.json")
    print("Multi-day replay complete. Diagnostics saved.")


def load_engine_dependencies() -> Dict[str, Any]:
    raise RuntimeError(
        "Mock dependencies are disabled. Provide real data deps and adapters."
    )


if __name__ == "__main__":
    main()
