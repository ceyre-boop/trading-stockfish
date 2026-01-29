"""
Run a full-day replay using the realtime engine loop.
Logs diagnostics and produces summary reports.
"""

import json
import os
from typing import Any, Dict, List

from realtime.engine_loop import create_engine_loop
from replay.replay_loader import load_replay_events
from validation.utils import save_diagnostics_report, setup_validation_logging


def main():

    setup_validation_logging()
    # Load dependencies for engine loop (mock or real)
    deps = load_engine_dependencies()
    # Pass experiment_one_shot and experiment_id into engine loop
    engine = create_engine_loop(
        {
            **deps,
            "experiment_one_shot": True,
            "experiment_id": "first_one_shot_live_paper_replay",
        }
    )

    # Load scenario for today
    scenario_path = os.path.join(
        "research", "scenarios", "today_replay_experiment.json"
    )
    events = load_replay_events(scenario_path)
    diagnostics: List[Any] = []
    for i, event in enumerate(events["ticks"]):
        # Build market state, evaluate, policy, governance, execution, etc.
        result = engine.run_step(event)
        diag = engine.get_last_diagnostics()
        diagnostics.append(diag)
    save_diagnostics_report(diagnostics, "validation/full_replay_diagnostics.json")
    print("Full replay complete. Diagnostics saved.")


def load_engine_dependencies() -> Dict[str, Any]:
    raise RuntimeError(
        "Mock dependencies are disabled. Provide real data deps and adapters."
    )


if __name__ == "__main__":
    main()
