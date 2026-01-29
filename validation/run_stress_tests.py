"""
Run stress tests on the realtime engine loop.
Simulates high-frequency and edge-case event streams.
"""

from typing import Any, Dict, List

from realtime.engine_loop import create_engine_loop
from validation.stress_test_cases import generate_stress_events
from validation.utils import save_diagnostics_report, setup_validation_logging


def main():
    setup_validation_logging()
    deps = load_engine_dependencies()
    engine = create_engine_loop(deps)
    diagnostics: List[Any] = []
    events = generate_stress_events()
    for event in events:
        engine.run_step(event)
        diagnostics.append(engine.get_last_diagnostics())
    save_diagnostics_report(diagnostics, "validation/stress_test_diagnostics.json")
    print("Stress tests complete. Diagnostics saved.")


def load_engine_dependencies() -> Dict[str, Any]:
    raise RuntimeError(
        "Mock dependencies are disabled. Provide real data deps and adapters."
    )


if __name__ == "__main__":
    main()
