"""
TournamentRunner for Trading Stockfish v4.0‑F
Deterministic, replay-safe tournament harness for engine comparison.
"""

import json
from typing import Any, Dict, List


class TournamentRunner:
    def __init__(self, execution_simulator):
        self.execution_simulator = execution_simulator

    def run_match(
        self, version_A, version_B, scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Both versions run on the same scenario data
        results = {}
        for version, label in [(version_A, "A"), (version_B, "B")]:
            sim_result = self.execution_simulator.run(version, scenario)
            results[label] = {
                "evaluation_trace": sim_result["evaluation_trace"],
                "policy_decisions": sim_result["policy_decisions"],
                "execution_results": sim_result["execution_results"],
                "pnl_curve": sim_result["pnl_curve"],
                "regime_transitions": sim_result["regime_transitions"],
            }
        return results

    def run_tournament(
        self, versions: List[Any], scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        tournament_results = {}
        for scenario in scenarios:
            scenario_name = scenario.get("name", "unknown")
            tournament_results[scenario_name] = {}
            for i, vA in enumerate(versions):
                for j, vB in enumerate(versions):
                    if i >= j:
                        continue
                    match_result = self.run_match(vA, vB, scenario)
                    key = f"{vA.version_id}_vs_{vB.version_id}"
                    tournament_results[scenario_name][key] = match_result
        return tournament_results

    @staticmethod
    def serialize_results(results: Dict[str, Any]) -> str:
        return json.dumps(results, indent=2, sort_keys=True)
