import argparse
import json
from pathlib import Path

from engine.tactical_cycle import (
    _DEFAULT_POLICY_PATH,
    load_active_policy,
    load_selector_artifacts,
    run_tactical_cycle,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tactical replay + retrain cycle")
    parser.add_argument("start_date", help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("end_date", help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument(
        "logs_dir", help="Directory containing decision log .jsonl files"
    )
    parser.add_argument("out_dir", help="Output directory for artifacts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    selector_artifact_path = out_dir / "selector_artifacts.joblib"
    belief_map_path = out_dir / "entry_belief_map.json"

    policy_df = load_active_policy(_DEFAULT_POLICY_PATH)
    existing_artifacts = load_selector_artifacts(selector_artifact_path)

    summary = run_tactical_cycle(
        start_date=args.start_date,
        end_date=args.end_date,
        logs_dir=args.logs_dir,
        selector_artifact_path=str(selector_artifact_path),
        belief_map_path=str(belief_map_path),
        policy_path=str(_DEFAULT_POLICY_PATH),
        brain_policy_entries=policy_df,
        entry_selector_artifacts=existing_artifacts,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
