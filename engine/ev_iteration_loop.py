import json
import shutil
from pathlib import Path
from typing import Dict

import pandas as pd

from .adversarial_replay import adversarial_replay, collect_decision_records
from .ev_brain_training import train_ev_brain
from .ev_dataset_builder import build_ev_dataset
from .policy_synthesis import synthesize_entry_policy


def validate_policy_stability(
    old_policy: Dict[str, str], new_policy: Dict[str, str]
) -> Dict[str, object]:
    keys = set(old_policy.keys()) | set(new_policy.keys())
    flips = [k for k in keys if old_policy.get(k) != new_policy.get(k)]
    flip_rate = len(flips) / max(len(keys), 1)
    severe_flip = any(
        old_policy.get(k) == "DISABLED" and new_policy.get(k) == "PREFERRED"
        for k in keys
    )
    stable = flip_rate <= 0.5 and not severe_flip
    return {
        "stable": stable,
        "flip_rate": flip_rate,
        "flipped": sorted(flips),
        "severe_flip": severe_flip,
    }


def _load_active_policy(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("policy") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            return {}
        return {
            str(rec.get("entry_model_id")): str(rec.get("label"))
            for rec in records
            if isinstance(rec, dict)
        }
    except Exception:
        return {}


def run_ev_iteration(
    replay_config,
    *,
    min_samples: int = 50,
    version: str = "v1",
    output_dir: str = "ev_iterations",
) -> Dict[str, object]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decision_logs: pd.DataFrame = replay_config.get("decision_logs")
    artifacts = replay_config.get("entry_selector_artifacts")
    brain_policy_entries = replay_config.get("brain_policy_entries")

    # 1. Replay (no behavior change)
    replay_df = adversarial_replay(decision_logs, artifacts, brain_policy_entries)

    # 2. Collect decision records
    decision_records = collect_decision_records(decision_logs)

    # 3. Build EV dataset
    ev_dataset = build_ev_dataset(replay_df, decision_records, version=version)
    dataset_path = out_dir / "dataset.parquet"
    try:
        ev_dataset.to_parquet(dataset_path, index=False)
    except Exception:
        ev_dataset.to_json(
            dataset_path.with_suffix(".json"), orient="records", lines=True
        )

    # 4. Train EV brain
    ev_brain = train_ev_brain(ev_dataset, version=version)
    model_path = out_dir / "ev_brain_v1.pkl"
    ev_brain.save(model_path)

    # 5. Policy synthesis
    learned_policy_path = out_dir / "brain_policy_entries.learned.json"
    learned_policy = synthesize_entry_policy(
        ev_dataset,
        version=version,
        min_samples=min_samples,
        output_path=learned_policy_path,
    )

    # 6. Stability validation
    active_policy_path = Path("storage/policies/brain/brain_policy_entries.active.json")
    active_policy = _load_active_policy(active_policy_path)
    validation = validate_policy_stability(active_policy, learned_policy)
    validation_path = out_dir / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    promoted_path = out_dir / "promoted_policy.json"
    promoted = False
    if validation.get("stable"):
        try:
            shutil.copyfile(learned_policy_path, promoted_path)
            promoted = True
        except Exception:
            promoted = False

    summary = {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "learned_policy_path": str(learned_policy_path),
        "validation_path": str(validation_path),
        "promoted_policy_path": str(promoted_path) if promoted else None,
        "promoted": promoted,
        "flip_rate": validation.get("flip_rate"),
    }
    return summary
