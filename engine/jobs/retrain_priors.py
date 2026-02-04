from __future__ import annotations

import argparse
import json
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from engine import ml_training


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic retraining for ML priors"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--days", type=int, help="Train on the last N days (inclusive)")
    group.add_argument("--from", dest="from_date", help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", help="End date YYYY-MM-DD")
    return parser.parse_args()


def _resolve_window(
    args: argparse.Namespace,
) -> Tuple[Optional[date], Optional[date], str]:
    if args.days:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=max(0, args.days - 1))
        return start, end, f"last_{args.days}_days"
    if args.from_date:
        start = date.fromisoformat(args.from_date)
        end = date.fromisoformat(args.to_date) if args.to_date else start
        return start, end, f"{start.isoformat()}_{end.isoformat()}"
    # default: use all data
    return None, None, "all_time"


def _latest_promoted_meta(model_name: str) -> Optional[Dict[str, object]]:
    base = Path("models") / model_name
    if not base.exists():
        return None
    metas = []
    for meta_path in base.rglob("*_metadata.json"):
        try:
            obj = json.loads(meta_path.read_text(encoding="utf-8"))
            obj["_path"] = meta_path
            metas.append(obj)
        except Exception:
            continue
    if not metas:
        return None
    metas = sorted(metas, key=lambda m: m.get("timestamp_utc", ""))
    for m in reversed(metas):
        if m.get("promoted") is True:
            return m
    return metas[-1]


def _better(new: Dict[str, object], old: Dict[str, object]) -> bool:
    new_val = (new.get("metrics", {}) or {}).get("val", {})
    old_val = (old.get("metrics", {}) or {}).get("val", {})
    new_acc = new_val.get("accuracy", 0.0)
    old_acc = old_val.get("accuracy", 0.0)
    if new_acc > old_acc:
        return True
    if new_acc < old_acc:
        return False
    # tie-breaker: lower brier wins
    return new_val.get("brier", 1.0) <= old_val.get("brier", 1.0)


def _flatten_artifacts(version_dir: Path, model_name: str) -> Tuple[Path, Path]:
    nested = version_dir / model_name
    meta_path = nested / f"{model_name}_metadata.json"
    weights_path = nested / f"{model_name}_weights.pkl"
    if nested.exists():
        for child in nested.iterdir():
            shutil.move(str(child), str(version_dir / child.name))
        shutil.rmtree(nested, ignore_errors=True)
    return (
        version_dir / f"{model_name}_weights.pkl",
        version_dir / f"{model_name}_metadata.json",
    )


def _write_promoted_metadata(
    meta_path: Path,
    model_name: str,
    version_tag: str,
    window: Tuple[Optional[date], Optional[date]],
    promoted: bool,
) -> Dict[str, object]:
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    start, end = window
    meta.update(
        {
            "model_name": model_name,
            "version": meta.get("version"),
            "training_window": {
                "start_date": start.isoformat() if start else None,
                "end_date": end.isoformat() if end else None,
            },
            "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "promoted": promoted,
            "version_tag": version_tag,
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta


def _train_single(
    model_name: str,
    train_fn,
    start: Optional[date],
    end: Optional[date],
    version_tag: str,
) -> Dict[str, object]:
    version_dir = Path("models") / model_name / version_tag
    cfg = ml_training.TrainingConfig(
        models_dir=version_dir, start_date=start, end_date=end
    )
    result = train_fn(cfg)
    # flatten artifacts (ml_training nests model_name)
    _, meta_path = _flatten_artifacts(version_dir, model_name)
    return {"result": result, "meta_path": meta_path, "version_dir": version_dir}


def main() -> None:
    args = _parse_args()
    start, end, version_tag = _resolve_window(args)
    window = (start, end)

    model_fns = {
        "macro_up_prob": ml_training.train_macro_up_model,
        "volatility_spike_prob": ml_training.train_volatility_spike_model,
        "regime_transition_prob": ml_training.train_regime_transition_model,
        "directional_confidence": ml_training.train_directional_confidence_model,
    }

    for name, fn in model_fns.items():
        prev = _latest_promoted_meta(name)
        outcome = _train_single(name, fn, start, end, version_tag)
        meta = json.loads(outcome["meta_path"].read_text(encoding="utf-8"))
        promote = True if prev is None else _better(meta, prev)
        updated = _write_promoted_metadata(
            outcome["meta_path"], name, version_tag, window, promote
        )
        if promote:
            # mark previous as not promoted
            if prev and "_path" in prev:
                try:
                    prev_meta = json.loads(prev["_path"].read_text(encoding="utf-8"))
                    prev_meta["promoted"] = False
                    prev["_path"].write_text(
                        json.dumps(prev_meta, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
        print(
            json.dumps(
                {
                    "model": name,
                    "promoted": promote,
                    "metrics": updated.get("metrics", {}),
                }
            )
        )


if __name__ == "__main__":
    main()
