"""Policy config validator for PAPER week.

Validates the active policy_config.json in project root against required fields
and sanity constraints for PAPER mode. Exits non-zero on failure.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "policy_config.json"
SESSION_REGIMES_PATH = PROJECT_ROOT / "docs" / "session_regimes.md"

REQUIRED_FIELDS = {"policy_version", "base_weights", "trust", "regime_multipliers", "metadata"}
ALLOWED_FIELDS = REQUIRED_FIELDS | {"safe_mode"}
TRUST_MIN = 0.0
TRUST_MAX = 5.0


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing config at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _parse_session_regimes(path: Path) -> List[str]:
    if not path.exists():
        return []
    regimes: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("|") is False:
            continue
        if set(line) <= {"|", "-", " "}:
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if parts:
            regimes.append(parts[0])
    return regimes


def _check_base_weights(cfg: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    weights = cfg.get("base_weights")
    if not isinstance(weights, dict):
        errors.append("base_weights must be a dict of finite numbers")
        return errors
    for k, v in weights.items():
        if not _is_finite_number(v):
            errors.append(f"base_weights[{k}] is not finite")
    return errors


def _check_trust(cfg: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    trust = cfg.get("trust")
    if isinstance(trust, dict):
        items = trust.items()
    elif _is_finite_number(trust):
        items = [("__global__", trust)]
    else:
        errors.append("trust must be a finite number or dict of finite numbers")
        return errors
    for k, v in items:
        if not _is_finite_number(v):
            errors.append(f"trust[{k}] is not finite")
            continue
        if not (TRUST_MIN <= float(v) <= TRUST_MAX):
            errors.append(f"trust[{k}] outside [{TRUST_MIN}, {TRUST_MAX}]")
    return errors


def _check_regime_multipliers(cfg: Dict[str, Any], required_regimes: List[str]) -> List[str]:
    errors: List[str] = []
    rm = cfg.get("regime_multipliers")
    if not isinstance(rm, dict):
        errors.append("regime_multipliers must be a dict")
        return errors
    missing = [r for r in required_regimes if r not in rm]
    if missing:
        errors.append(f"regime_multipliers missing regimes: {', '.join(missing)}")
    for regime, weights in rm.items():
        if not isinstance(weights, dict):
            errors.append(f"regime_multipliers[{regime}] must be a dict")
            continue
        for k, v in weights.items():
            if not _is_finite_number(v):
                errors.append(f"regime_multipliers[{regime}][{k}] is not finite")
    return errors


def _check_metadata(cfg: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    metadata = cfg.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be a dict")
    return errors


def _check_safe_mode(cfg: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    metadata = cfg.get("metadata") if isinstance(cfg.get("metadata"), dict) else {}
    safe_mode = cfg.get("safe_mode") if "safe_mode" in cfg else metadata.get("safe_mode") or metadata.get("safe_mode_state")
    if safe_mode is None:
        errors.append("SAFE_MODE flag/state missing (expected safe_mode or metadata.safe_mode[_state])")
        return errors
    if isinstance(safe_mode, bool):
        return errors
    if isinstance(safe_mode, str):
        if safe_mode.upper() not in {"NORMAL", "SAFE_MODE"}:
            errors.append("safe_mode string must be NORMAL or SAFE_MODE")
        return errors
    if isinstance(safe_mode, dict):
        if "enabled" in safe_mode and not isinstance(safe_mode.get("enabled"), bool):
            errors.append("safe_mode.enabled must be bool when present")
        return errors
    errors.append("safe_mode must be bool, dict, or state string")
    return errors


def _check_required_fields(cfg: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for field in REQUIRED_FIELDS:
        if field not in cfg:
            errors.append(f"missing required field: {field}")
    return errors


def _check_deprecated(cfg: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    extras = [k for k in cfg.keys() if k not in ALLOWED_FIELDS]
    if extras:
        errors.append(f"deprecated/unknown fields present: {', '.join(extras)}")
    return errors


def _validate(cfg: Dict[str, Any], regimes: List[str]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    errors.extend(_check_required_fields(cfg))
    errors.extend(_check_deprecated(cfg))
    if not isinstance(cfg.get("policy_version"), str) or not cfg.get("policy_version"):
        errors.append("policy_version must be a non-empty string")
    errors.extend(_check_base_weights(cfg))
    errors.extend(_check_trust(cfg))
    errors.extend(_check_regime_multipliers(cfg, regimes))
    errors.extend(_check_metadata(cfg))
    errors.extend(_check_safe_mode(cfg))
    return (len(errors) == 0, errors)


def main() -> None:
    try:
        cfg = _load_json(CONFIG_PATH)
    except Exception as exc:  # pragma: no cover
        print(f"FAIL: {exc}")
        sys.exit(1)

    regimes = _parse_session_regimes(SESSION_REGIMES_PATH)
    ok, errors = _validate(cfg, regimes)
    if ok:
        print("PASS: policy_config.json is valid for PAPER")
        sys.exit(0)
    print("FAIL: policy_config.json is invalid")
    for err in errors:
        print(f" - {err}")
    sys.exit(1)


if __name__ == "__main__":
    main()
