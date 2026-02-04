from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from engine.modes import Mode, get_adapter


def preflight_check(tests_green: bool, policy_path: Path, safe_mode_state: Optional[str], connectors_healthy: bool) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if not tests_green:
        issues.append("test_suite_not_green")
    if not policy_path.exists():
        issues.append("policy_missing")
    if safe_mode_state is None:
        issues.append("safe_mode_unknown")
    if not connectors_healthy:
        issues.append("connectors_unhealthy")
    return len(issues) == 0, issues


def runtime_limits(pnl_today: float, max_daily_loss: float, position: float, max_position: float) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if pnl_today <= -abs(max_daily_loss):
        issues.append("max_daily_loss_exceeded")
    if abs(position) > abs(max_position):
        issues.append("max_position_exceeded")
    return len(issues) == 0, issues


def kill_switch(mode: Mode, adapter=None) -> Dict[str, object]:
    adapter = adapter or get_adapter(mode)
    adapter.disable_orders()
    return {"mode": mode.value, "adapter": adapter.name, "disabled": adapter.disabled}
