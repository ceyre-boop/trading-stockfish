import importlib
import pkgutil
import sys
import os
from datetime import datetime, timezone

# Ensure project root is on sys.path so `import engine` works when running from tools/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODULES = [
    "engine",
    "engine.session_logging",
    "engine.session_context",
    "engine.portfolio_risk",
    "engine.portfolio_risk_manager",
]


def try_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        print(f"OK IMPORT: {name}")
        return True
    except Exception as e:
        print(f"ERROR IMPORT: {name} -> {e}")
        return False


def run_smoke():
    from engine.session_context import SessionContext, FlowContext
    from engine.portfolio_risk import PortfolioRiskManager as NewPRM
    from engine.portfolio_risk_manager import PortfolioRiskManager as OldPRM

    results = {
        "session_instantiate": False,
        "flow_instantiate": False,
        "new_prm": False,
        "old_prm": False,
        "session_transition": False,
        "flow_update": False,
        "capacity_calc": False,
    }

    try:
        sc = SessionContext()
        results["session_instantiate"] = True
        fc = sc.flow
        results["flow_instantiate"] = True

        # Transition
        t0 = datetime(2026, 1, 15, 3, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 15, 9, tzinfo=timezone.utc)
        sc.update(t0, recent_prices=[5000.0])
        sc.update(t1, recent_prices=[5002.0])
        results["session_transition"] = True

        # flow update
        fc.update_from_price_series([5000.0, 5002.0, 4998.0], [100, 200, 150], prior_high=5020.0, prior_low=4970.0, overnight_high=5030.0, overnight_low=4960.0, round_levels=[5000.0])
        results["flow_update"] = True
    except Exception as e:
        print("Session/Flow smoke failed:", e)

    try:
        prm = NewPRM()
        results["new_prm"] = True
        r = prm.enforce_capacity_constraints("ES", size=10, price=5000.0, volume_1m=100000, volume_5m=400000, volatility=0.01, depth=2000000)
        results["capacity_calc"] = True
    except Exception as e:
        print("NewPRM smoke failed:", e)

    try:
        op = OldPRM(total_capital=100000.0, max_symbol_exposure=20000.0, max_total_exposure=50000.0, max_daily_loss=5000.0)
        results["old_prm"] = True
    except Exception as e:
        print("OldPRM smoke failed:", e)

    print("\nSMOKE SUMMARY:")
    for k, v in results.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")

    all_pass = all(results.values())
    print('\nOverall:', 'PASS' if all_pass else 'FAIL')
    return 0 if all_pass else 2


def main():
    ok = True
    for m in MODULES:
        if not try_import(m):
            ok = False

    rc = 0
    if ok:
        rc = run_smoke()
    else:
        print("Skipping smoke tests due to import errors.")
        rc = 2
    sys.exit(rc)


if __name__ == '__main__':
    main()
