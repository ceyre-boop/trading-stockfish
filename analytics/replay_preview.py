"""Simple replay preview that walks synthetic ES minute bars for a day
and prints session, flow summary, and capacity limits.
"""
from datetime import datetime, timezone, timedelta
import math
import random

from engine.session_context import SessionContext
from engine.portfolio_risk import PortfolioRiskManager


def generate_synthetic_day(start_dt: datetime, minutes: int = 60 * 10):
    prices = []
    volumes = []
    p0 = 5000.0
    for i in range(minutes):
        t = i / 60.0
        drift = 2.0 * math.sin(t / 3.0)
        noise = random.normalvariate(0, 1.5)
        p = p0 + drift + noise
        v = max(100, int(1000 + 200 * math.cos(t / 10.0)))
        prices.append(p)
        volumes.append(v)
    return prices, volumes


def main():
    sc = SessionContext()
    prm = PortfolioRiskManager()

    start = datetime(2026, 1, 15, 3, 0, tzinfo=timezone.utc)
    minutes = 60 * 10
    prices, vols = generate_synthetic_day(start, minutes)

    # prior/overnight highs/lows for demonstration
    prior_high = max(prices[:60]) + 10
    prior_low = min(prices[:60]) - 10
    overnight_high = prior_high + 5
    overnight_low = prior_low - 5

    round_levels = [5000.0, 18000.0]

    # Walk the series, update session every 30 minutes
    for i in range(0, minutes, 30):
        ts = start + timedelta(minutes=i)
        window_prices = prices[max(0, i-30):i+1] or [prices[i]]
        window_vols = vols[max(0, i-30):i+1] or [vols[i]]
        sc.update(ts, recent_prices=window_prices, recent_volumes=window_vols,
                  prior_high=prior_high, prior_low=prior_low,
                  overnight_high=overnight_high, overnight_low=overnight_low,
                  round_levels=round_levels)

        # sample capacity check: try placing a modest order and a large order
        small = prm.enforce_capacity_constraints("ES", size=5, price=prices[i], volume_1m=100000, volume_5m=400000, volatility=0.01, depth=2000000)
        large = prm.enforce_capacity_constraints("ES", size=5000, price=prices[i], volume_1m=100000, volume_5m=400000, volatility=0.02, depth=2000000)

    # Print summary
    print("Session:", sc.get_session().value)
    print("Modifiers:", sc.get_session_modifiers())
    print("Flow VWAP:", sc.flow.vwap)
    print("Prior H/L:", sc.flow.prior_high, sc.flow.prior_low)
    print("Overnight H/L:", sc.flow.overnight_high, sc.flow.overnight_low)

    # capacity limits
    for sym in ["ES", "NQ"]:
        nl = prm.compute_notional_limit(sym)
        print(f"{sym} notional limit: {nl}")

    print("Replay preview complete. Logs written to logs/session/")


if __name__ == "__main__":
    main()
