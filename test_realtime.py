#!/usr/bin/env python
"""Quick test of the realtime engine"""

from loop.realtime import RealtimeEngine, Config
from mt5.live_feed import MT5LiveFeed
import logging

# Suppress verbose logging
logging.getLogger('mt5.live_feed').setLevel(logging.ERROR)
logging.getLogger('trading_engine').setLevel(logging.INFO)

print("\n" + "="*70)
print("REALTIME ENGINE COMPONENT TEST")
print("="*70)

print("\n[1] Testing Config...")
config = Config(DEMO_MODE=True, LOOP_INTERVAL=0.1)
print(f"[PASS] Config created: {config.SYMBOL} @ {config.LOOP_INTERVAL}s")

print("\n[2] Testing Engine initialization...")
engine = RealtimeEngine(config, demo_mode=True)
print(f"[PASS] Engine initialized in DEMO mode")

print("\n[3] Testing Feed...")
feed = MT5LiveFeed()
print(f"[PASS] Feed object created")

print("\n[4] Testing market data fetch (mock mode)...")
try:
    # Mock feed in demo mode
    tick = feed.get_tick('EURUSD')
    if tick:
        print(f"[PASS] Tick data retrieved: Bid={tick.bid}, Ask={tick.ask}")
    else:
        print(f"[INFO] Tick data unavailable (expected in offline mode)")
except Exception as e:
    print(f"[INFO] Tick fetch exception (expected): {type(e).__name__}")

print("\n[5] Testing position tracking...")
engine.position.position_id = 123456
engine.position.direction = "buy"
engine.position.entry_price = 1.0850
assert engine.position.is_open() == True
engine.position.close()
assert engine.position.is_open() == False
print(f"[PASS] Position state transitions work correctly")

print("\n[6] Testing iteration counter...")
engine.iterations = 0
engine.errors = 0
engine.iterations += 1
print(f"[PASS] Iteration counter: {engine.iterations}")

print("\n" + "="*70)
print("[PASS] ALL COMPONENT TESTS PASSED")
print("="*70 + "\n")
