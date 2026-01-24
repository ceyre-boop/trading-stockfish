# QUICK START - LIVE TRADING

**Phase RT-3: Getting Started with Live Trading**

## 5-Minute Setup

### 1. Install Dependencies

```bash
pip install ib-insync quickfix pyzmq curses  # Optional depending on connectors
```

### 2. Create Trading Session

```python
from realtime import (
    LiveTradingOrchestrator,
    ExchangeManager,
    RealTimeEngineLoop,
    DataFeedRouter,
    SafetyLayer,
    MonitoringDashboard
)
from realtime.exchange_ibkr_connector import IBKRConnector

# Create components
em = ExchangeManager()
el = RealTimeEngineLoop()
dfr = DataFeedRouter()
sl = SafetyLayer()

# Add connector (IBKR example)
ibkr = IBKRConnector()
em.add_connector(ibkr, 'ibkr', primary=True)

# Create orchestrator
orchestrator = LiveTradingOrchestrator(em, el, dfr, sl)

# Create dashboard
dashboard = MonitoringDashboard()
```

### 3. Start Session

```python
# Start components
dashboard.start()
orchestrator.start()

# Monitor state
print(f"State: {orchestrator.get_state().value}")
```

### 4. Trading Loop

```python
while orchestrator.is_running():
    # Post dashboard update
    update = DashboardUpdate(
        timestamp=time.time(),
        price_data=get_prices(),
        pnl_data=get_pnl()
    )
    dashboard.post_update(update)
    
    time.sleep(0.5)
```

### 5. Stop Session

```python
orchestrator.stop()
dashboard.stop()
```

**Time to first live trade: ~5 minutes**

---

## Complete Example

```python
#!/usr/bin/env python3
"""
Complete live trading example - Trading Stockfish v1.0
"""

import time
import logging
from realtime import (
    LiveTradingOrchestrator,
    ExchangeManager,
    RealTimeEngineLoop,
    DataFeedRouter,
    SafetyLayer,
    MonitoringDashboard,
    DashboardUpdate
)
from realtime.exchange_ibkr_connector import IBKRConnector

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    """Main trading session."""
    
    # ===== Setup Components =====
    print("Initializing trading components...")
    
    em = ExchangeManager()
    el = RealTimeEngineLoop()
    dfr = DataFeedRouter()
    sl = SafetyLayer()
    
    # Add IBKR connector
    ibkr = IBKRConnector()
    em.add_connector(ibkr, 'ibkr', primary=True)
    
    # Create orchestrator with logging
    orchestrator = LiveTradingOrchestrator(em, el, dfr, sl)
    
    # Create dashboard
    dashboard = MonitoringDashboard()
    
    # ===== Register Callbacks =====
    print("Registering callbacks...")
    
    def on_state_change(old_state, new_state):
        print(f"State transition: {old_state.value} → {new_state.value}")
    
    def on_safety_event(event):
        print(f"Safety event: [{event.severity}] {event.message}")
    
    def on_pause():
        print("Trading paused by user")
    
    def on_flatten():
        print("Flattening all positions...")
    
    orchestrator.register_state_change_callback(on_state_change)
    orchestrator.register_safety_event_callback(on_safety_event)
    
    dashboard.register_pause_callback(on_pause)
    dashboard.register_flatten_callback(on_flatten)
    
    # ===== Start Session =====
    print("Starting live trading session...")
    
    if not dashboard.start():
        print("Error: Failed to start dashboard")
        return False
    
    if not orchestrator.start():
        print("Error: Failed to start orchestrator")
        dashboard.stop()
        return False
    
    print(f"Session started - State: {orchestrator.get_state().value}")
    
    # ===== Trading Loop =====
    print("Entering trading loop (press Q in dashboard to quit)...")
    
    try:
        while orchestrator.is_running():
            # Get current state and stats
            state = orchestrator.get_state()
            stats = orchestrator.get_stats()
            conn_status = orchestrator.get_connector_status()
            
            # Create dashboard update
            update = DashboardUpdate(
                timestamp=time.time(),
                price_data={
                    'SPY': {
                        'bid': 450.00,
                        'ask': 450.50,
                        'last': 450.25
                    }
                },
                engine_stats={
                    'decisions_made': stats['total_trades'],
                    'orders_submitted': stats['total_trades'],
                    'orders_filled': stats['successful_trades'],
                    'orders_rejected': stats['rejected_trades'],
                    'latency_p50': 5.0,
                    'latency_p99': 15.0,
                    'errors': 0
                },
                pnl_data={
                    'realized_pnl': stats['realized_pnl'],
                    'unrealized_pnl': stats['unrealized_pnl'],
                    'total_pnl': stats['realized_pnl'] + stats['unrealized_pnl']
                },
                health_status={
                    'status': 'healthy' if state.value in ['LIVE', 'READY'] else 'degraded'
                },
                connector_status={
                    name: {
                        'connected': info['is_connected'],
                        'latency': info['latency_ms'],
                        'errors': info['error_count']
                    }
                    for name, info in conn_status.items()
                },
                alerts=[]
            )
            
            dashboard.post_update(update)
            
            # Log stats every 10 seconds
            if int(time.time()) % 10 == 0:
                print(f"State: {state.value}, PnL: {stats['realized_pnl']:.2f}")
            
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"Error in trading loop: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # ===== Shutdown =====
        print("Shutting down...")
        
        # Get final stats
        final_stats = orchestrator.get_stats()
        print(f"\nSession Statistics:")
        print(f"  Duration: {final_stats['session_duration']:.1f}s")
        print(f"  State changes: {final_stats['state_changes']}")
        print(f"  Trades: {final_stats['total_trades']} (filled: {final_stats['successful_trades']})")
        print(f"  Realized PnL: {final_stats['realized_pnl']:.2f}")
        print(f"  Safety events: {final_stats['safety_events']}")
        print(f"  Failsafe activations: {final_stats['failsafe_activations']}")
        
        # Stop components
        orchestrator.stop()
        dashboard.stop()
        
        print("Session ended")
        return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
```

**Run with:**
```bash
python live_trading_example.py
```

---

## First Live Trade Checklist

- [ ] Connect to Interactive Brokers (or other broker)
- [ ] Dashboard shows "HEALTHY" status
- [ ] All connectors show "✓" (connected)
- [ ] Price data updating in real-time
- [ ] Engine making decisions (see "Decisions" counter)
- [ ] Orders being submitted and filled
- [ ] PnL tracking showing realized/unrealized gains
- [ ] No safety alerts or warnings
- [ ] Monitoring dashboard responsive to keyboard input

---

## Keyboard Controls

During live trading:

| Key | Action |
|-----|--------|
| **S** | Summary view |
| **D** | Detailed view |
| **A** | Alerts view |
| **P** | Pause trading |
| **R** | Resume trading |
| **F** | Flatten positions |
| **Q** | Quit session |

---

## Common Issues

### Issue: "Connection failed"

**Cause:** Broker connection error
**Solution:**
1. Check broker credentials
2. Verify network connectivity
3. Check broker server status
4. Review logs in `logs/live/`

### Issue: "No data updates"

**Cause:** Data feed not connected
**Solution:**
1. Verify subscription setup
2. Check data feed permissions
3. Confirm market is open
4. Review connector logs

### Issue: "Orders rejected"

**Cause:** Execution error
**Solution:**
1. Check buying power/cash
2. Verify order parameters
3. Check instrument availability
4. Review broker error logs

### Issue: "High latency"

**Cause:** Network or system issue
**Solution:**
1. Check network connection
2. Monitor system CPU/memory
3. Check broker latency
4. Reduce dashboard refresh rate

### Issue: "FAILSAFE triggered"

**Cause:** Safety anomaly detected
**Solution:**
1. Review safety logs
2. Check market conditions
3. Recover via `recover_from_degraded()`
4. Investigate root cause before resuming

---

## Configuration Tips

### For Fast Execution

```python
config = {
    'health_check_interval_s': 1.0,  # More frequent checks
}
orchestrator = LiveTradingOrchestrator(em, el, dfr, sl, config=config)
```

### For Stability

```python
config = {
    'failsafe_degradation_count': 5,  # More tolerant
}
safety_config = {
    'max_price_jump_pct': 50.0,  # More lenient
    'reject_loop_threshold': 10   # More lenient
}
```

### For Monitoring

```python
dashboard_config = {
    'refresh_interval_s': 1.0,  # Less frequent updates
}
dashboard = MonitoringDashboard(config=dashboard_config)
```

---

## Next Steps

1. **Read detailed documentation:**
   - [LIVE_TRADING_ORCHESTRATOR_SPEC.md](LIVE_TRADING_ORCHESTRATOR_SPEC.md)
   - [SAFETY_LAYER_SPEC.md](SAFETY_LAYER_SPEC.md)
   - [MONITORING_DASHBOARD_GUIDE.md](MONITORING_DASHBOARD_GUIDE.md)

2. **Tune your trading:**
   - Adjust engine parameters
   - Configure safety thresholds
   - Set position limits
   - Configure risk management

3. **Monitor performance:**
   - Track PnL metrics
   - Review latency statistics
   - Monitor safety events
   - Analyze trade statistics

4. **Scale up:**
   - Add more symbols
   - Add more connectors
   - Increase order frequency
   - Expand position size

---

## Emergency Procedures

### Pause All Trading

Press **[P]** or call:
```python
orchestrator.engine_loop.pause()
```

### Immediate Position Closure

Press **[F]** or call:
```python
orchestrator.engine_loop.flatten_positions()
```

### Emergency Shutdown

Press **[Q]** or call:
```python
orchestrator.stop()
```

---

## Performance Baseline

Expected performance metrics:

| Metric | Typical | Good | Excellent |
|--------|---------|------|-----------|
| Latency P50 | 10ms | 5ms | <2ms |
| Latency P99 | 50ms | 20ms | <10ms |
| Orders/sec | 10 | 50 | 100+ |
| Fill rate | 95% | 98% | 99%+ |
| Safety events | <5/day | 0/day | 0/day |

---

## 24/7 Trading (Continuous)

For systems that trade across multiple markets/timezones:

```python
while True:
    # Check market hours
    if not is_market_open():
        print("Market closed, waiting...")
        time.sleep(300)  # Sleep 5 minutes
        continue
    
    # Run session
    run_trading_session()
    
    # Sleep between sessions
    time.sleep(60)
```

---

## Support

### Documentation

- [LIVE_TRADING_ORCHESTRATOR_SPEC.md](LIVE_TRADING_ORCHESTRATOR_SPEC.md)
- [SAFETY_LAYER_SPEC.md](SAFETY_LAYER_SPEC.md)
- [MONITORING_DASHBOARD_GUIDE.md](MONITORING_DASHBOARD_GUIDE.md)
- [PHASE_RT2_IMPLEMENTATION.md](../PHASE_RT2_IMPLEMENTATION.md)

### Logs

Check logs in: `logs/live/`

- `live_trading_<timestamp>.log` - Main log
- `safety_events_<timestamp>.log` - Safety events
- `connector_health_<timestamp>.log` - Connector status
- `governance_events_<timestamp>.log` - Governance alerts

### Debugging

Enable debug logging:
```python
import logging
logging.getLogger('realtime').setLevel(logging.DEBUG)
```

---

## Next Phase

Ready for advanced features? See:
- Multi-connector failover
- Advanced risk management
- Custom strategy integration
- Performance optimization

---

**Happy trading! 🚀**
