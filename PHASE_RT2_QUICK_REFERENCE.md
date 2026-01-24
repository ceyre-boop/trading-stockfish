# PHASE RT-2 Quick Reference

## File Structure

```
realtime/
├── __init__.py                      # Module exports
├── data_models.py                   # Market data structures (6.6 KB)
├── exchange_base_connector.py       # Abstract base class (11.7 KB)
├── exchange_ibkr_connector.py       # Interactive Brokers (17.6 KB)
├── exchange_fix_connector.py        # FIX Protocol (19.8 KB)
├── exchange_zmq_connector.py        # ZeroMQ/Crypto (17.7 KB)
└── exchange_manager.py              # Multi-connector orchestrator (18.4 KB)

Total: 7 files, ~93 KB, 1,700+ lines
```

## Key Classes

### Order Management
```python
from realtime import Order, OrderSide, OrderType, OrderStatus

# Create order
order = Order(
    symbol='SPY',
    side=OrderSide.BUY,
    quantity=100,
    order_type=OrderType.LIMIT,
    price=450.0
)

# Track status
print(order.status)  # OrderStatus.PENDING
print(order.order_id)  # None until submitted
print(order.filled_quantity)  # 0 until filled
```

### Market Data
```python
from realtime import MarketUpdate, DataType, PriceTick

# Handle incoming updates
def on_market_update(update: MarketUpdate):
    if update.data_type == DataType.PRICE_TICK:
        tick: PriceTick = update.payload
        print(f"{tick.symbol}: {tick.mid_price} ({tick.spread_bps} bps)")
```

### ExchangeManager API
```python
from realtime import ExchangeManager, FailoverStrategy

manager = ExchangeManager(
    failover_strategy=FailoverStrategy.BEST_AVAILABLE
)

# Lifecycle
manager.start_all()         # Connect all
manager.stop_all()          # Disconnect all

# Subscriptions
manager.subscribe_price(['SPY', 'QQQ'])
manager.subscribe_orderbook(['SPY'], depth=20)
manager.unsubscribe(['QQQ'])

# Orders
order_id = manager.send_order(order)
manager.cancel_order(order_id)
status = manager.get_order_status(order_id)

# Stats
stats = manager.get_stats()
health = manager.get_health_status()

# List connectors
for name, status, is_primary in manager.list_connectors():
    print(f"{name}: {status} (primary={is_primary})")
```

## Data Types

| Class | Purpose | Fields |
|-------|---------|--------|
| `PriceTick` | Level 1 quote | symbol, bid, ask, last, volumes, timestamp |
| `OrderBookSnapshot` | L2+ orderbook | symbol, bids, asks, sequence, timestamp |
| `OHLCVBar` | Candlestick | symbol, open, high, low, close, volume, interval |
| `NewsEvent` | News + sentiment | title, content, source, sentiment, confidence |
| `MacroEvent` | Economic data | event, country, value, forecast, importance |
| `MarketUpdate` | Universal wrapper | data_type, payload, sequence_number, timestamp |

## Connector Reference

| Connector | Use Case | Features | Dependencies |
|-----------|----------|----------|--------------|
| **IBKR** | US stocks, futures, forex | Real-time quotes, execution, positions | ib_insync |
| **FIX** | Professional trading | High-frequency, low-latency | QuickFIX or socket |
| **ZMQ** | Crypto, custom feeds | Pub/Sub streaming, JSON | pyzmq |

## Common Tasks

### 1. Connect Single Connector
```python
from realtime.exchange_ibkr_connector import IBKRConnector

connector = IBKRConnector(host='127.0.0.1', port=7497)
if connector.connect():
    connector.subscribe_price('SPY')
    print("Connected!")
else:
    print("Failed to connect")
```

### 2. Multi-Connector Setup
```python
from realtime import ExchangeManager
from realtime.exchange_ibkr_connector import IBKRConnector
from realtime.exchange_zmq_connector import ZeroMQConnector, ZMQFeedType

manager = ExchangeManager()

# Primary execution
ibkr = IBKRConnector()
manager.add_connector(ibkr, 'ibkr', primary=True)

# Crypto data
zmq = ZeroMQConnector(
    endpoints=['tcp://crypto.feed:5555'],
    feed_type=ZMQFeedType.TICKER
)
manager.add_connector(zmq, 'zmq', data_only=True)

manager.start_all()
```

### 3. Handle Market Updates
```python
from realtime import DataType, MarketUpdate

class MyStrategy:
    def on_price_tick(self, update: MarketUpdate):
        tick = update.payload
        print(f"Price: {tick.symbol} {tick.last}")
    
    def on_orderbook(self, update: MarketUpdate):
        ob = update.payload
        print(f"Best bid/ask: {ob.best_bid}/{ob.best_ask}")

router.add_subscriber(MyStrategy())
```

### 4. Order Execution
```python
from realtime import Order, OrderSide, OrderType

# Create and send
order = Order('SPY', OrderSide.BUY, 100, OrderType.LIMIT, price=450)
order_id = manager.send_order(order)

# Monitor
while True:
    status = manager.get_order_status(order_id)
    if status == OrderStatus.FILLED:
        print(f"Filled at {order.avg_fill_price}")
        break
    time.sleep(0.1)
```

## Order Status Lifecycle

```
PENDING → SUBMITTED → PARTIALLY_FILLED → FILLED
              ↓              ↓
           REJECTED/ERROR  CANCELLED
```

## Connector Status Lifecycle

```
DISCONNECTED → CONNECTING → CONNECTED
                   ↓            ↓
                ERROR    ← RECONNECTING
```

## Environment Setup

### IBKR
1. Download TWS or Gateway
2. Enable Socket API: Main Menu → API → Settings
3. Uncheck "Read-only API"
4. Set port to 7497 (paper) or 7496 (live)

### FIX
1. Obtain FIX credentials from broker
2. Set sender_comp_id and target_comp_id
3. Verify protocol version (FIX42/44/50)
4. Whitelist IP address if required

### ZMQ
1. Ensure data provider is publishing
2. Verify endpoint URL is accessible
3. Check network firewall rules
4. Confirm feed format (JSON)

## Testing

```bash
# Run unit tests
python -m pytest tests/ -v

# Integration test
python tests/test_exchange_integration.py

# Monitor live data (IBKR)
python -c "
from realtime.exchange_ibkr_connector import IBKRConnector
conn = IBKRConnector()
if conn.connect():
    conn.subscribe_price('SPY')
    import time; time.sleep(60)
    conn.disconnect()
"
```

## Performance Tips

1. **Batch subscriptions**: Subscribe to multiple symbols at once
2. **Use ZMQ for data**: Crypto/custom feeds only, not execution
3. **Primary for orders**: Keep IBKR/FIX as primary execution
4. **Monitor stats**: Check `get_stats()` for bottlenecks
5. **Tune buffers**: Adjust ZMQ reorder buffer size if needed

## Troubleshooting

### IBKR won't connect
- Check TWS running: `telnet 127.0.0.1 7497`
- Verify API enabled in settings
- Check client_id is unique
- Review logs for socket errors

### FIX logon fails
- Verify sender/target comp IDs
- Check heartbeat interval (108 field)
- Confirm protocol version match
- Validate network connectivity

### ZMQ no data
- Test endpoint: `zmq-cli -r tcp://endpoint:port`
- Verify feed is publishing
- Check message format (JSON)
- Review sequence numbers for gaps

## Common Patterns

### Pattern 1: Fail-Safe Trading
```python
manager = ExchangeManager(failover_strategy=FailoverStrategy.PRIMARY_BACKUP)
manager.add_connector(ibkr, 'ibkr', primary=True)
manager.add_connector(fix, 'fix', primary=False)
# Auto-fails to FIX if IBKR down
```

### Pattern 2: Multi-Feed Data
```python
manager.add_connector(zmq_us, 'zmq_us', data_only=True)
manager.add_connector(zmq_crypto, 'zmq_crypto', data_only=True)
manager.subscribe_price(['SPY', 'QQQ'], connectors=['zmq_us'])
manager.subscribe_price(['BTCUSD'], connectors=['zmq_crypto'])
```

### Pattern 3: Health-Aware Trading
```python
while trading:
    health = manager.get_health_status()
    if health['ibkr']['connected']:
        order_id = manager.send_order(order)
    else:
        print("Primary connector down, waiting for recovery...")
```

---

**Last Updated**: January 19, 2026  
**Version**: 1.0 (Production)
