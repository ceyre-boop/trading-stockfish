"""
Real-Time Trading Module

Complete real-time exchange integration for Trading Stockfish:

Phase RT-1: Core Infrastructure
  - data_models.py: Market data structures
  - data_feed_router.py: Thread-safe update routing
  - engine_loop.py: Real-time trading engine
  - simulated_feeds.py: Backtesting feeds

Phase RT-2: Live Exchange Integration
  - exchange_base_connector.py: Abstract base connector interface
  - exchange_ibkr_connector.py: Interactive Brokers integration
  - exchange_fix_connector.py: FIX protocol support
  - exchange_zmq_connector.py: ZeroMQ/crypto feeds
  - exchange_manager.py: Multi-connector orchestrator

Phase RT-3: Live Trading Orchestration & Safety
  - live_trading_orchestrator.py: Session and state management
  - safety_layer.py: Real-time anomaly detection
  - monitoring_dashboard.py: CLI monitoring interface
  - logging_config.py: Centralized logging infrastructure

Author: Trading Stockfish Development Team
Date: January 19, 2026
"""

from realtime.data_models import (
    DataType,
    PriceTick,
    OrderBookLevel,
    OrderBookSnapshot,
    OHLCVBar,
    NewsEvent,
    MacroEvent,
    MarketUpdate
)

from realtime.exchange_base_connector import (
    BaseConnector,
    ConnectorStatus,
    Order,
    OrderSide,
    OrderStatus,
    OrderType
)

from realtime.exchange_manager import (
    ExchangeManager,
    FailoverStrategy
)

from realtime.live_trading_orchestrator import (
    LiveTradingOrchestrator,
    OrchestratorState,
    SessionState,
    ConnectorHealthEvent,
    EngineHealthEvent,
    GovernanceEvent
)

from realtime.safety_layer import (
    SafetyLayer,
    SafetyEvent,
    SafetyEventType,
    SafetySeverity
)

from realtime.monitoring_dashboard import (
    MonitoringDashboard,
    DashboardUpdate,
    DashboardMode
)

from realtime.logging_config import (
    LiveTradingLogManager,
    get_logger
)

__all__ = [
    # Data Models (Phase RT-1)
    'DataType',
    'PriceTick',
    'OrderBookLevel',
    'OrderBookSnapshot',
    'OHLCVBar',
    'NewsEvent',
    'MacroEvent',
    'MarketUpdate',
    
    # Base Connector (Phase RT-2)
    'BaseConnector',
    'ConnectorStatus',
    'Order',
    'OrderSide',
    'OrderStatus',
    'OrderType',
    
    # Exchange Manager (Phase RT-2)
    'ExchangeManager',
    'FailoverStrategy',
    
    # Live Trading Orchestrator (Phase RT-3)
    'LiveTradingOrchestrator',
    'OrchestratorState',
    'SessionState',
    'ConnectorHealthEvent',
    'EngineHealthEvent',
    'GovernanceEvent',
    
    # Safety Layer (Phase RT-3)
    'SafetyLayer',
    'SafetyEvent',
    'SafetyEventType',
    'SafetySeverity',
    
    # Monitoring Dashboard (Phase RT-3)
    'MonitoringDashboard',
    'DashboardUpdate',
    'DashboardMode',
    
    # Logging (Phase RT-3)
    'LiveTradingLogManager',
    'get_logger'
]

__version__ = '1.0.0'
