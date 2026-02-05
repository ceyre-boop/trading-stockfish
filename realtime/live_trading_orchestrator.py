"""
Live Trading Orchestrator - Phase RT-3

Manages the complete trading session lifecycle with state machine,
component orchestration, session management, and error recovery.

State Machine:
    STARTUP → READY → LIVE → (DEGRADED) → FAILSAFE → SHUTDOWN

    STARTUP: Initializing all components
    READY:   All components healthy, waiting for market open
    LIVE:    Active trading, real-time processing
    DEGRADED: Anomaly detected, monitoring increased
    FAILSAFE: Critical anomaly, trading disabled, positions flattening
    SHUTDOWN: Graceful termination
"""

import asyncio
import logging
import queue
import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from engine import guardrails
from engine.connector_health import (
    ConnectorHealthConfig,
    ConnectorHealthMonitor,
    HealthEvent,
)

from .data_feed_router import DataFeedRouter
from .data_models import MarketUpdate, OrderBookSnapshot, PriceTick
from .engine_loop import RealTimeEngineLoop
from .exchange_manager import ExchangeManager
from .safety_layer import SafetyEvent, SafetyEventType, SafetyLayer


class OrchestratorState(Enum):
    """Trading orchestrator operational state."""

    STARTUP = "STARTUP"
    READY = "READY"
    LIVE = "LIVE"
    DEGRADED = "DEGRADED"
    FAILSAFE = "FAILSAFE"
    SHUTDOWN = "SHUTDOWN"


class SessionState(Enum):
    """Market session state."""

    PRE_MARKET = "PRE_MARKET"
    MARKET_OPEN = "MARKET_OPEN"
    CONTINUOUS = "CONTINUOUS"
    MARKET_CLOSE = "MARKET_CLOSE"
    POST_MARKET = "POST_MARKET"


@dataclass
class ConnectorHealthEvent:
    """Connector health status event."""

    timestamp: float
    connector_name: str
    is_connected: bool
    latency_ms: float
    messages_received: int
    messages_dropped: int
    error_count: int
    last_update: Optional[float]


@dataclass
class EngineHealthEvent:
    """Engine health status event."""

    timestamp: float
    decisions_made: int
    orders_submitted: int
    orders_filled: int
    orders_rejected: int
    latency_p50_ms: float
    latency_p99_ms: float
    error_count: int


@dataclass
class GovernanceEvent:
    """Governance system event."""

    timestamp: float
    event_type: str  # 'position_limit', 'loss_limit', 'rate_limit'
    symbol: str
    current_value: float
    limit_value: float
    severity: str  # 'info', 'warning', 'critical'


@dataclass
class OrchestratorStats:
    """Statistics for orchestrator session."""

    start_time: float
    session_duration: float = 0.0
    state_changes: int = 0
    connector_failures: int = 0
    connector_recoveries: int = 0
    safety_events: int = 0
    failsafe_activations: int = 0
    total_trades: int = 0
    successful_trades: int = 0
    rejected_trades: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    peak_drawdown: float = 0.0
    uptime_percentage: float = 100.0


class LiveTradingOrchestrator:
    """
    Main orchestrator for live trading sessions.

    Manages:
    - Component lifecycle (initialization, start, stop)
    - State transitions and consistency
    - Session management (pre-market, market open, continuous, close)
    - Connector health monitoring
    - Safety layer integration
    - Error recovery and failsafe transitions
    - Comprehensive logging and monitoring
    """

    def __init__(
        self,
        exchange_manager: ExchangeManager,
        engine_loop: RealTimeEngineLoop,
        data_feed_router: DataFeedRouter,
        safety_layer: SafetyLayer,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LiveTradingOrchestrator.

        Args:
            exchange_manager: Multi-connector exchange manager
            engine_loop: Real-time trading engine
            data_feed_router: Data feed routing layer
            safety_layer: Safety checks and anomaly detection
            logger: Optional logger instance
            config: Optional configuration overrides
        """
        self.exchange_manager = exchange_manager
        self.engine_loop = engine_loop
        self.data_feed_router = data_feed_router
        self.safety_layer = safety_layer

        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}

        # State management
        self._state = OrchestratorState.STARTUP
        self._state_lock = threading.Lock()
        self._session_state = SessionState.PRE_MARKET

        # Component status tracking
        self._connector_health: Dict[str, ConnectorHealthEvent] = {}
        self._last_health_check: Dict[str, float] = {}
        self._health_check_interval = self.config.get("health_check_interval_s", 5.0)

        # Event handling
        self._connector_event_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._health_event_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._governance_event_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._safety_event_queue: queue.Queue = queue.Queue(maxsize=1000)

        # Callback management
        self._on_state_change_callbacks: List[Callable] = []
        self._on_safety_event_callbacks: List[Callable] = []
        self._on_health_degradation_callbacks: List[Callable] = []

        # Threading
        self._run_thread: Optional[threading.Thread] = None
        self._health_monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Statistics
        self.stats = OrchestratorStats(start_time=time.time())

        # Failsafe configuration
        self._failsafe_degradation_count = self.config.get(
            "failsafe_degradation_count", 3
        )
        self._consecutive_degradations = 0
        self._last_state_change = time.time()

        # Connector health monitoring (Phase 11)
        health_cfg = self.config.get("connector_health", {})
        self.connector_health_monitor = ConnectorHealthMonitor(
            ConnectorHealthConfig(
                heartbeat_threshold=timedelta(
                    seconds=health_cfg.get("heartbeat_threshold_s", 5)
                ),
                latency_threshold_ms=float(
                    health_cfg.get("latency_threshold_ms", 1500.0)
                ),
                failure_threshold=int(health_cfg.get("failure_threshold", 3)),
                stale_data_threshold=timedelta(
                    seconds=health_cfg.get("stale_data_threshold_s", 5)
                ),
            )
        )

        self.exchange_manager.attach_health_monitor(
            self.connector_health_monitor,
            guardrail_callback=self._guardrail_callback,
            safe_mode_callback=self._safe_mode_callback,
            anomaly_callback=self._anomaly_callback,
            enable_safe_mode=True,
            evaluation_interval=self.config.get("health_check_interval_s", 1.0),
        )

        self.logger.info(
            "LiveTradingOrchestrator initialized with state=%s, "
            "health_check_interval=%.1fs",
            self._state.value,
            self._health_check_interval,
        )

    # ===== State Management =====

    def get_state(self) -> OrchestratorState:
        """Get current orchestrator state."""
        with self._state_lock:
            return self._state

    def _set_state(self, new_state: OrchestratorState) -> bool:
        """
        Transition to new state with validation.

        Returns True if state change successful, False otherwise.
        """
        with self._state_lock:
            old_state = self._state

            # Validate state transition
            if not self._validate_state_transition(old_state, new_state):
                self.logger.warning(
                    "Invalid state transition: %s → %s",
                    old_state.value,
                    new_state.value,
                )
                return False

            self._state = new_state
            self._last_state_change = time.time()
            self.stats.state_changes += 1

            self.logger.info(
                "State transition: %s → %s", old_state.value, new_state.value
            )

            # Trigger callbacks
            for callback in self._on_state_change_callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    self.logger.error("State change callback error: %s", e)

            return True

    @staticmethod
    def _validate_state_transition(
        from_state: OrchestratorState, to_state: OrchestratorState
    ) -> bool:
        """Validate if state transition is allowed."""
        valid_transitions = {
            OrchestratorState.STARTUP: [
                OrchestratorState.READY,
                OrchestratorState.SHUTDOWN,
            ],
            OrchestratorState.READY: [
                OrchestratorState.LIVE,
                OrchestratorState.SHUTDOWN,
            ],
            OrchestratorState.LIVE: [
                OrchestratorState.DEGRADED,
                OrchestratorState.SHUTDOWN,
            ],
            OrchestratorState.DEGRADED: [
                OrchestratorState.LIVE,
                OrchestratorState.FAILSAFE,
                OrchestratorState.SHUTDOWN,
            ],
            OrchestratorState.FAILSAFE: [
                OrchestratorState.READY,
                OrchestratorState.SHUTDOWN,
            ],
            OrchestratorState.SHUTDOWN: [],
        }
        return to_state in valid_transitions.get(from_state, [])

    # ===== Session Management =====

    def start(self) -> bool:
        """
        Start live trading session.

        Returns True if successful, False otherwise.
        """
        if self._running:
            self.logger.warning("Session already running")
            return False

        try:
            self.logger.info("Starting live trading session")

            # Initialize components
            if not self._initialize_components():
                self.logger.error("Component initialization failed")
                return False

            # Transition to READY
            if not self._set_state(OrchestratorState.READY):
                self.logger.error("Failed to transition to READY state")
                return False

            # Start background threads
            self._stop_event.clear()
            self._running = True

            self._health_monitor_thread = threading.Thread(
                target=self._health_monitor_loop,
                daemon=True,
                name="OrchestratorHealthMonitor",
            )
            self._health_monitor_thread.start()

            self._run_thread = threading.Thread(
                target=self._main_loop, daemon=True, name="OrchestratorMainLoop"
            )
            self._run_thread.start()

            self.logger.info("Live trading session started successfully")
            return True

        except Exception as e:
            self.logger.error("Error starting session: %s", e, exc_info=True)
            self._running = False
            return False

    def stop(self) -> bool:
        """
        Stop live trading session gracefully.

        Returns True if successful, False otherwise.
        """
        if not self._running:
            self.logger.warning("Session not running")
            return False

        try:
            self.logger.info("Stopping live trading session")

            # Signal stop
            self._stop_event.set()
            self._running = False

            # Wait for threads
            if self._run_thread and self._run_thread.is_alive():
                self._run_thread.join(timeout=10.0)

            if self._health_monitor_thread and self._health_monitor_thread.is_alive():
                self._health_monitor_thread.join(timeout=5.0)

            # Shutdown components
            self._shutdown_components()

            # Transition to SHUTDOWN
            self._set_state(OrchestratorState.SHUTDOWN)

            self.logger.info("Live trading session stopped")
            return True

        except Exception as e:
            self.logger.error("Error stopping session: %s", e, exc_info=True)
            return False

    def _initialize_components(self) -> bool:
        """Initialize all trading components."""
        try:
            self.logger.info("Initializing trading components")

            # Start exchange manager
            if not self.exchange_manager.start_all():
                self.logger.error("Failed to start exchange manager")
                return False

            # Start data feed router
            if not self.data_feed_router.start():
                self.logger.error("Failed to start data feed router")
                self.exchange_manager.stop_all()
                return False

            # Start engine loop
            if not self.engine_loop.start():
                self.logger.error("Failed to start engine loop")
                self.data_feed_router.stop()
                self.exchange_manager.stop_all()
                return False

            # Initialize safety layer
            self.safety_layer.reset()

            self.logger.info("All components initialized successfully")
            return True

        except Exception as e:
            self.logger.error("Component initialization error: %s", e, exc_info=True)
            return False

    def _shutdown_components(self) -> None:
        """Shutdown all trading components."""
        try:
            self.logger.info("Shutting down trading components")

            # Stop in reverse order
            if self.engine_loop and self.engine_loop.is_running():
                self.engine_loop.stop()

            if self.data_feed_router and self.data_feed_router.is_running():
                self.data_feed_router.stop()

            if self.exchange_manager:
                self.exchange_manager.stop_all()

            self.logger.info("All components shut down")

        except Exception as e:
            self.logger.error("Component shutdown error: %s", e, exc_info=True)

    # ===== Main Event Loops =====

    def _main_loop(self) -> None:
        """Main orchestrator loop handling events and state transitions."""
        try:
            self.logger.info("Orchestrator main loop started")

            while not self._stop_event.is_set():
                try:
                    # Process safety events
                    if not self._safety_event_queue.empty():
                        event = self._safety_event_queue.get_nowait()
                        self._handle_safety_event(event)

                    # Process health events
                    if not self._health_event_queue.empty():
                        event = self._health_event_queue.get_nowait()
                        self._handle_health_event(event)

                    # Process connector events
                    if not self._connector_event_queue.empty():
                        event = self._connector_event_queue.get_nowait()
                        self._handle_connector_event(event)

                    # Update statistics
                    self.stats.session_duration = time.time() - self.stats.start_time

                    # Sleep briefly to avoid busy-waiting
                    time.sleep(0.01)

                except queue.Empty:
                    time.sleep(0.01)
                except Exception as e:
                    self.logger.error("Main loop error: %s", e, exc_info=True)
                    time.sleep(0.1)

        except Exception as e:
            self.logger.error("Fatal main loop error: %s", e, exc_info=True)
        finally:
            self.logger.info("Orchestrator main loop exited")

    def _health_monitor_loop(self) -> None:
        """Health monitoring loop checking component status."""
        try:
            self.logger.info("Health monitor loop started")

            while not self._stop_event.is_set():
                try:
                    current_time = time.time()

                    # Check connector health
                    self._check_connector_health()

                    # Check engine health
                    self._check_engine_health()

                    # Assess overall health
                    self._assess_overall_health()

                    # Sleep until next check
                    time.sleep(self._health_check_interval)

                except Exception as e:
                    self.logger.error("Health monitor error: %s", e, exc_info=True)
                    time.sleep(1.0)

        except Exception as e:
            self.logger.error("Fatal health monitor error: %s", e, exc_info=True)
        finally:
            self.logger.info("Health monitor loop exited")

    # ===== Health Monitoring =====

    def _check_connector_health(self) -> None:
        """Check health of all exchange connectors."""
        try:
            for connector_name, connector in self.exchange_manager._connectors.items():
                current_time = time.time()
                last_check = self._last_health_check.get(connector_name, 0)

                # Check at appropriate intervals
                if current_time - last_check < self._health_check_interval * 0.8:
                    continue

                # Perform health check
                is_connected = connector.is_connected_check()
                stats = connector.get_stats()

                event = ConnectorHealthEvent(
                    timestamp=current_time,
                    connector_name=connector_name,
                    is_connected=is_connected,
                    latency_ms=stats.get("latency_ms", 0),
                    messages_received=stats.get("messages_received", 0),
                    messages_dropped=stats.get("messages_dropped", 0),
                    error_count=stats.get("error_count", 0),
                    last_update=stats.get("last_update", None),
                )

                self._connector_health[connector_name] = event
                self._last_health_check[connector_name] = current_time

                # Enqueue for processing
                try:
                    self._health_event_queue.put_nowait(event)
                except queue.Full:
                    self.logger.warning("Health event queue full, dropping event")

        except Exception as e:
            self.logger.error("Connector health check error: %s", e, exc_info=True)

    def _check_engine_health(self) -> None:
        """Check health of trading engine."""
        try:
            stats = self.engine_loop.get_stats()

            event = EngineHealthEvent(
                timestamp=time.time(),
                decisions_made=stats.get("decisions_made", 0),
                orders_submitted=stats.get("orders_submitted", 0),
                orders_filled=stats.get("orders_filled", 0),
                orders_rejected=stats.get("orders_rejected", 0),
                latency_p50_ms=stats.get("latency_p50_ms", 0),
                latency_p99_ms=stats.get("latency_p99_ms", 0),
                error_count=stats.get("error_count", 0),
            )

            try:
                self._health_event_queue.put_nowait(event)
            except queue.Full:
                self.logger.warning("Health event queue full, dropping event")

        except Exception as e:
            self.logger.error("Engine health check error: %s", e, exc_info=True)

    def _assess_overall_health(self) -> None:
        """Assess overall system health and manage state transitions."""
        try:
            current_state = self.get_state()

            # Don't transition if already shutdown
            if current_state == OrchestratorState.SHUTDOWN:
                return

            # Check if any critical failures
            critical_failures = 0
            all_connectors_down = True

            for connector_health in self._connector_health.values():
                if connector_health.is_connected:
                    all_connectors_down = False

                # Check for repeated errors
                if connector_health.error_count > 10:
                    critical_failures += 1

            # Handle all connectors down
            if all_connectors_down and current_state in [
                OrchestratorState.LIVE,
                OrchestratorState.READY,
            ]:
                self.logger.critical(
                    "All connectors disconnected - entering DEGRADED state"
                )
                self.stats.connector_failures += 1
                self._set_state(OrchestratorState.DEGRADED)
                self._consecutive_degradations += 1

            # Handle critical failures
            if critical_failures > 1 and current_state == OrchestratorState.LIVE:
                self.logger.critical(
                    "Multiple connector failures - entering DEGRADED state"
                )
                self.stats.connector_failures += 1
                self._set_state(OrchestratorState.DEGRADED)
                self._consecutive_degradations += 1

            # Trigger failsafe if degradation persists
            if self._consecutive_degradations >= self._failsafe_degradation_count:
                if current_state == OrchestratorState.DEGRADED:
                    self.logger.critical(
                        "Repeated degradation (%d times) - entering FAILSAFE state",
                        self._consecutive_degradations,
                    )
                    self.stats.failsafe_activations += 1
                    self._set_state(OrchestratorState.FAILSAFE)
                    self._trigger_failsafe()

            # Reset degradation count if recovered
            if all_connectors_down is False and current_state == OrchestratorState.LIVE:
                if self._consecutive_degradations > 0:
                    self.logger.info("System recovered - resetting degradation counter")
                    self._consecutive_degradations = 0

        except Exception as e:
            self.logger.error("Health assessment error: %s", e, exc_info=True)

    # ===== Connector Health Callbacks (Phase 11) =====

    def _guardrail_callback(self, event: HealthEvent) -> None:
        """Forward connector health events to guardrail engine."""
        try:
            decision = guardrails.GuardrailDecision(
                triggered=True,
                reason=f"connector_health:{event.event_type}",
                guardrail_type=event.event_type,
                safe_mode_required=event.safe_mode_required,
                timestamp=event.timestamp,
            )
            guardrails.apply_guardrail_decision(decision)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Guardrail callback failed: %s", exc, exc_info=True)

    def _safe_mode_callback(self, event: HealthEvent) -> str:
        """Trigger SAFE_MODE when required."""
        try:
            if event.safe_mode_required:
                return guardrails._activate_safe_mode()  # type: ignore[attr-defined]
            return "NOT_REQUIRED"
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("SAFE_MODE callback failed: %s", exc, exc_info=True)
            return "ERROR"

    def _anomaly_callback(self, event: HealthEvent) -> None:
        """Forward anomalies to safety layer/anomaly detector."""
        try:
            severity = "warning"
            if event.safe_mode_required:
                severity = "critical"

            safety_event_type = SafetyEventType.DATA_ANOMALY
            if event.event_type == "stale_data":
                safety_event_type = SafetyEventType.FEED_STALE

            safety_event = SafetyEvent(
                timestamp=time.time(),
                event_type=safety_event_type,
                severity=severity,
                symbol=event.connector_name,
                message=f"connector_health:{event.event_type}",
                data={"details": event.details},
            )
            self.on_safety_event(safety_event)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Anomaly callback failed: %s", exc, exc_info=True)

    # ===== Event Handlers =====

    def on_connector_event(self, event: ConnectorHealthEvent) -> None:
        """Handle connector health event."""
        try:
            self._connector_event_queue.put_nowait(event)
        except queue.Full:
            self.logger.warning("Connector event queue full, dropping event")

    def on_health_event(self, event: EngineHealthEvent) -> None:
        """Handle engine health event."""
        try:
            self._health_event_queue.put_nowait(event)
        except queue.Full:
            self.logger.warning("Health event queue full, dropping event")

    def on_governance_event(self, event: GovernanceEvent) -> None:
        """Handle governance system event."""
        try:
            self._governance_event_queue.put_nowait(event)
        except queue.Full:
            self.logger.warning("Governance event queue full, dropping event")

    def on_safety_event(self, event: SafetyEvent) -> None:
        """Handle safety layer event."""
        try:
            self._safety_event_queue.put_nowait(event)
        except queue.Full:
            self.logger.warning("Safety event queue full, dropping event")

    def _handle_connector_event(self, event: ConnectorHealthEvent) -> None:
        """Process connector health event."""
        try:
            if not event.is_connected:
                self.logger.warning(
                    "Connector %s disconnected: latency=%.1fms, errors=%d",
                    event.connector_name,
                    event.latency_ms,
                    event.error_count,
                )
                self.stats.connector_failures += 1
            else:
                self.logger.debug(
                    "Connector %s healthy: latency=%.1fms, messages=%d",
                    event.connector_name,
                    event.latency_ms,
                    event.messages_received,
                )
                if self.stats.connector_failures > 0:
                    self.stats.connector_recoveries += 1

        except Exception as e:
            self.logger.error("Connector event handling error: %s", e, exc_info=True)

    def _handle_health_event(self, event: EngineHealthEvent) -> None:
        """Process engine health event."""
        try:
            if event.error_count > 0:
                self.logger.warning(
                    "Engine errors detected: count=%d, latency_p99=%.1fms",
                    event.error_count,
                    event.latency_p99_ms,
                )

            # Track trades
            self.stats.total_trades = event.orders_submitted
            self.stats.successful_trades = event.orders_filled
            self.stats.rejected_trades = event.orders_rejected

        except Exception as e:
            self.logger.error("Health event handling error: %s", e, exc_info=True)

    def _handle_safety_event(self, event: SafetyEvent) -> None:
        """Process safety layer event."""
        try:
            self.stats.safety_events += 1

            self.logger.warning(
                "Safety event: type=%s, severity=%s, message=%s",
                event.event_type.value,
                event.severity,
                event.message,
            )

            # Trigger degradation on safety anomalies
            if event.event_type in [
                SafetyEventType.DATA_ANOMALY,
                SafetyEventType.EXECUTION_ANOMALY,
                SafetyEventType.FEED_STALE,
            ]:
                current_state = self.get_state()
                if current_state == OrchestratorState.LIVE:
                    self.logger.warning("Transitioning to DEGRADED due to safety event")
                    self._set_state(OrchestratorState.DEGRADED)

            # Trigger failsafe on critical events
            if event.severity == "critical":
                self.logger.critical("Critical safety event - triggering failsafe")
                self._trigger_failsafe()

            # Trigger callbacks
            for callback in self._on_safety_event_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error("Safety event callback error: %s", e)

        except Exception as e:
            self.logger.error("Safety event handling error: %s", e, exc_info=True)

    # ===== Failsafe & Recovery =====

    def _trigger_failsafe(self) -> None:
        """
        Trigger failsafe: flatten positions and disable trading.
        """
        try:
            self.logger.critical(
                "FAILSAFE TRIGGERED - Flattening positions and disabling trading"
            )

            # Flatten all positions
            try:
                flatten_result = self.engine_loop.flatten_positions()
                self.logger.info("Positions flattened: %s", flatten_result)
            except Exception as e:
                self.logger.error("Error flattening positions: %s", e)

            # Disable trading in engine
            try:
                self.engine_loop.disable_trading()
                self.logger.info("Trading disabled")
            except Exception as e:
                self.logger.error("Error disabling trading: %s", e)

            # Transition to FAILSAFE state
            self._set_state(OrchestratorState.FAILSAFE)

        except Exception as e:
            self.logger.error("Failsafe trigger error: %s", e, exc_info=True)

    def recover_from_degraded(self) -> bool:
        """
        Attempt to recover from DEGRADED state back to LIVE.

        Returns True if recovery successful, False otherwise.
        """
        try:
            current_state = self.get_state()

            if current_state != OrchestratorState.DEGRADED:
                self.logger.warning("Cannot recover - not in DEGRADED state")
                return False

            self.logger.info("Attempting recovery from DEGRADED state")

            # Reset safety layer
            self.safety_layer.reset()

            # Verify connector health
            healthy_connectors = 0
            for connector_health in self._connector_health.values():
                if connector_health.is_connected:
                    healthy_connectors += 1

            if healthy_connectors == 0:
                self.logger.warning("No healthy connectors - cannot recover")
                return False

            # Reset degradation counter
            self._consecutive_degradations = 0

            # Transition to LIVE
            success = self._set_state(OrchestratorState.LIVE)

            if success:
                self.logger.info("Recovery successful - returned to LIVE state")

            return success

        except Exception as e:
            self.logger.error("Recovery error: %s", e, exc_info=True)
            return False

    # ===== Callbacks =====

    def register_state_change_callback(
        self, callback: Callable[[OrchestratorState, OrchestratorState], None]
    ) -> None:
        """Register callback for state changes."""
        self._on_state_change_callbacks.append(callback)

    def register_safety_event_callback(
        self, callback: Callable[[SafetyEvent], None]
    ) -> None:
        """Register callback for safety events."""
        self._on_safety_event_callbacks.append(callback)

    def register_health_degradation_callback(
        self, callback: Callable[[str], None]
    ) -> None:
        """Register callback for health degradation."""
        self._on_health_degradation_callbacks.append(callback)

    # ===== Utilities =====

    def get_connector_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all connectors."""
        result = {}
        for name, health in self._connector_health.items():
            result[name] = {
                "is_connected": health.is_connected,
                "latency_ms": health.latency_ms,
                "messages_received": health.messages_received,
                "messages_dropped": health.messages_dropped,
                "error_count": health.error_count,
                "timestamp": datetime.fromtimestamp(health.timestamp).isoformat(),
            }
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "state": self.get_state().value,
            "session_duration": self.stats.session_duration,
            "state_changes": self.stats.state_changes,
            "connector_failures": self.stats.connector_failures,
            "connector_recoveries": self.stats.connector_recoveries,
            "safety_events": self.stats.safety_events,
            "failsafe_activations": self.stats.failsafe_activations,
            "total_trades": self.stats.total_trades,
            "successful_trades": self.stats.successful_trades,
            "rejected_trades": self.stats.rejected_trades,
            "realized_pnl": self.stats.realized_pnl,
            "unrealized_pnl": self.stats.unrealized_pnl,
            "peak_drawdown": self.stats.peak_drawdown,
            "uptime_percentage": self.stats.uptime_percentage,
        }

    def is_running(self) -> bool:
        """Check if session is running."""
        return self._running

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        if exc_type is not None:
            self.logger.error(
                "Context manager exception: %s: %s", exc_type.__name__, exc_val
            )
        return False
