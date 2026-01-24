"""
Logging Infrastructure - Phase RT-3

Centralized logging configuration for live trading components.

Logs organized by type:
  - live_trading_<timestamp>.log: Main trading engine log
  - safety_events_<timestamp>.log: Safety layer events
  - connector_health_<timestamp>.log: Connector health monitoring
  - governance_events_<timestamp>.log: Governance system events
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, Dict


class LiveTradingLogManager:
    """Manages logging for live trading components."""
    
    LOG_DIR = "logs/live"
    
    # Log formats
    SIMPLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DETAILED_FORMAT = "%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s"
    JSON_FORMAT = '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
    
    _loggers: Dict[str, logging.Logger] = {}
    _initialized = False
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize logging infrastructure."""
        if cls._initialized:
            return
        
        # Create logs directory if needed
        if not os.path.exists(cls.LOG_DIR):
            os.makedirs(cls.LOG_DIR)
        
        # Set root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        log_type: str = "general",
        level: int = logging.INFO
    ) -> logging.Logger:
        """
        Get or create a logger for a component.
        
        Args:
            name: Logger name (e.g., "LiveTradingOrchestrator")
            log_type: Type of log ('trading', 'safety', 'connector', 'governance', 'general')
            level: Logging level
        
        Returns:
            Configured logger instance
        """
        cls.initialize()
        
        logger_key = f"{name}_{log_type}"
        
        if logger_key in cls._loggers:
            return cls._loggers[logger_key]
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create handlers
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(cls.SIMPLE_FORMAT)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
        
        # File handler (type-specific)
        if log_type == "trading":
            filename = f"live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        elif log_type == "safety":
            filename = f"safety_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        elif log_type == "connector":
            filename = f"connector_health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        elif log_type == "governance":
            filename = f"governance_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        else:
            filename = f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        filepath = os.path.join(cls.LOG_DIR, filename)
        
        # Rotating file handler (10MB per file, keep last 5)
        file_handler = logging.handlers.RotatingFileHandler(
            filepath,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(cls.DETAILED_FORMAT)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        
        # Add handlers to logger
        for handler in handlers:
            logger.addHandler(handler)
        
        cls._loggers[logger_key] = logger
        return logger


def get_logger(
    name: str,
    log_type: str = "trading",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Convenience function to get a logger.
    
    Args:
        name: Logger name
        log_type: Type of log ('trading', 'safety', 'connector', 'governance', 'general')
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    return LiveTradingLogManager.get_logger(name, log_type, level)


# Initialize on module import
LiveTradingLogManager.initialize()
