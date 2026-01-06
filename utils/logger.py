# utils/logger.py
import logging
import os
from datetime import datetime
from typing import Optional


def configure_logger(run_name: str, log_level: str = "INFO"):
    """Configure unified console logging with consistent formatting across all modules.
    
    Args:
        run_name: Name for this run (used for file logging if enabled)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Clean root handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Standardized formatter for all loggers
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s', 
        datefmt='%H:%M:%S'
    )
    
    # Console handler with configurable level
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Root logger configuration
    logging.root.addHandler(console_handler)
    logging.root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Configure specific loggers with consistent behavior
    logger_configs = {
        'trade': logging.INFO,      # Trade execution logging
        'app': logging.INFO,         # Application events
        'ai': logging.INFO,          # AI decision logging
        'risk': logging.INFO,          # Risk management logging
        'exchange': logging.WARNING, # Exchange operations
        'position': logging.INFO,    # Position management
    }
    
    for logger_name, level in logger_configs.items():
        logger = logging.getLogger(logger_name)
        logger.propagate = True  # Allow propagation to root logger
        # Console level is controlled by root handler
