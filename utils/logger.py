# utils/logger.py
import logging
from datetime import datetime


def configure_logger(run_name: str):
    """Configure logging to console only"""
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Simple formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    # Console handler only
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler]
    )
