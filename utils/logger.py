# utils/logger.py
import logging
from datetime import datetime
from pathlib import Path


def configure_logger(run_name: str):
    """Configure minimal logging to file and console"""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    (logs_dir / "ai_responses").mkdir(exist_ok=True)
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Simple formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    # File handler
    date_str = datetime.now().strftime('%Y%m%d')
    log_file = logs_dir / f"{run_name}_{date_str}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
