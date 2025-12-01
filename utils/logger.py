# utils/logger.py
import logging
import os
from datetime import datetime
from pathlib import Path


def configure_logger(run_name: str):
    """Configure logging to use logs/ directory with both console and file output"""
    # Create logs directory and subdirectories
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    # Create subdirectories
    (logs_dir / "ai_responses").mkdir(exist_ok=True)
    (logs_dir / "trading").mkdir(exist_ok=True)
    # Remove all existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # File handler for all logs
    date_str = datetime.now().strftime('%Y%m%d')
    log_file = logs_dir / f"{run_name}_{date_str}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    # Console handler with colors

    class ColorFormatter(logging.Formatter):
        """Custom formatter with colors for different log levels"""
        GREY = "\x1b[38;21m"
        BLUE = "\x1b[38;5;39m"
        YELLOW = "\x1b[38;5;226m"
        RED = "\x1b[38;5;196m"
        BOLD_RED = "\x1b[31;1m"
        RESET = "\x1b[0m"
        FORMATS = {
            logging.DEBUG: GREY + '%(asctime)s | %(levelname)-8s | %(message)s' + RESET,
            logging.INFO: BLUE + '%(asctime)s | %(levelname)-8s | %(message)s' + RESET,
            logging.WARNING: YELLOW + '%(asctime)s | %(levelname)-8s | %(message)s' + RESET,
            logging.ERROR: RED + '%(asctime)s | %(levelname)-8s | %(message)s' + RESET,
            logging.CRITICAL: BOLD_RED +
            '%(asctime)s | %(levelname)-8s | %(message)s' + RESET
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
            return formatter.format(record)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter())
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    # Log startup information
    logging.info("=" * 60)
    logging.info(f"🚀 Starting Trading Bot: {run_name}")
    logging.info(f"📁 Log file: {log_file}")
    logging.info(
        f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 60)
