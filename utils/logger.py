# utils/logger.py
import logging
import os
from datetime import datetime


def configure_logger(run_name: str):
    """Configure segregated logging: files by type (app/ai/http/trade/market), minimal console (trades + errors)"""
    # Clean root handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    formatter = logging.Formatter(
        '%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    loggers_config = {
        'app': 'app.log',
        'ai': 'ai.log',
        'http': 'http.log',
        'trade': 'trade.log',
        'market': 'market.log',
    }
    for name, fname in loggers_config.items():
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        file_path = os.path.join(log_dir, fname)
        fh = logging.FileHandler(file_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    # Minimal console: trade INFO+, root ERROR+
    trade_logger = logging.getLogger('trade')
    console_trade = logging.StreamHandler()
    console_trade.setFormatter(formatter)
    console_trade.setLevel(logging.INFO)
    trade_logger.addHandler(console_trade)
    console_root = logging.StreamHandler()
    console_root.setFormatter(formatter)
    console_root.setLevel(logging.ERROR)
    logging.root.addHandler(console_root)
    logging.root.setLevel(logging.ERROR)
