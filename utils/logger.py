# utils/logger.py
import logging
import os
from datetime import datetime


def configure_logger(run_name: str):
    """Configure minimal console logging (trades + errors)"""
    # Clean root handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    formatter = logging.Formatter(
        '%(asctime)s | %(message)s', datefmt='%H:%M:%S')
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
