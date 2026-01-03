# utils/logger.py
import logging
import os
from datetime import datetime


def configure_logger(run_name: str):
    """Configure informative console logging (trades + key info + errors)"""
    # Clean root handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
    # Informative console: trade INFO+, app INFO+, root ERROR+
    trade_logger = logging.getLogger('trade')
    trade_logger.setLevel(logging.INFO)
    trade_logger.propagate = False
    console_trade = logging.StreamHandler()
    console_trade.setFormatter(formatter)
    console_trade.setLevel(logging.INFO)
    trade_logger.addHandler(console_trade)

    app_logger = logging.getLogger('app')
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False
    console_app = logging.StreamHandler()
    console_app.setFormatter(formatter)
    console_app.setLevel(logging.INFO)
    app_logger.addHandler(console_app)

    console_root = logging.StreamHandler()
    console_root.setFormatter(formatter)
    console_root.setLevel(logging.ERROR)
    logging.root.addHandler(console_root)
    logging.root.setLevel(logging.ERROR)
