# exchanges/__init__.py
from typing import Union
from .base import BaseExchange, Position
from .bitunix import BitunixFutures
from .forward_tester import ForwardTester
from config.settings import TradingConfig
import logging

__all__ = ['BaseExchange', 'BitunixFutures', 'ForwardTester', 'Position']


def create_exchange(config: TradingConfig, api_key: str = None, api_secret: str = None) -> BaseExchange:
    if config.FORWARD_TESTING:
        logging.info("Creating Forward Tester (Paper Trading)")
        return ForwardTester(config)
    logging.info("Creating Bitunix Futures client")
    return BitunixFutures(api_key, api_secret, config)