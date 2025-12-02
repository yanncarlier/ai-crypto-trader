# exchanges/__init__.py
from typing import Union
from .base import BaseExchange
from .binance import BinanceFutures
from .bitunix import BitunixFutures
from .forward_tester import ForwardTester
from config.settings import TradingConfig
import logging


def create_exchange(config: TradingConfig, api_key: str = None, api_secret: str = None) -> BaseExchange:
    if config.FORWARD_TESTING:
        logging.info("Creating Forward Tester (Paper Trading)")
        return ForwardTester(config)
    exchange_type = config.EXCHANGE.upper()
    if exchange_type == 'BINANCE':
        logging.info(
            f"Creating Binance Futures client (Testnet: {config.TEST_NET})")
        return BinanceFutures(api_key, api_secret, config.TEST_NET)
    elif exchange_type == 'BITUNIX':
        logging.info("Creating Bitunix Futures client")
        return BitunixFutures(api_key, api_secret)
    else:
        raise ValueError(f"Unsupported exchange: {exchange_type}")
