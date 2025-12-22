# exchanges/__init__.py
from typing import Union, Dict, Any
from .base import BaseExchange, Position
from .bitunix import BitunixFutures
from .forward_tester import ForwardTester
import logging

__all__ = ['BaseExchange', 'BitunixFutures', 'ForwardTester', 'Position']


def create_exchange(config: Dict[str, Any], api_key: str = None, api_secret: str = None) -> BaseExchange:
    """Create an exchange instance based on configuration.
    
    Args:
        config: Dictionary containing configuration parameters
        api_key: Optional API key (if not in config)
        api_secret: Optional API secret (if not in config)
    """
    if config.get('FORWARD_TESTING'):
        logging.info("Creating Forward Tester (Paper Trading)")
        return ForwardTester(config)
        
    logging.info("Creating Bitunix Futures client")
    api_key = api_key or config.get('EXCHANGE_API_KEY')
    api_secret = api_secret or config.get('EXCHANGE_API_SECRET')
    return BitunixFutures(api_key, api_secret, config)