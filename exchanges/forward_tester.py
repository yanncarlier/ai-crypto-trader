# exchanges/forward_tester.py
import random
import logging
from typing import Optional
from .base import BaseExchange, Position


class ForwardTester(BaseExchange):
    def __init__(self, config):
        self.config = config
        self.balance = config.INITIAL_CAPITAL
        self.position: Optional[Position] = None
        self.price = 67_000.0
        self.volatility = 0.004

    def get_current_price(self, symbol: str) -> float:
        change = random.gauss(0, self.volatility)
        self.price *= (1 + change)
        self.price = max(self.price, 20_000.0)
        logging.info(f"[Paper] {symbol} price: ${self.price:,.2f}")
        return round(self.price, 2)

    def get_account_balance(self, currency: str) -> float:
        return round(self.balance, 2)

    def get_pending_positions(self, symbol: str) -> Optional[Position]:
        return self.position

    def set_leverage(self, symbol: str, leverage: int):
        logging.info(f"[Paper] Leverage: {leverage}x")

    def set_margin_mode(self, symbol: str, mode: str):
        logging.info(f"[Paper] Margin mode: {mode}")

    def open_position(self, symbol: str, side: str, size: str, sl_pct: Optional[int]):
        price = self.get_current_price(symbol)
        self.position = Position(
            positionId="paper_001",
            side=side.upper(),
            size=0.15,  # Simulated size
            entry_price=price,
            symbol=symbol
        )
        logging.info(f"[Paper] Opened {side.upper()} @ ${price:,.2f}")

    def flash_close_position(self, symbol: str):
        if self.position:
            logging.info(f"[Paper] Closed position at market")
            self.position = None
    # Add this method inside ForwardTester class
    # def exchange(self):
    #     # Dummy object to satisfy fetch_ohlcv call in paper mode
    #     class Dummy:
    #         def fetch_ohlcv(self, symbol, timeframe, limit):
    #             # Simulate 15 minutes of fake price data
    #             import time
    #             import random
    #             base_price = self.price
    #             data = []
    #             for i in range(limit):
    #                 price = base_price * (1 + random.gauss(0, 0.001))
    #                 data.append([time.time() - (limit-i)*60,
    #                             price, price, price, price, 1000000])
    #             return data
    #     return Dummy()
    # DELETE the old exchange() method and add this instead:

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 15):
        """Simulate OHLCV for paper trading"""
        import time
        base_price = self.price
        data = []
        for i in range(limit):
            price = base_price * (1 + random.gauss(0, 0.001))
            timestamp = int(time.time() - (limit - i) * 60)
            data.append([timestamp, price, price, price,
                        price, 100000000])  # high volume
        return data
