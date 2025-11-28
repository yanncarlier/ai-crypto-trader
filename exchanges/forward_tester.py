# exchanges/forward_tester.py
import random
import logging
from typing import Optional
import ccxt  # Add this import
from .base import BaseExchange, Position


class ForwardTester(BaseExchange):
    def __init__(self, config):
        self.config = config
        self.balance = config.INITIAL_CAPITAL
        self.position: Optional[Position] = None
        self.volatility = 0.004  # Keep for minor noise if desired, or set to 0
        self.exchange = ccxt.binance({
            'options': {
                'defaultType': 'future',  # For USDT-margined futures
                'adjustForTimeDifference': True,
            },
            'enableRateLimit': True,
        })
        self.exchange.load_markets()  # Initialize markets

    def get_current_price(self, symbol: str) -> float:
        ticker = self.exchange.fetch_ticker(symbol)
        price = float(ticker['last'])
        # Optional: Add tiny simulation noise if you want (uncomment if desired)
        # change = random.gauss(0, self.volatility)
        # price *= (1 + change)
        logging.info(f"[Paper] {symbol} price: ${price:,.2f}")
        return round(price, 2)

    def get_account_balance(self, currency: str) -> float:
        return round(self.balance, 2)

    def get_pending_positions(self, symbol: str) -> Optional[Position]:
        return self.position

    def set_leverage(self, symbol: str, leverage: int):
        logging.info(f"[Paper] Leverage: {leverage}x")

    def set_margin_mode(self, symbol: str, mode: str):
        logging.info(f"[Paper] Margin mode: {mode}")

    def open_position(self, symbol: str, side: str, size: str, sl_pct: Optional[int]):
        price = self.get_current_price(symbol)  # Use real price
        if size.endswith("%"):
            pct = float(size[:-1]) / 100
            usdt_value = self.balance * pct
        else:
            usdt_value = float(size)
        # For Binance futures, contract size is 1 for BTCUSDT (1 contract = 1 USD worth of BTC)
        # Adjust qty calculation accordingly (contracts = usdt_value / (price * contract_size))
        market = self.exchange.market(symbol)
        contract_size = market['contractSize'] or 1
        qty = self.exchange.amount_to_precision(
            symbol, usdt_value / (price * contract_size))
        self.position = Position(
            positionId="paper_001",
            side=side.upper(),
            size=float(qty),  # Use calculated qty
            entry_price=price,
            symbol=symbol
        )
        logging.info(
            f"[Paper] Opened {side.upper()} {qty} contracts @ ${price:,.2f}")

    def flash_close_position(self, symbol: str):
        if self.position:
            logging.info(f"[Paper] Closed position at market")
            self.position = None

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 15):
        return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
