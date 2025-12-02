# exchanges/forward_tester.py
import random
import logging
from typing import Optional, List, Dict, Any
import ccxt
from .base import BaseExchange, Position


class ForwardTester(BaseExchange):
    def __init__(self, config):
        self.config = config
        self.balance = config.INITIAL_CAPITAL
        self.positions: List[Position] = []
        self.exchange = ccxt.binance({
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            },
            'enableRateLimit': True,
        })
        self.exchange.load_markets()

    def get_current_price(self, symbol: str) -> float:
        ticker = self.exchange.fetch_ticker(symbol)
        price = float(ticker['last'])
        return round(price, 2)

    def get_account_balance(self, currency: str) -> float:
        return round(self.balance, 2)

    def get_pending_positions(self, symbol: str) -> Optional[Position]:
        if self.positions:
            return self.positions[0]
        return None

    def get_all_positions(self, symbol: str) -> List[Position]:
        return self.positions

    def get_account_summary(self, currency: str, symbol: str) -> Dict[str, Any]:
        """Get complete account summary including balance and positions"""
        current_price = self.get_current_price(symbol)
        total_exposure = 0.0
        for position in self.positions:
            position_value = position.size * current_price
            total_exposure += position_value
            # Calculate PnL
            pnl = (current_price - position.entry_price) * position.size
            position.pnl = pnl
            position.pnl_percent = (
                (current_price - position.entry_price) / position.entry_price) * 100
        return {
            "balance": self.balance,
            "currency": currency,
            "positions": self.positions,
            "total_positions": len(self.positions),
            "total_exposure": total_exposure,
        }

    def set_leverage(self, symbol: str, leverage: int):
        pass

    def set_margin_mode(self, symbol: str, mode: str):
        pass

    def open_position(self, symbol: str, side: str, size: str, sl_pct: Optional[int]):
        price = self.get_current_price(symbol)
        if size.endswith("%"):
            pct = float(size[:-1]) / 100
            usdt_value = self.balance * pct
        else:
            usdt_value = float(size)
        market = self.exchange.market(symbol)
        contract_size = market['contractSize'] or 1
        qty = self.exchange.amount_to_precision(
            symbol, usdt_value / (price * contract_size))
        # Close existing position if any (paper trading allows only one)
        if self.positions:
            self.flash_close_position(symbol)
        position = Position(
            positionId="paper_001",
            side=side.upper(),
            size=float(qty),
            entry_price=price,
            symbol=symbol
        )
        self.positions = [position]
        # Update balance (deduct position value)
        self.balance -= usdt_value

    def flash_close_position(self, symbol: str):
        if self.positions:
            position = self.positions[0]
            current_price = self.get_current_price(symbol)
            # Calculate PnL and add back to balance
            pnl = (current_price - position.entry_price) * position.size
            position_value = position.size * current_price
            # Return the position value plus PnL to balance
            self.balance += position_value + pnl
            self.positions = []
            logging.info(f"Paper: Closed {position.side} position")

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 15):
        return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
