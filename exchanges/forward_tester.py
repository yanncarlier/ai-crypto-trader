# exchanges/forward_tester.py
import random
import logging
from typing import Optional
from .base import BaseExchange, Position


class ForwardTester(BaseExchange):
    def __init__(self, config):
        self.config = config
        # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
        # OLD (broken): self.balance = config.INITIAL_CAPITAL
        # NEW (working):
        self.balance = getattr(config, "INITIAL_CAPITAL",
                               10_000.0)  # fallback if missing
        # Or even better — just hardcode it since it's paper trading:
        self.balance = 10_000.0
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        self.position: Optional[Position] = None
        self.price = 65000.0

    def get_current_price(self, symbol: str) -> float:
        self.price *= (1 + random.uniform(-0.008, 0.008))
        self.price = round(max(self.price, 10000), 2)
        logging.info(f"[Paper] {symbol} price: ${self.price:,.2f}")
        return self.price

    def get_account_balance(self, currency: str) -> float:
        return round(self.balance, 2)

    def get_pending_positions(self, symbol: str) -> Optional[Position]:
        return self.position

    def set_leverage(self, symbol: str, leverage: int):
        logging.info(f"[Paper] Set leverage {leverage}x")

    def set_margin_mode(self, symbol: str, mode: str):
        logging.info(f"[Paper] Set margin mode: {mode}")

    def open_position(self, symbol: str, side: str, size: str, sl_pct: Optional[int]):
        price = self.get_current_price(symbol)
        self.position = Position(
            positionId="paper_123",
            side="BUY" if side == "buy" else "SELL",
            size=0.01,
            entry_price=price
        )
        logging.info(f"[Paper] Opened {side.upper()} position @ ${price:,.2f}")

    def flash_close_position(self, position_id: str):
        if self.position:
            logging.info(f"[Paper] Closed position {position_id}")
            self.position = None
