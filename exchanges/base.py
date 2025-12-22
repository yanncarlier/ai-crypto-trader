# exchanges/base.py
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict


class Position:
    def __init__(self, positionId: str, side: str, size: float, entry_price: float, symbol: str, timestamp: int):
        self.positionId = positionId
        self.side = side  # "BUY" or "SELL"
        self.size = size
        self.entry_price = entry_price
        self.symbol = symbol
        self.timestamp = timestamp
        self.pnl: float = 0.0  # Profit/Loss in USD
        self.pnl_percent: float = 0.0  # Profit/Loss percentage

    def __str__(self):
        return f"{self.side} {self.size:.4f} {self.symbol} @ ${self.entry_price:,.2f}"


class BaseExchange(ABC):
    @abstractmethod
    def get_current_price(self, symbol: str) -> float: ...
    @abstractmethod
    def get_account_balance(self, currency: str) -> float: ...
    @abstractmethod
    def get_pending_positions(self, symbol: str) -> Optional[Position]: ...
    @abstractmethod
    def get_all_positions(self, symbol: str) -> List[Position]: ...

    @abstractmethod
    def get_account_summary(self, currency: str,
                            symbol: str) -> Dict[str, Any]: ...

    @abstractmethod
    def set_leverage(self, symbol: str, leverage: int): ...
    @abstractmethod
    def set_margin_mode(self, symbol: str, mode: str): ...

    @abstractmethod
    def open_position(self, symbol: str, side: str,
                      size: str, sl_pct: Optional[float] = None, 
                      tp_pct: Optional[float] = None) -> Dict: ...

    @abstractmethod
    def flash_close_position(self, symbol: str): ...
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int): ...
