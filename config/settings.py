# settings.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class TradingConfig:
    CRYPTO: str = "Bitcoin"
    SYMBOL: str = "BTCUSDT"
    CURRENCY: str = "USDT"
    CYCLE_MINUTES: int = 10
    LEVERAGE: int = 2
    MARGIN_MODE: str = "ISOLATED"  # "ISOLATED" or "CROSS"
    POSITION_SIZE: str = "10%"     # "10%", "50%", or "500" (fixed USDT)
    STOP_LOSS_PERCENT: Optional[int] = 10
    TAKE_PROFIT_PERCENT: Optional[int] = None  # Future use
    FORWARD_TESTING: bool = False
    INITIAL_CAPITAL: float = 10_000.0
    TAKER_FEE: float = 0.0006
    EXCHANGE: str = "BITUNIX"  # "BINANCE" or "BITUNIX"
    TEST_NET: bool = True  # For Binance testnet

    @property
    def RUN_NAME(self) -> str:
        mode = "paper" if self.FORWARD_TESTING else "LIVE"
        return f"{mode}_{self.EXCHANGE}_{self.CRYPTO}_{self.SYMBOL}_{self.LEVERAGE}x"
