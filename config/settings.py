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
    POSITION_SIZE: str = "10%"   # "10%", "50%", or 100 (fixed USDT)
    STOP_LOSS_PERCENT: Optional[int] = 10
    FORWARD_TESTING: bool = False
    # Toggle paper vs live
    # FORWARD_TESTING: bool = True
    # INITIAL_CAPITAL: float = 10_000.0
    # TAKER_FEE: float = 0.0006

    @property
    def RUN_NAME(self) -> str:
        return f"run_{self.CRYPTO}_{self.SYMBOL}_{self.LEVERAGE}x"
