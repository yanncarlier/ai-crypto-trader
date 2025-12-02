# config/settings.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class TradingConfig:
    # Trading Configuration
    CRYPTO: str = "Bitcoin"
    SYMBOL: str = "BTCUSDT"
    CURRENCY: str = "USDT"
    CYCLE_MINUTES: int = 10  # Dynamic cycle length - you can change this to any value
    LEVERAGE: int = 2
    MARGIN_MODE: str = "ISOLATED"  # "ISOLATED" or "CROSS"
    POSITION_SIZE: str = "10%"     # "10%", "50%", or "500" (fixed USDT)
    STOP_LOSS_PERCENT: Optional[int] = 10
    TAKE_PROFIT_PERCENT: Optional[int] = None
    INITIAL_CAPITAL: float = 10_000.0
    TAKER_FEE: float = 0.0006
    # Configuration
    FORWARD_TESTING: bool = False  # true for paper trading
    # "xai", "groq", "openai", "openrouter", "deepseek", "mistral"
    LLM_PROVIDER: str = "xai"
    LLM_MODEL: str = "default"  # Will use provider defaults if set to "default"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 800
    EXCHANGE: str = "BITUNIX"  # Only Bitunix now
    TEST_NET: bool = False  # Not applicable for Bitunix

    @property
    def RUN_NAME(self) -> str:
        mode = "paper" if self.FORWARD_TESTING else "LIVE"
        return f"{mode}_{self.EXCHANGE}_{self.CRYPTO}_{self.SYMBOL}_{self.CYCLE_MINUTES}min_{self.LEVERAGE}x"
